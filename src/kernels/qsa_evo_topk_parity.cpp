// Synthetic HIP test, no model. Not a P2 server/cancel/snapshot qualification.
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace k = strata::kernels;
static void ck(cudaError_t e, const char* action) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(e)); std::exit(2); }
}
struct Fixture {
    static constexpr int qcap = 16, blocks = 32770;
    k::QsaShapes shapes = k::qsa_real_shapes();
    int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, shapes);
    float* scores = nullptr;
    int32_t *steps = nullptr, *ids = nullptr, *reference = nullptr;
    k::QsaTopkScratch scratch;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaEvent_t done = nullptr;
    int nq = 0;
    std::vector<int32_t> hs;
    Fixture() {
        scratch.capacity = k::evo_topk::layout(qcap, blocks);
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        ck(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "event");
        ck(cudaMalloc(&scores, std::size_t(qcap) * blocks * sizeof(float)), "scores");
        ck(cudaMalloc(&steps, qcap * k::kStepCount * sizeof(int32_t)), "steps");
        ck(cudaMalloc(&ids, std::size_t(qcap) * cap * sizeof(int32_t)), "ids");
        ck(cudaMalloc(&reference, std::size_t(qcap) * cap * sizeof(int32_t)), "reference");
        ck(cudaMalloc(&scratch.data, scratch.capacity.bytes + 256), "scratch with guard");
    }
    void discard_graph() {
        ck(cudaStreamSynchronize(stream), "before graph reuse");
        if (exec) ck(cudaGraphExecDestroy(exec), "exec destroy");
        if (graph) ck(cudaGraphDestroy(graph), "graph destroy");
        exec = nullptr; graph = nullptr;
    }
    ~Fixture() {
        discard_graph();
        cudaEventDestroy(done); cudaStreamDestroy(stream);
        cudaFree(scores); cudaFree(steps); cudaFree(ids); cudaFree(reference); cudaFree(scratch.data);
    }
    void load(int context, int queries, int kind) {
        // Completion fence before input/scratch reuse, analogous to the required slot
        // reuse contract. This does not pretend to exercise the server cancel API.
        ck(cudaStreamSynchronize(stream), "before input reuse");
        nq = queries;
        hs.assign(qcap * k::kStepCount, 0);
        std::vector<float> sc(std::size_t(qcap) * blocks, 0.f);
        for (int q = 0; q < nq; ++q) {
            int32_t* st = hs.data() + q * k::kStepCount;
            const int n = context - nq + q + 1;
            st[k::kStepPos] = n - 1; st[k::kStepNKv] = n; st[k::kStepNBid] = n / 4;
            st[k::kStepWidth] = int(k::qsa_selection_width(n, shapes));
            for (int b = 0; b < blocks; ++b) {
                float value = float((b * 31 + q * 13) % 67 - 33);
                if (kind == 1) value = .25f;
                if (kind == 2) value = (b & 1) ? -0.f : 0.f;
                if (kind == 3) { const float special[] = {NAN, -NAN, INFINITY, -INFINITY, 0.f, -0.f, 1.f}; value = special[(b + q) % 7]; }
                if (kind == 4) value = NAN;
                sc[std::size_t(q) * blocks + b] = value;
            }
        }
        ck(cudaMemcpyAsync(scores, sc.data(), sc.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "scores upload");
        ck(cudaMemcpyAsync(steps, hs.data(), hs.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "steps upload");
        ck(cudaMemsetAsync(scratch.data, 0xa5, scratch.capacity.bytes + 256, stream), "poison scratch/guard");
        ck(cudaMemsetAsync(ids, 0xff, std::size_t(qcap) * cap * sizeof(int32_t), stream), "poison ids");
        ck(cudaMemsetAsync(reference, 0xff, std::size_t(qcap) * cap * sizeof(int32_t), stream), "poison reference");
        ck(cudaStreamSynchronize(stream), "finish pageable uploads");
    }
    void launch() {
        if (!k::qsa_block_topk_evo_split(scores, steps, nq, blocks, cap, shapes, ids, stream, scratch)) {
            std::fprintf(stderr, "FAIL eligible synthetic split call was rejected\n"); std::exit(1);
        }
    }
    void capture() {
        discard_graph();
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin");
        launch();
        ck(cudaStreamEndCapture(stream, &graph), "end");
        ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    }
    void replay() { ck(cudaGraphLaunch(exec, stream), "graph launch"); ck(cudaEventRecord(done, stream), "record reuse fence"); }
    bool check() {
        ck(cudaEventSynchronize(done), "reuse fence");
        k::qsa_block_topk_ref(scores, steps, nq, blocks, cap, shapes, reference, stream);
        ck(cudaStreamSynchronize(stream), "reference complete");
        std::vector<int32_t> actual(std::size_t(qcap) * cap), expected(actual.size());
        std::vector<unsigned char> guard(256);
        ck(cudaMemcpy(actual.data(), ids, actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "actual");
        ck(cudaMemcpy(expected.data(), reference, expected.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "reference");
        ck(cudaMemcpy(guard.data(), reinterpret_cast<unsigned char*>(scratch.data) + scratch.capacity.bytes,
                      guard.size(), cudaMemcpyDeviceToHost), "guard");
        return actual == expected && std::all_of(guard.begin(), guard.end(), [](unsigned char c) { return c == 0xa5; });
    }
};
int main() {
    if (!k::qsa_evo_topk_device_supported()) {
        std::puts("SKIP: requires Linux HIP, exact gfx1151 integrated wave32, one visible device, no arch override");
        return 77;
    }
    Fixture a, b;
    if (a.scratch.data == b.scratch.data) return 1;
    // Allocated Q=16/C=17 with active Q=1/2/16 catches accidental active-query stride.
    // Each context has different inputs and scratch; both graphs enqueue before waiting.
    int cases = 0;
    for (int q : {1, 2, 16}) {
        for (int ctx : {64, 8191, 8192, 8193, 65535, 65536, 65537, 98303, 98304, 98305,
                        114687, 114688, 114689, 131071}) {
            a.load(ctx, q, 0); b.load(ctx - 1, q, 3);
            a.capture(); b.capture();
            for (int kind = 0; kind < 5; ++kind) {
                a.load(ctx, q, kind); b.load(ctx - 1, q, 4 - kind);
                a.replay(); b.replay();
                if (!a.check() || !b.check()) {
                    std::fprintf(stderr, "FAIL context=%d Q=%d kind=%d\n", ctx, q, kind); return 1;
                }
                ++cases;
            }
        }
    }
    // Rejections must happen before memset/kernel enqueue.
    if (k::qsa_block_topk_evo_split(a.scores, a.steps, 17, Fixture::blocks, a.cap, a.shapes,
                                   a.ids, a.stream, a.scratch)) return 1;
    auto invalid = a.scratch; invalid.data++;
    if (k::qsa_block_topk_evo_split(a.scores, a.steps, 1, Fixture::blocks, a.cap, a.shapes,
                                   a.ids, a.stream, invalid)) return 1;
    std::printf("PASS %d two-context graph replay cases, exact IDs/order and scratch guards\n", cases);
    return 0;
}

// Adapted from nicejunjie/Strata-rdna3 at f557b5440fe345757cae69ede2ed34826da5e612,
// src/kernels/cuda/qsa_select.cu, split top-k section. MIT, copyright (c) 2026
// Niko1221 and the Strata contributors; see LICENSE. Only the six-pass algorithm is
// retained. All device-global storage/capture overrides were removed. Captured calls
// carry context-owned pointers and the ALLOCATED query/chunk strides, not active nq.
// Included inside qsa_select.cu's anonymous namespace, after order_key.
#if defined(__HIPCC__) && defined(__linux__)
namespace evo_split {
constexpr int SP_T = 256, SP_W = SP_T / 32, SP_CH = evo_topk::kChunkBlocks;
struct SplitScratch {
    int32_t *hist, *state, *count;
    int queries, chunks;
    __device__ int32_t* h(int q, int pass) const { return hist + (q * 4 + pass) * 256; }
    __device__ int32_t* s(int pass, int q) const { return state + (pass * queries + q) * 2; }
    __device__ int32_t* c(int q, int chunk) const { return count + (q * chunks + chunk) * 2; }
};

// the digit of pass `p` from its complete histogram: block_topk_kernel's scan as a parallel suffix sum
__device__ void evo_sp_digit(const int* __restrict__ h, int width, int above, int& digit, int& above_out, int* s_buf) {
    const int t = threadIdx.x;
    // s_buf[d] = sum of h[d'] for d' > d (exclusive suffix), d in 0..255
    const int lane = t & 31, w = t >> 5;
    __shared__ int s_w[SP_W];
    // scan in reverse order: thread t holds element r = 255 - t
    int e = h[255 - t];
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, e, o); if (lane >= o) e += y; }
    if (lane == 31) s_w[w] = e;
    __syncthreads();
    int add = 0;
    for (int i = 0; i < w; ++i) add += s_w[i];
    e += add;                                      // inclusive suffix sum of h from 255 down to r = 255 - t
    s_buf[255 - t] = e;                            // s_buf[r] = sum_{d >= r} h[d]
    __syncthreads();
    // the scan picks the largest d in 1..255 with above + sum_{d' >= d} h >= width, else 0
    __shared__ int s_best;
    if (t == 0) s_best = 0;
    __syncthreads();
    if (t >= 1 && above + s_buf[t] >= width) atomicMax(&s_best, t);
    __syncthreads();
    const int d = s_best;
    digit = d;
    above_out = above + (d == 255 ? 0 : s_buf[d + 1]);   // cells strictly above digit d (the scan's cum)
    if (d == 0) above_out = above + s_buf[1];
    __syncthreads();
}

template <int P>
__global__ void __launch_bounds__(SP_T) evo_topk_split_hist_kernel(const float* __restrict__ scores,
                                                               const int32_t* __restrict__ steps, int64_t max_blocks, SplitScratch scratch) {
    __shared__ int hist[SP_W][256];
    __shared__ int s_buf[256];
    const int q = blockIdx.y, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int32_t* st = steps + (int64_t) q * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    if (n_kv <= width) return;
    const int64_t nb = n_bid + 1, b0 = (int64_t) blockIdx.x * SP_CH;
    uint32_t prefix = 0;
    int above = 0;
    if (P > 0) {
        const int pa = scratch.s(P - 1, q)[0], ab = scratch.s(P - 1, q)[1];
        int d, ab2;
        evo_sp_digit(scratch.h(q, P - 1), (int) width, ab, d, ab2, s_buf);
        prefix = (uint32_t) pa | ((uint32_t) d << (24 - 8 * (P - 1)));
        above = ab2;
        if (blockIdx.x == 0 && t == 0) { scratch.s(P, q)[0] = (int) prefix; scratch.s(P, q)[1] = above; }
    } else if (blockIdx.x == 0 && t == 0) {
        scratch.s(0, q)[0] = 0; scratch.s(0, q)[1] = 0;
    }
    if (P == 4 || b0 >= nb) return;
    constexpr int shift = 24 - 8 * (P < 4 ? P : 0);
    uint32_t hi_mask = 0u;
    if constexpr (P > 0) hi_mask = 0xffffffffu << (shift + 8);
    for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
    __syncthreads();
    const float* sc = scores + (int64_t) q * max_blocks;
    const int64_t b1 = b0 + SP_CH < nb ? b0 + SP_CH : nb;
    for (int64_t b = b0 + t; b < b1; b += SP_T) {
        const int w = b < n_bid ? R : (int) (n_kv - n_bid * R);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(k >> shift) & 255], w);
    }
    __syncthreads();
    int s = 0;
    for (int w2 = 0; w2 < SP_W; ++w2) s += hist[w2][t];
    if (s) atomicAdd(&scratch.h(q, P)[t], s);
}

// the threshold from pass 3's histogram, then each chunk's cells above / at it
__global__ void __launch_bounds__(SP_T) evo_topk_split_count_kernel(const float* __restrict__ scores,
                                                                const int32_t* __restrict__ steps, int64_t max_blocks, SplitScratch scratch) {
    __shared__ int s_buf[256];
    __shared__ int s_r[2][SP_W];
    const int q = blockIdx.y, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int32_t* st = steps + (int64_t) q * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    if (n_kv <= width) return;
    const int pa = scratch.s(3, q)[0], ab = scratch.s(3, q)[1];
    int d, ab2;
    evo_sp_digit(scratch.h(q, 3), (int) width, ab, d, ab2, s_buf);
    const uint32_t thr = (uint32_t) pa | (uint32_t) d;
    if (blockIdx.x == 0 && t == 0) { scratch.s(4, q)[0] = (int) thr; scratch.s(4, q)[1] = ab2; }
    const int64_t nb = n_bid + 1, b0 = (int64_t) blockIdx.x * SP_CH;
    const int64_t b1 = b0 + SP_CH < nb ? b0 + SP_CH : nb;
    const float* sc = scores + (int64_t) q * max_blocks;
    int gt = 0, eq = 0;
    for (int64_t b = b0 + t; b < b1; b += SP_T) {
        const int w = b < n_bid ? R : (int) (n_kv - n_bid * R);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) { s_r[0][warp] = gt; s_r[1][warp] = eq; }
    __syncthreads();
    if (t == 0) {
        int g2 = 0, e2 = 0;
        for (int w2 = 0; w2 < SP_W; ++w2) { g2 += s_r[0][w2]; e2 += s_r[1][w2]; }
        scratch.c(q, blockIdx.x)[0] = g2;
        scratch.c(q, blockIdx.x)[1] = e2;
    }
}

// the emit: block_topk_wide_kernel's within a chunk, offset by the chunks before it
__global__ void __launch_bounds__(SP_T) evo_topk_split_emit_kernel(const float* __restrict__ scores,
                                                               const int32_t* __restrict__ steps, int64_t max_blocks,
                                                               int64_t cap, int32_t* __restrict__ ids, SplitScratch scratch) {
    __shared__ int s_gt[SP_W], s_eq[SP_W];
    __shared__ int s_sel0, s_eqb0;
    const int q = blockIdx.y, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int32_t* st = steps + (int64_t) q * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + (int64_t) q * cap;
    const int64_t nb = n_bid + 1, b0 = (int64_t) blockIdx.x * SP_CH;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        const int64_t c0 = b0 * R, c1 = (b0 + SP_CH) * R < n_kv ? (b0 + SP_CH) * R : n_kv;
        for (int64_t j = c0 + t; j < c1; j += SP_T) out[j] = (int32_t) j;
        return;
    }
    if (b0 >= nb) return;
    const uint32_t thr = (uint32_t) scratch.s(4, q)[0];
    const int eq_budget = (int) (width - scratch.s(4, q)[1]);
    if (t == 0) {                                      // the cells the chunks before this one select / hold at thr
        int sel = 0, eqb = 0;
        for (int c = 0; c < (int) blockIdx.x; ++c) {
            const int g2 = scratch.c(q, c)[0], e2 = scratch.c(q, c)[1];
            const int take = eq_budget - eqb < 0 ? 0 : (eq_budget - eqb > e2 ? e2 : eq_budget - eqb);
            sel += g2 + take;
            eqb += e2;
        }
        s_sel0 = sel;
        s_eqb0 = eqb;
    }
    const int64_t b1 = b0 + SP_CH < nb ? b0 + SP_CH : nb;
    const int64_t per = (SP_CH / 32) / SP_W;           // rows of 32 blocks per warp
    const int64_t w0 = b0 + (int64_t) warp * 32 * per, w1 = w0 + 32 * per < b1 ? w0 + 32 * per : b1;
    const float* sc = scores + (int64_t) q * max_blocks;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    int gt = 0, eq = 0;
    for (int64_t b = w0 + lane; b < w1; b += 32) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    __syncthreads();
    if (t == 0) {
        int eb = s_eqb0, sb = s_sel0;
        for (int w2 = 0; w2 < SP_W; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb;
            s_eq[w2] = eb;
            sb += g2 + take;
            eb += e2;
        }
    }
    __syncthreads();
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    for (int64_t r0 = w0; r0 < w1; r0 += 32) {
        const int64_t b = r0 + lane;
        const int w = b < w1 ? weight(b) : 0;
        const uint32_t k = w ? order_key(sc[b]) : 0u;
        const int my_gt = (w && k > thr) ? w : 0, my_eq = (w && k == thr) ? w : 0;
        if (__ballot_sync(0xffffffffu, (my_gt | my_eq) != 0) == 0u) continue;
        int pe = my_eq;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, pe, o);
            if (lane >= o) pe += y;
        }
        const int left = eq_budget - (run_eq + pe - my_eq);
        const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
        const int my_sel = my_gt + take;
        int ps = my_sel;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, ps, o);
            if (lane >= o) ps += y;
        }
        int32_t* dst = out + run_sel + ps - my_sel;
        for (int c = 0; c < my_sel; ++c) dst[c] = (int32_t) (b * R + c);
        run_eq += __shfl_sync(0xffffffffu, pe, 31);
        run_sel += __shfl_sync(0xffffffffu, ps, 31);
    }
}

}  // namespace evo_split
#endif

"""Explicit synthetic replay and optional credential-free loopback SSE capture.

No network activity occurs on import, during replay, or in the default mode.
The opt-in HTTP adapter supports only literal loopback IPs and one pinned route.
It never starts/stops a service, follows redirects, reads credentials or proxies,
or treats API finish_reason=stop as proof of an engine's normal completion.
"""

import argparse
import concurrent.futures
import http.client
import ipaddress
import json
import re
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .contract import finite, output_limit
from .fixtures import strict_json
from .sse import StreamCapture
from .workloads import frozen_digest

ROUTE = "/v1/chat/completions"
PAYLOAD_KEYS = {"model", "messages", "stream", "max_tokens", "max_completion_tokens",
                "temperature", "top_p", "top_k", "presence_penalty", "seed", "stop",
                "chat_template_kwargs", "stream_options"}


def prepare_request(request):
    """Freeze the exact wire object without adding or changing sampling fields."""
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    name, payload, target = (request.get(key) for key in ("id", "payload", "target_input_tokens"))
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name) is None:
        raise ValueError("request id must be a short ASCII identifier")
    if not isinstance(payload, dict) or set(payload) - PAYLOAD_KEYS:
        raise ValueError("unsupported request fields; credentials, tools, MCP and metadata are not accepted")
    if payload.get("stream") is not True or output_limit(payload) != 512:
        raise ValueError("capture requires an explicit streamed 512-output-token request")
    if not any(payload.get(key) == 512 for key in ("max_tokens", "max_completion_tokens")):
        raise ValueError("explicit 512 output limit required")
    if not isinstance(payload.get("model"), str) or not payload["model"]:
        raise ValueError("an explicit model alias is required")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("nonempty text-only messages required")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant") or not isinstance(message["content"], str)):
            raise ValueError("only plain role/content messages are supported")
    if "chat_template_kwargs" in payload and payload["chat_template_kwargs"] != {"enable_thinking": False}:
        raise ValueError("only explicit no-thinking template kwargs are supported")
    if "stream_options" in payload and payload["stream_options"] != {"include_usage": True}:
        raise ValueError("only include_usage=true stream options are supported")
    if type(target) is not int or not 0 <= target <= 130552:
        raise ValueError("invalid requested final input token count")
    wire = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(wire) > 8 * 1024 * 1024:
        raise ValueError("request exceeds 8 MiB bound")
    return {"id": name, "payload": payload, "wire": wire, "request_sha256": frozen_digest(payload),
            "target_input_tokens": target, "requested_output_tokens": 512}


@dataclass
class StreamResponse:
    status: int
    content_type: str
    frames: object
    close: object


class FakeStreamTransport:
    """Replay timestamped fictional byte chunks; no socket, sleep or HTTP call."""

    evidence_kind = "synthetic_test"

    def __init__(self, frames, *, status=200, content_type="text/event-stream"):
        self.frames, self.status, self.content_type = frames, status, content_type
        self.closed = False

    def open(self, request, send_s):
        def frames():
            for frame in self.frames:
                offset, data = frame.get("at_s"), frame.get("data")
                if not finite(offset) or offset < 0 or not isinstance(data, (str, bytes)):
                    raise ValueError("invalid synthetic frame")
                yield send_s + offset, data.encode("utf-8") if isinstance(data, str) else data
        return StreamResponse(self.status, self.content_type, frames(), lambda: setattr(self, "closed", True))


def loopback_endpoint(endpoint):
    """Validate without DNS: no hostname, userinfo, redirect, query or fragment."""
    if not isinstance(endpoint, str) or any(character.isspace() for character in endpoint):
        raise ValueError("endpoint must be an explicit loopback URL")
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.username is not None or parsed.password is not None
            or parsed.path != ROUTE or parsed.query or parsed.fragment):
        raise ValueError("only credential-free HTTP loopback /v1/chat/completions is allowed")
    try:
        host = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError as exc:
        raise ValueError("endpoint must use a literal loopback IP and valid port") from exc
    if not host.is_loopback or port is None or not 1 <= port <= 65535 or "%" in str(host):
        raise ValueError("literal loopback IP and explicit port are required")
    return str(host), port


class LoopbackHTTPTransport:
    """Disabled by default. Construction only validates; open is the sole I/O step."""

    evidence_kind = "captured"

    def __init__(self, endpoint, *, enabled=False, socket_timeout_s=30, maximum_duration_s=1800):
        self.host, self.port = loopback_endpoint(endpoint)
        if not all(finite(value) and value > 0 for value in (socket_timeout_s, maximum_duration_s)):
            raise ValueError("positive finite timeouts are required")
        self.enabled = enabled is True
        self.socket_timeout_s, self.maximum_duration_s = socket_timeout_s, maximum_duration_s

    def open(self, request, send_s):
        if not self.enabled:
            raise ValueError("loopback networking requires explicit opt-in")
        deadline = send_s + self.maximum_duration_s
        expired = threading.Event()

        def timeout(stamp=None):
            remaining = deadline - (time.monotonic() if stamp is None else stamp)
            if remaining <= 0 or expired.is_set():
                raise TimeoutError("maximum capture duration exceeded")
            return min(self.socket_timeout_s, remaining)

        # http.client does not consult HTTP_PROXY, netrc, browser or credential stores.
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout())
        response = connected_socket = watchdog = None
        closed = False

        def close():
            nonlocal closed
            if closed:
                return
            closed = True
            if watchdog is not None:
                watchdog.cancel()
                watchdog.join()
            try:
                if response is not None:
                    response.close()
            finally:
                try:
                    connection.close()
                finally:
                    if connected_socket is not None:
                        connected_socket.close()

        try:
            # Bound connect separately, then retain the socket: getresponse() can
            # detach it from the connection when the server sends Connection: close.
            connection.connect()
            connected_socket = connection.sock
            connected_socket.settimeout(timeout())

            def expire():
                expired.set()
                try:
                    connected_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # It may already have reached EOF or been closed.

            # Socket idle timeouts alone do not bound trickling headers or chunk
            # framing. Shutdown interrupts those reads without a lingering worker.
            timer = threading.Timer(max(0, deadline - time.monotonic()), expire)
            timer.daemon = True
            timer.start()
            watchdog = timer
            connected_socket.settimeout(timeout())
            connection.request("POST", ROUTE, body=request["wire"],
                               headers={"Content-Type": "application/json", "Accept": "text/event-stream", "Connection": "close"})
            connected_socket.settimeout(timeout())
            response = connection.getresponse()
            content_type = response.getheader("Content-Type", "")
            timeout()
        except Exception:
            try:
                timeout()  # Normalize a watchdog-induced EOF/reset to a timeout.
            finally:
                close()
            raise
        except BaseException:
            close()
            raise

        def frames():
            try:
                if response.status != 200:
                    return  # Do not retain an authentication page, redirect body or error text.
                while True:
                    timeout()
                    try:
                        data = response.read1(65536)
                    except Exception:
                        timeout()
                        raise
                    stamp = time.monotonic()
                    timeout(stamp)  # Never accept bytes or EOF received past the deadline.
                    if not data:
                        return
                    yield stamp, data
            finally:
                close()
        return StreamResponse(response.status, content_type, frames(), close)


def _capture(request, transport, send_s):
    response = None
    # Until response headers arrive, the HTTP status is unknown, not 200.
    capture = StreamCapture(send_s, status_code=None, content_type="")
    error = None
    try:
        response = transport.open(request, send_s)
        capture = StreamCapture(send_s, response.status, response.content_type)
        for stamp, data in response.frames:
            capture.feed(data, stamp)
            if capture.done_s is not None:
                break  # Do not wait for HTTP close after the SSE terminator.
    except (ValueError, UnicodeError) as exc:
        error = "invalid_stream:" + type(exc).__name__
    except Exception as exc:
        error = "transport_failure:" + type(exc).__name__
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                error = error or "transport_close_failure"
    result = capture.finish(transport_error=error)
    result.update(client_request_id=request["id"], request_sha256=request["request_sha256"],
                  target_input_tokens=request["target_input_tokens"],
                  requested_output_tokens=request["requested_output_tokens"],
                  evidence_kind=transport.evidence_kind, dispatch_mode="barrier")
    return result


def capture_pair(requests, transport_factory, *, barrier_timeout_s=5, clock=None):
    """Synchronize two sends, then capture independently. No automatic retries."""
    if not isinstance(requests, list) or len(requests) != 2:
        raise ValueError("exactly two synthetic requests are required")
    prepared = [prepare_request(request) for request in requests]
    if len({request["id"] for request in prepared}) != 2 or len({request["request_sha256"] for request in prepared}) != 2:
        raise ValueError("distinct request IDs and distinct request content required")
    if len({request["target_input_tokens"] for request in prepared}) != 1:
        raise ValueError("paired requests require the same target input length")
    if not finite(barrier_timeout_s) or barrier_timeout_s <= 0:
        raise ValueError("invalid barrier timeout")
    # Prepare adapters before the barrier. A construction error cannot strand a worker.
    transports = [transport_factory(request) for request in prepared]
    kinds = {transport.evidence_kind for transport in transports}
    if len(kinds) != 1 or not kinds <= {"synthetic_test", "captured"}:
        raise ValueError("mixed or unknown evidence kinds")
    if clock is not None and kinds != {"synthetic_test"}:
        raise ValueError("injected clocks are only allowed for synthetic tests")
    clock_source = "python_time_monotonic" if clock is None else "synthetic_injected_clock"
    clock = time.monotonic if clock is None else clock
    barrier = threading.Barrier(3)

    def worker(request, transport):
        barrier.wait(timeout=barrier_timeout_s)
        return _capture(request, transport, clock())

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, request, transport) for request, transport in zip(prepared, transports)]
        barrier.wait(timeout=barrier_timeout_s)
        records = [future.result() for future in futures]
    kind = next(iter(kinds))
    return {"schema_version": 1, "run_id": uuid.uuid4().hex, "clock_id": uuid.uuid4().hex,
            "clock_source": clock_source, "evidence_kind": kind,
            "records": records, "dispatch_mode": "barrier",
            "start_skew_s": abs(records[0]["send_s"] - records[1]["send_s"]),
            "api_streams_complete": all(record["api_stream_complete"] for record in records),
            "correlation_complete": False, "production_qualified": False,
            "note": "No engine request binding, final tokenization or slot/cache evidence has been inferred."}


def read_document(path):
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("input exceeds 64 MiB")
    return strict_json(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "loopback", "correlate"))
    parser.add_argument("input", type=Path, help="explicit local synthetic bundle or capture JSON")
    parser.add_argument("--engine-evidence", type=Path)
    parser.add_argument("--endpoint", help="explicit literal-loopback URL, no credentials")
    parser.add_argument("--enable-loopback", action="store_true", help="explicitly permit requests to an independently prepared local test server")
    args = parser.parse_args(argv)
    try:
        bundle = read_document(args.input)
        if args.mode == "correlate":
            if args.engine_evidence is None or args.endpoint is not None or args.enable_loopback:
                raise ValueError("correlate needs only capture JSON and --engine-evidence")
            from .correlation import correlate_pair
            result = correlate_pair(bundle, read_document(args.engine_evidence))
            passed = result["correlation_complete"]
        else:
            if (not isinstance(bundle, dict) or type(bundle.get("schema_version")) is not int
                    or bundle["schema_version"] != 1 or bundle.get("synthetic_only") is not True):
                raise ValueError("only explicit schema-v1 synthetic-only request bundles are accepted")
            if args.engine_evidence is not None:
                raise ValueError("correlate engine evidence separately after capture")
            if args.mode == "loopback":
                if args.endpoint is None or not args.enable_loopback:
                    raise ValueError("loopback mode requires --endpoint and --enable-loopback")
                factory = lambda request: LoopbackHTTPTransport(args.endpoint, enabled=True)
            else:
                if args.endpoint is not None or args.enable_loopback:
                    raise ValueError("replay never accepts networking options")
                streams = bundle["streams"]
                factory = lambda request: FakeStreamTransport(streams[request["id"]])
            result = capture_pair(bundle["requests"], factory)
            passed = result["api_streams_complete"]
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
        sys.stdout.write("\n")
        return 0 if passed else 1
    except (OSError, ValueError, TypeError, KeyError, AttributeError, threading.BrokenBarrierError) as exc:
        print(f"invalid capture input: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

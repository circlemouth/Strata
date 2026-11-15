import copy
import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from bench.evo_x2.correlation import correlate_pair
from bench.evo_x2.runner import (FakeStreamTransport, LoopbackHTTPTransport,
                                 _capture, capture_pair, loopback_endpoint, prepare_request)
from bench.evo_x2.sse import SSEDecoder, StreamCapture


def event(delta=None, *, finish=None, usage=None, identity="chatcmpl-synthetic"):
    value = {"id": identity, "model": "synthetic-model", "object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": {} if delta is None else delta, "finish_reason": finish}]}
    if usage is not None:
        value["usage"] = usage
    return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"


def exact_usage(**changes):
    return dict({"prompt_tokens": 4096, "completion_tokens": 512, "total_tokens": 4608,
                 "prompt_tokens_details": {"cached_tokens": 0}}, **changes)


def frames(identity="chatcmpl-synthetic"):
    return [{"at_s": .01, "data": ": keep-alive\n\n"},
            {"at_s": .02, "data": event({"role": "assistant", "content": ""}, identity=identity)},
            {"at_s": .03, "data": event({"reasoning_content": "架空の推論"}, identity=identity)},
            {"at_s": .04, "data": event({"content": "架空の回答"}, identity=identity)},
            {"at_s": 1.0, "data": event(finish="length", usage=exact_usage(), identity=identity)},
            {"at_s": 1.1, "data": "data: [DONE]\n\n"}]


def request(name="left"):
    return {"id": name, "target_input_tokens": 4096,
            "payload": {"model": "synthetic-model", "messages": [{"role": "user", "content": f"架空テスト {name}"}],
                        "stream": True, "max_tokens": 512, "temperature": 0, "seed": 1}}


def captured_pair():
    return capture_pair([request("left"), request("right")],
                        lambda req: FakeStreamTransport(frames("chatcmpl-" + req["id"])), clock=lambda: 100.0)


def engine_sidecar(capture):
    manifest = "a" * 64
    rows = []
    for slot, record in enumerate(capture["records"]):
        engine_id = "engine-" + record["client_request_id"]
        rows.append({"binding": {"method": "instrumented_request_binding",
                                 "client_request_id": record["client_request_id"],
                                 "request_sha256": record["request_sha256"], "api_response_id": record["response_id"],
                                 "engine_request_id": engine_id},
                     "tokenization": {"source": "synthetic_tokenizer", "request_sha256": record["request_sha256"],
                                      "manifest_sha256": manifest, "includes_template_and_expansion": True,
                                      "encoded_input_tokens": 4096, "tokenizer_sha256": "b" * 64,
                                      "template_sha256": "c" * 64},
                     "engine": {"request_id": engine_id, "slot": slot, "batch_slots": 2,
                                "context_tokens": 131072, "cache_n": 0, "completion_tokens": 512,
                                "terminal_status": "completed", "resident": True, "cancelled": False,
                                "disconnected": False, "truncated": False, "output_limit_adjusted": False,
                                "prefill_started_s": record["send_s"] + .001,
                                "token_timestamps_s": [record["send_s"] + .025 + i * .001 for i in range(512)]}})
    return {"schema_version": 1, "run_id": capture["run_id"], "clock_id": capture["clock_id"],
            "evidence_kind": "synthetic_test", "clock_binding": "shared_monotonic_domain",
            "manifest_sha256": manifest, "configured_parallel": 2, "props_total_slots": 2,
            "fit_max_tokens": False, "records": rows}


class SSEParsing(unittest.TestCase):
    def test_utf8_split_at_every_byte(self):
        raw = (event({"content": "左、右。"}) + event(finish="length", usage=exact_usage()) + "data: [DONE]\n\n").encode()
        capture = StreamCapture(0)
        for i, byte in enumerate(raw):
            capture.feed(bytes([byte]), i / 1000)
        result = capture.finish()
        self.assertTrue(result["api_stream_complete"], result["errors"])
        self.assertEqual(result["content_sha256"], hashlib.sha256("左、右。".encode()).hexdigest())

    def test_crlf_split_between_carriage_return_and_linefeed(self):
        decoder = SSEDecoder()
        self.assertEqual(decoder.feed(b"data: one\r"), [])
        self.assertEqual(decoder.feed(b"\n\r"), [])
        self.assertEqual(decoder.feed(b"\n"), ["one"])

    def test_comments_and_multiline_data(self):
        decoder = SSEDecoder()
        self.assertEqual(decoder.feed(b": ping\n\nevent: message\ndata: first\ndata: second\n\n"), ["first\nsecond"])

    def test_bom_at_stream_start(self):
        decoder = SSEDecoder()
        self.assertEqual(decoder.feed("\ufeffdata: first\n\n".encode()), ["first"])

    def test_missing_blank_line_is_not_completion(self):
        capture = StreamCapture(0)
        capture.feed(b"data: [DONE]", 1)
        result = capture.finish()
        self.assertFalse(result["api_stream_complete"])
        self.assertIn("missing_done", result["errors"])

    def test_invalid_utf8_rejected(self):
        with self.assertRaises(UnicodeError):
            SSEDecoder().feed(b"data: \xff\n\n")

    def test_bounded_event_and_total_stream(self):
        with self.assertRaises(ValueError):
            SSEDecoder(max_event_bytes=4).feed(b"data: larger\n\n")
        with self.assertRaises(ValueError):
            SSEDecoder(max_total_bytes=3).feed(b"data")

    def test_multiple_json_events_in_one_network_read(self):
        capture = StreamCapture(0)
        raw = "".join(row["data"] for row in frames()).encode()
        capture.feed(raw, 1)
        result = capture.finish()
        self.assertTrue(result["api_stream_complete"])
        self.assertEqual(result["ttft_api_s"], 1)
        self.assertEqual(len(set(result["sse_event_times_s"])), 1)
        self.assertIsNone(result["engine_token_intervals"])

    def test_role_reasoning_and_answer_have_distinct_meanings(self):
        capture = StreamCapture(10)
        for row in frames():
            capture.feed(row["data"].encode(), 10 + row["at_s"])
        result = capture.finish()
        self.assertAlmostEqual(result["ttft_any_s"], .03)
        self.assertAlmostEqual(result["ttft_api_s"], .04)
        self.assertFalse(result["normal_completion"])

    def test_usage_is_exact_not_counted_from_chunks(self):
        capture = StreamCapture(0)
        for row in frames():
            capture.feed(row["data"].encode(), row["at_s"])
        result = capture.finish()
        self.assertEqual(result["usage"]["completion_tokens"], 512)
        self.assertEqual(result["usage"]["cached_tokens"], 0)
        self.assertLess(len(result["sse_event_times_s"]), 512)

    def test_wrong_usage_total_rejected(self):
        with self.assertRaises(ValueError):
            StreamCapture(0).feed(event(finish="length", usage=exact_usage(total_tokens=4607)).encode(), 1)

    def test_boolean_usage_count_rejected(self):
        with self.assertRaises(ValueError):
            StreamCapture(0).feed(event(usage=exact_usage(completion_tokens=True)).encode(), 1)

    def test_response_identity_change_rejected(self):
        capture = StreamCapture(0)
        capture.feed(event(identity="one").encode(), 1)
        with self.assertRaises(ValueError):
            capture.feed(event(identity="two").encode(), 2)

    def test_duplicate_usage_rejected(self):
        capture = StreamCapture(0)
        capture.feed(event(usage=exact_usage()).encode(), 1)
        with self.assertRaises(ValueError):
            capture.feed(event(usage=exact_usage()).encode(), 2)

    def test_post_finish_content_rejected(self):
        capture = StreamCapture(0)
        capture.feed(event(finish="stop").encode(), 1)
        with self.assertRaises(ValueError):
            capture.feed(event({"content": "late"}).encode(), 2)

    def test_error_event_even_with_done_is_failure(self):
        capture = StreamCapture(0)
        capture.feed(b'data: {"error":{"message":"synthetic failure"}}\n\ndata: [DONE]\n\n', 1)
        result = capture.finish()
        self.assertFalse(result["api_stream_complete"])
        self.assertIn("api_error_event", result["errors"])

    def test_done_without_finish_and_usage_is_failure(self):
        capture = StreamCapture(0)
        capture.feed(b"data: [DONE]\n\n", 1)
        self.assertFalse(capture.finish()["api_stream_complete"])

    def test_unfinished_stream_is_partial(self):
        capture = StreamCapture(0)
        capture.feed(event({"content": "partial"}).encode(), 1)
        self.assertFalse(capture.finish()["api_stream_complete"])

    def test_duplicate_json_fields_rejected(self):
        with self.assertRaises(ValueError):
            StreamCapture(0).feed(b'data: {"id":"a","id":"b"}\n\n', 1)

    def test_engine_timing_aggregates_are_not_token_timestamps(self):
        capture = StreamCapture(0)
        value = json.loads(event(finish="length", usage=exact_usage())[6:])
        value["timings"] = {"cache_n": 0, "prompt_ms": 50, "predicted_ms": 100}
        capture.feed(("data: " + json.dumps(value) + "\n\ndata: [DONE]\n\n").encode(), 1)
        result = capture.finish()
        self.assertIsNone(result["engine_token_intervals"])
        self.assertEqual(result["api_timings"]["predicted_ms"], 100)

    def test_data_after_done_in_same_read_rejected(self):
        with self.assertRaises(ValueError):
            StreamCapture(0).feed(b"data: [DONE]\n\ndata: [DONE]\n\n", 1)

    def test_tool_call_is_not_silently_treated_as_answer(self):
        with self.assertRaises(ValueError):
            StreamCapture(0).feed(event({"tool_calls": []}).encode(), 1)

    def test_unknown_api_cached_count_is_not_filled_with_zero(self):
        capture = StreamCapture(0)
        usage = exact_usage()
        del usage["prompt_tokens_details"]
        capture.feed((event(finish="length", usage=usage) + "data: [DONE]\n\n").encode(), 1)
        self.assertIsNone(capture.finish()["usage"]["cached_tokens"])


class RunnerBoundaries(unittest.TestCase):
    def test_replay_is_network_free_and_closes_both_adapters(self):
        adapters = []
        def factory(req):
            value = FakeStreamTransport(frames("chatcmpl-" + req["id"]))
            adapters.append(value)
            return value
        with mock.patch("socket.socket", side_effect=AssertionError("network forbidden")):
            result = capture_pair([request("left"), request("right")], factory)
        self.assertTrue(result["api_streams_complete"])
        self.assertTrue(all(adapter.closed for adapter in adapters))
        self.assertEqual(result["evidence_kind"], "synthetic_test")
        self.assertFalse(result["correlation_complete"])
        self.assertFalse(result["production_qualified"])

    def test_dual_dispatch_is_barrier_synchronized(self):
        result = captured_pair()
        self.assertEqual(result["dispatch_mode"], "barrier")
        self.assertGreaterEqual(result["start_skew_s"], 0)
        self.assertEqual({row["client_request_id"] for row in result["records"]}, {"left", "right"})

    def test_identical_content_not_a_distinct_p2_request(self):
        one, two = request("left"), request("left")
        two["id"] = "right"
        with self.assertRaises(ValueError):
            capture_pair([one, two], lambda req: FakeStreamTransport(frames()))

    def test_plaintext_answers_and_prompts_not_in_capture_report(self):
        serialized = json.dumps(captured_pair(), ensure_ascii=False)
        self.assertNotIn("架空の回答", serialized)
        self.assertNotIn("架空テスト", serialized)

    def test_frozen_wire_hash_matches_bytes(self):
        prepared = prepare_request(request())
        self.assertEqual(prepared["request_sha256"], hashlib.sha256(prepared["wire"]).hexdigest())

    def test_credential_tool_and_external_action_fields_rejected(self):
        for key in ("Authorization", "api_key", "tools", "strata_mcp", "web_search", "user"):
            data = request()
            data["payload"][key] = "synthetic"
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_request(data)

    def test_multimodal_content_not_accepted_by_text_only_runner(self):
        data = request()
        data["payload"]["messages"][0]["content"] = [{"type": "image_url", "image_url": "https://example.invalid"}]
        with self.assertRaises(ValueError):
            prepare_request(data)

    def test_alias_conflict_rejected_before_transport_creation(self):
        data = request()
        data["payload"]["max_completion_tokens"] = 100
        with self.assertRaises(ValueError):
            prepare_request(data)

    def test_http_adapter_disabled_by_default(self):
        adapter = LoopbackHTTPTransport("http://127.0.0.1:8080/v1/chat/completions")
        with mock.patch("http.client.HTTPConnection") as connection, self.assertRaises(ValueError):
            adapter.open(prepare_request(request()), 0)
        connection.assert_not_called()

    def test_only_literal_loopback_and_pinned_route_allowed(self):
        self.assertEqual(loopback_endpoint("http://127.0.0.1:8080/v1/chat/completions"), ("127.0.0.1", 8080))
        self.assertEqual(loopback_endpoint("http://[::1]:8080/v1/chat/completions"), ("::1", 8080))
        for endpoint in ("http://localhost:8080/v1/chat/completions", "http://192.0.2.1:8080/v1/chat/completions",
                         "https://127.0.0.1:8080/v1/chat/completions", "http://127.0.0.1/v1/chat/completions",
                         "http://user:pass@127.0.0.1:8080/v1/chat/completions", "http://127.0.0.1:8080/health",
                         "http://127.0.0.1:8080/v1/chat/completions?key=synthetic", "http://127.0.0.1:8080/v1/chat/completions#x",
                         "http://127.1:8080/v1/chat/completions", "http://example.invalid:8080/v1/chat/completions"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                loopback_endpoint(endpoint)

    def test_http_adapter_has_no_auth_headers_or_redirect_following(self):
        connection = mock.Mock()
        response = connection.getresponse.return_value
        response.status = 302
        response.getheader.return_value = "text/html"
        with mock.patch("http.client.HTTPConnection", return_value=connection), mock.patch("socket.socket", side_effect=AssertionError("network forbidden")):
            adapter = LoopbackHTTPTransport("http://127.0.0.1:8080/v1/chat/completions", enabled=True)
            opened = adapter.open(prepare_request(request()), time.monotonic())
            self.assertEqual(list(opened.frames), [])
            opened.close()
        self.assertEqual(connection.request.call_count, 1)
        headers = connection.request.call_args.kwargs["headers"]
        self.assertEqual(set(headers), {"Content-Type", "Accept", "Connection"})
        response.read1.assert_not_called()

    def test_invalid_stream_is_retained_as_failed_member(self):
        result = capture_pair([request("left"), request("right")],
                              lambda req: FakeStreamTransport(frames() if req["id"] == "left" else [{"at_s": 1, "data": "data: invalid\n\n"}]))
        self.assertFalse(result["api_streams_complete"])
        self.assertEqual(len(result["records"]), 2)
        self.assertTrue(result["records"][1]["errors"])

    def test_unauthorized_response_never_attempts_authentication(self):
        result = capture_pair([request("left"), request("right")], lambda req: FakeStreamTransport([], status=401, content_type="application/json"))
        self.assertFalse(result["api_streams_complete"])
        self.assertEqual(result["records"][0]["status_code"], 401)

    def test_captured_transport_rejects_injected_clock_before_open(self):
        adapters = []
        def factory(req):
            adapter = mock.Mock(evidence_kind="captured")
            adapters.append(adapter)
            return adapter
        with self.assertRaises(ValueError):
            capture_pair([request("left"), request("right")], factory, clock=lambda: 0)
        for adapter in adapters:
            adapter.open.assert_not_called()

    def test_http_happy_path_uses_mock_only_and_closes_connection(self):
        connection = mock.Mock()
        response = connection.getresponse.return_value
        response.status = 200
        response.getheader.return_value = "text/event-stream"
        response.read1.side_effect = [b"data: [DONE]\n\n", b""]
        with mock.patch("http.client.HTTPConnection", return_value=connection), mock.patch("socket.socket", side_effect=AssertionError("network forbidden")), mock.patch("time.monotonic", return_value=10):
            adapter = LoopbackHTTPTransport("http://127.0.0.1:8080/v1/chat/completions", enabled=True)
            opened = adapter.open(prepare_request(request()), 10)
            self.assertEqual(list(opened.frames), [(10, b"data: [DONE]\n\n")])
            opened.close()
        connection.close.assert_called_once()
        response.close.assert_called_once()

    def test_connection_failure_closes_without_retry(self):
        connection = mock.Mock()
        connection.request.side_effect = TimeoutError()
        with mock.patch("http.client.HTTPConnection", return_value=connection), mock.patch("socket.socket", side_effect=AssertionError("network forbidden")):
            adapter = LoopbackHTTPTransport("http://127.0.0.1:8080/v1/chat/completions", enabled=True)
            with self.assertRaises(TimeoutError):
                adapter.open(prepare_request(request()), time.monotonic())
        connection.close.assert_called_once()
        self.assertEqual(connection.request.call_count, 1)

    def test_no_http_response_keeps_status_unknown(self):
        def factory(req):
            adapter = mock.Mock(evidence_kind="synthetic_test")
            adapter.open.side_effect = TimeoutError()
            return adapter
        result = capture_pair([request("left"), request("right")], factory)
        self.assertFalse(result["api_streams_complete"])
        self.assertIsNone(result["records"][0]["status_code"])
        self.assertIn("transport_failure:TimeoutError", result["records"][0]["errors"])


class HTTPDeadlines(unittest.TestCase):
    endpoint = "http://127.0.0.1:8080/v1/chat/completions"

    def test_expired_before_open_never_connects(self):
        adapter = LoopbackHTTPTransport(self.endpoint, enabled=True, maximum_duration_s=1)
        with mock.patch("time.monotonic", return_value=11), mock.patch("http.client.HTTPConnection") as factory:
            with self.assertRaisesRegex(TimeoutError, "maximum capture duration"):
                adapter.open(prepare_request(request()), 10)
        factory.assert_not_called()

    def test_connect_is_bounded_and_late_connect_never_sends(self):
        for idle, expected in ((10, 1), (.25, .25)):
            with self.subTest(idle=idle):
                now = [10]
                connection = mock.Mock()
                connection.connect.side_effect = lambda: now.__setitem__(0, 11)
                with mock.patch("time.monotonic", side_effect=lambda: now[0]), mock.patch(
                        "http.client.HTTPConnection", return_value=connection) as factory:
                    adapter = LoopbackHTTPTransport(self.endpoint, enabled=True,
                                                    socket_timeout_s=idle, maximum_duration_s=1)
                    with self.assertRaises(TimeoutError):
                        adapter.open(prepare_request(request()), 10)
                self.assertEqual(factory.call_args.kwargs["timeout"], expected)
                connection.connect.assert_called_once()
                connection.request.assert_not_called()
                connection.close.assert_called_once()
                connection.sock.close.assert_called_once()

    def test_late_body_bytes_and_eof_are_rejected_and_closed(self):
        for data in (b"data: [DONE]\n\n", b""):
            with self.subTest(data=data):
                now = [10]
                connection, timer = mock.Mock(), mock.Mock()
                response = connection.getresponse.return_value
                response.status = 200

                def read(size):
                    now[0] = 11
                    return data

                response.read1.side_effect = read
                with mock.patch("time.monotonic", side_effect=lambda: now[0]), mock.patch(
                        "http.client.HTTPConnection", return_value=connection), mock.patch(
                        "threading.Timer", return_value=timer):
                    adapter = LoopbackHTTPTransport(self.endpoint, enabled=True, maximum_duration_s=1)
                    opened = adapter.open(prepare_request(request()), 10)
                    with self.assertRaises(TimeoutError):
                        next(opened.frames)
                    opened.close()  # Cleanup remains idempotent after generator failure.
                response.close.assert_called_once()
                connection.close.assert_called_once()
                timer.cancel.assert_called_once()
                timer.join.assert_called_once()

    def test_transport_errors_cancel_and_join_watchdog_without_retry(self):
        for stage in ("request", "getresponse", "getheader", "read1"):
            with self.subTest(stage=stage):
                connection, timer = mock.Mock(), mock.Mock()
                response = connection.getresponse.return_value
                response.status = 200
                target = connection if stage in ("request", "getresponse") else response
                getattr(target, stage).side_effect = OSError("synthetic transport error")
                with mock.patch("time.monotonic", return_value=10), mock.patch(
                        "http.client.HTTPConnection", return_value=connection), mock.patch(
                        "threading.Timer", return_value=timer):
                    adapter = LoopbackHTTPTransport(self.endpoint, enabled=True, maximum_duration_s=1)
                    with self.assertRaisesRegex(OSError, "synthetic transport error"):
                        opened = adapter.open(prepare_request(request()), 10)
                        next(opened.frames)
                connection.connect.assert_called_once()
                connection.request.assert_called_once()
                connection.close.assert_called_once()
                connection.sock.close.assert_called_once()
                timer.cancel.assert_called_once()
                timer.join.assert_called_once()
                if stage in ("getheader", "read1"):
                    response.close.assert_called_once()

    def socketpair_capture(self, prefix=b"", trickle=b"", *, duration=.1, idle=2):
        """Exercise real HTTP buffering over local IPC, never a TCP server or GPU."""
        client, peer = socket.socketpair()
        stop = threading.Event()
        timers = []
        timer_type = threading.Timer

        def make_timer(*args, **kwargs):
            timer = timer_type(*args, **kwargs)
            timers.append(timer)
            return timer

        def connect(connection):
            self.assertEqual((connection.host, connection.port), ("127.0.0.1", 8080))
            client.settimeout(connection.timeout)
            connection.sock = client

        def serve():
            with peer:
                peer.settimeout(1)
                try:
                    peer.sendall(prefix)
                    # Finite even with the old implementation, so a regression
                    # fails instead of leaving the test's sender stuck forever.
                    end = time.monotonic() + .8
                    while time.monotonic() < end and not stop.wait(.01):
                        if trickle:
                            peer.sendall(trickle)
                except OSError:
                    pass  # Deadline shutdown or early SSE completion is expected.

        sender = threading.Thread(target=serve, daemon=True)
        sender.start()
        try:
            with mock.patch("http.client.HTTPConnection.connect", autospec=True, side_effect=connect) as factory, mock.patch(
                    "threading.Timer", side_effect=make_timer):
                adapter = LoopbackHTTPTransport(self.endpoint, enabled=True,
                                                socket_timeout_s=idle, maximum_duration_s=duration)
                start = time.monotonic()
                result = _capture(prepare_request(request()), adapter, start)
                elapsed = time.monotonic() - start
            factory.assert_called_once()
            self.assertEqual(client.fileno(), -1, "response/socket descriptor leaked")
            self.assertEqual(len(timers), 1)
            self.assertFalse(timers[0].is_alive(), "deadline watchdog was not joined")
            self.assertLess(elapsed, .5, "capture exceeded its short deadline")
        finally:
            stop.set()
            sender.join(1)
            client.close()
            peer.close()
        self.assertFalse(sender.is_alive(), "test sender did not stop")
        return result

    def test_delayed_headers_stop_before_long_socket_timeout(self):
        result = self.socketpair_capture()
        self.assertIsNone(result["status_code"])
        self.assertIn("transport_failure:TimeoutError", result["errors"])

    def test_trickling_headers_cannot_extend_absolute_deadline(self):
        result = self.socketpair_capture(b"HTTP/1.1 200 OK\r\nX-Slow: ", b"x")
        self.assertIsNone(result["status_code"])
        self.assertIn("transport_failure:TimeoutError", result["errors"])

    def test_trickling_chunk_framing_uses_retained_socket(self):
        result = self.socketpair_capture(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Connection: close\r\nTransfer-Encoding: chunked\r\n\r\n", b"1")
        self.assertEqual(result["status_code"], 200)
        self.assertIn("transport_failure:TimeoutError", result["errors"])
        self.assertFalse(result["api_stream_complete"])

    def test_shorter_socket_idle_timeout_is_preserved(self):
        result = self.socketpair_capture(duration=2, idle=.03)
        self.assertIn("transport_failure:TimeoutError", result["errors"])

    def test_done_closes_response_and_watchdog_without_waiting_for_peer(self):
        body = "".join(row["data"] for row in frames()).encode()
        headers = ("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                   "Connection: close\r\nContent-Length: " + str(len(body)) + "\r\n\r\n").encode()
        result = self.socketpair_capture(headers + body, duration=2)
        self.assertTrue(result["api_stream_complete"], result["errors"])


class Correlation(unittest.TestCase):
    def setUp(self):
        self.capture = captured_pair()
        self.sidecar = engine_sidecar(self.capture)

    def test_explicit_synthetic_binding_permits_only_offline_metrics(self):
        result = correlate_pair(self.capture, self.sidecar)
        self.assertTrue(result["correlation_complete"], result["errors"])
        self.assertFalse(result["production_qualified"])
        self.assertEqual(result["evidence_kind"], "synthetic_test")
        self.assertEqual(result["records"][0]["engine_token_intervals"]["count"], 511)
        self.assertIsNotNone(result["p2_metrics"])

    def test_api_stop_is_not_proof_of_engine_completion(self):
        self.capture["records"][0]["finish_reason"] = "stop"
        self.sidecar["records"][0]["engine"]["terminal_status"] = "cancelled"
        result = correlate_pair(self.capture, self.sidecar)
        self.assertFalse(result["correlation_complete"])
        self.assertIn("left:engine_not_normally_completed", result["errors"])

    def test_correlation_by_slot_or_timing_is_rejected(self):
        for method in ("nearest_timestamp", "slot_number", "response_order"):
            self.sidecar["records"][0]["binding"]["method"] = method
            self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_swapped_response_ids_rejected(self):
        self.sidecar["records"][0]["binding"]["api_response_id"] = self.capture["records"][1]["response_id"]
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_stale_run_sidecar_rejected(self):
        self.sidecar["run_id"] = "different-run"
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_unverified_clock_domain_rejected(self):
        self.sidecar["clock_binding"] = "estimated_offset"
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_synthetic_evidence_cannot_be_mixed_with_captured(self):
        self.sidecar["evidence_kind"] = "captured"
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_tokenization_receipt_must_bind_exact_wire_request(self):
        self.sidecar["records"][0]["tokenization"]["request_sha256"] = "f" * 64
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_character_count_is_not_runtime_tokenization(self):
        self.sidecar["records"][0]["tokenization"]["source"] = "character_estimate"
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_template_or_image_expansion_cannot_be_omitted(self):
        self.sidecar["records"][0]["tokenization"]["includes_template_and_expansion"] = False
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_actual_prompt_count_must_equal_target(self):
        self.sidecar["records"][0]["tokenization"]["encoded_input_tokens"] = 4095
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_cold_engine_cache_must_be_zero(self):
        self.sidecar["records"][0]["engine"]["cache_n"] = 128
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_api_cache_receipt_must_match_engine(self):
        self.capture["records"][0]["usage"]["cached_tokens"] = 128
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_missing_token_timestamps_do_not_fall_back_to_sse(self):
        del self.sidecar["records"][0]["engine"]["token_timestamps_s"]
        result = correlate_pair(self.capture, self.sidecar)
        self.assertFalse(result["correlation_complete"])
        self.assertIsNone(result["p2_metrics"])

    def test_511_engine_tokens_do_not_count_as_512(self):
        self.sidecar["records"][0]["engine"]["token_timestamps_s"].pop()
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_engine_timestamp_outside_request_interval_rejected(self):
        self.sidecar["records"][0]["engine"]["token_timestamps_s"][0] = self.capture["records"][0]["send_s"] - 1
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_same_engine_slot_is_not_p2(self):
        self.sidecar["records"][1]["engine"]["slot"] = 0
        result = correlate_pair(self.capture, self.sidecar)
        self.assertIn("distinct_engine_slots", result["errors"])
        self.assertFalse(result["correlation_complete"])

    def test_same_engine_request_cannot_be_joined_twice(self):
        self.sidecar["records"][1]["engine"]["request_id"] = "engine-left"
        self.sidecar["records"][1]["binding"]["engine_request_id"] = "engine-left"
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_slot_downgrade_blocks_metrics(self):
        self.sidecar["props_total_slots"] = 1
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_missing_flag_is_unknown_not_false(self):
        del self.sidecar["records"][0]["engine"]["cancelled"]
        self.assertFalse(correlate_pair(self.capture, self.sidecar)["correlation_complete"])

    def test_missing_sidecar_cannot_succeed(self):
        self.assertFalse(correlate_pair(self.capture, {})["correlation_complete"])

    def test_out_of_order_sidecars_join_by_explicit_id(self):
        self.sidecar["records"].reverse()
        self.assertTrue(correlate_pair(self.capture, self.sidecar)["correlation_complete"])


class CaptureCli(unittest.TestCase):
    def run_cli(self, document, *arguments):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return subprocess.run([sys.executable, "-m", "bench.evo_x2.runner", arguments[0], str(path), *arguments[1:]],
                                  capture_output=True, text=True, check=False)

    def test_replay_cli_outputs_only_synthetic_capture(self):
        document = {"schema_version": 1, "synthetic_only": True, "requests": [request("left"), request("right")],
                    "streams": {"left": frames("left"), "right": frames("right")}}
        result = self.run_cli(document, "replay")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["evidence_kind"], "synthetic_test")
        self.assertFalse(report["production_qualified"])

    def test_loopback_cli_needs_explicit_optin_before_networking(self):
        document = {"schema_version": 1, "synthetic_only": True, "requests": [request("left"), request("right")]}
        result = self.run_cli(document, "loopback", "--endpoint", "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --endpoint and --enable-loopback", result.stderr)

    def test_replay_rejects_network_options(self):
        document = {"schema_version": 1, "synthetic_only": True, "requests": [request("left"), request("right")]}
        result = self.run_cli(document, "replay", "--enable-loopback")
        self.assertEqual(result.returncode, 2)
        self.assertIn("never accepts networking", result.stderr)


if __name__ == "__main__":
    unittest.main()

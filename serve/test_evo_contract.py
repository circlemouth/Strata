"""CPU-only HTTP contract checks; mock counts do not qualify real P2 hardware."""
import http.client
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve import evo_contract as evo
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve


class Limits(unittest.TestCase):
    def test_missing_null_and_matching_aliases(self):
        for request in ({}, {key: None for key in evo.OUTPUT_ALIASES}):
            self.assertEqual(evo.normalize_output_limit(request), 16384)
        self.assertEqual(evo.normalize_output_limit(dict.fromkeys(evo.OUTPUT_ALIASES, 512)), 512)
        self.assertEqual(evo.normalize_output_limit({"max_tokens": None, "max_output_tokens": 1}), 1)

    def test_no_coercion_or_conflict(self):
        for key in evo.OUTPUT_ALIASES:
            for value in (False, True, 0, -1, -2, 16385, 1.0, 1.5, "512", [], {}, float("nan")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    evo.normalize_output_limit({key: value})
        with self.assertRaises(ValueError):
            evo.normalize_output_limit({"max_tokens": 1, "max_completion_tokens": 2})

    def test_configuration_requires_explicit_contract(self):
        good = {"evo_contract": True, "parallel": 2, "fit_max_tokens": False}
        self.assertTrue(evo.validate_config(good))
        self.assertFalse(evo.validate_config({}))
        for delta in ({"evo_contract": "true"}, {"parallel": True}, {"parallel": 1},
                      {"fit_max_tokens": None}, {"fit_max_tokens": True}, {"lazy_load": True}):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                evo.validate_config({**good, **delta})
        for flags in ({"fit_cli": True}, {"lazy_cli": True}):
            with self.assertRaises(ValueError):
                evo.validate_config(good, **flags)

    def test_observed_counts_not_configuration(self):
        good = dict(max_context=131072, batch=2, info={"batch_slots": "2"})
        evo.validate_runtime(SimpleNamespace(**good))
        for delta in (dict(max_context=65536), dict(batch=1), dict(info={}),
                      dict(info={"batch_slots": 1}), dict(info={"batch_slots": True}),
                      dict(batch=True), dict(max_context="131072")):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                evo.validate_runtime(SimpleNamespace(**{**good, **delta}))


class HttpContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "ok", max_context=131072)
        cls.engine.batch = 2
        cls.engine.info = {"batch_slots": 2}
        cls.svc = Service(cls.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"),
                          fit_max_tokens=False, evo_contract=True)
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, fields, prompt=128):
        body = {"input": "hi"} if path.endswith("responses") else {
            "messages": [{"role": "user", "content": "hi"}]}
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            with mock.patch.object(self.svc, "encode_prompt", return_value=[ord("x")] * prompt):
                connection.request("POST", path, json.dumps({**body, **fields}), {"Content-Type": "application/json"})
                response = connection.getresponse()
                return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_all_dialects_reject_invalid_alias_before_generation(self):
        for path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
            for fields in ({"max_tokens": 0}, {"max_output_tokens": -1},
                           {"max_completion_tokens": 1.5}, {"max_tokens": True},
                           {"max_tokens": 3, "max_output_tokens": 4}, {"max_tokens": 16385}):
                with self.subTest(path=path, fields=fields), mock.patch.object(self.engine, "generate") as gen:
                    status, body = self.post(path, fields)
                    self.assertEqual(status, 400, body)
                    gen.assert_not_called()

    def test_defaults_and_aliases_reach_engine_unchanged(self):
        self.svc.shared = {"max_tokens": 1}
        try:
            for path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
                for fields, expected in (({}, 16384), ({"max_tokens": None}, 16384),
                                         ({"max_output_tokens": 512}, 512),
                                         ({"max_completion_tokens": 4096}, 4096)):
                    with self.subTest(path=path, fields=fields), \
                            mock.patch.object(self.engine, "generate", wraps=self.engine.generate) as gen:
                        status, body = self.post(path, fields)
                        self.assertEqual(status, 200, body)
                        self.assertEqual(gen.call_args.args[1], expected)
        finally:
            self.svc.shared = {}

    def test_encoded_boundary_is_exact_for_all_routes(self):
        for path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
            for output, prompt in ((512, 130552), (4096, 126968), (16384, 114680)):
                with self.subTest(path=path, output=output):
                    status, body = self.post(path, {"max_tokens": output}, prompt)
                    self.assertEqual(status, 200, body)
                    with mock.patch.object(self.engine, "generate") as gen:
                        status, body = self.post(path, {"max_tokens": output, "stream": True}, prompt + 1)
                        self.assertEqual(status, 400, body)
                        gen.assert_not_called()

    def test_restart_downgrade_refuses_health_and_requests(self):
        with mock.patch.object(self.engine, "batch", 1):
            status, body = self.post("/v1/chat/completions", {"max_tokens": 1})
            self.assertEqual(status, 503, body)
            conn = http.client.HTTPConnection("127.0.0.1", self.port)
            try:
                conn.request("GET", "/health")
                response = conn.getresponse()
                self.assertEqual(response.status, 503, response.read())
            finally:
                conn.close()

    def test_props_does_not_claim_concurrency_qualification(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        try:
            conn.request("GET", "/props")
            response = conn.getresponse()
            body = json.loads(response.read())
            self.assertEqual(response.status, 200, body)
            self.assertEqual(body["total_slots"], 2)
            self.assertEqual(body["default_generation_settings"]["params"]["n_predict"], 16384)
            self.assertTrue(body["evo_contract"]["qualification_required"])
            self.assertFalse(body["evo_contract"]["concurrent_progress_verified"])
        finally:
            conn.close()

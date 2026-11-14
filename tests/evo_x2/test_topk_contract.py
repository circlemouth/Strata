"""Host-only checks. Passing this test does not validate the HIP kernels."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TopkContractTest(unittest.TestCase):
    def compile_and_run(self, source):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("C++ compiler not available")
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="strata-topk-") as temp:
            target = Path(temp) / "topk-contract"
            subprocess.run(
                [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(root / "include"),
                 str(root / "tests/evo_x2" / source), "-o", str(target)],
                check=True, capture_output=True, text=True,
            )
            return subprocess.run([str(target)], check=True, capture_output=True, text=True).stdout

    def test_host_contract_and_independent_oracle(self):
        output = self.compile_and_run("topk_contract_test.cpp")
        self.assertIn("425 independent oracle cases; HIP NOT RUN", output)

    def test_batch_lifecycle_reference(self):
        output = self.compile_and_run("topk_batch_lifecycle_test.cpp")
        self.assertIn("CPU batch lifecycle/key/threshold reference; NATIVE P2 AND HIP NOT RUN", output)


if __name__ == "__main__":
    unittest.main()

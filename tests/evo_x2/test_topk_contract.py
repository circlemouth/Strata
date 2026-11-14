"""Host-only checks. Passing this test does not validate the HIP kernels."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TopkContractTest(unittest.TestCase):
    def test_host_contract_and_independent_oracle(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("C++ compiler not available")
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="strata-topk-") as temp:
            target = Path(temp) / "topk-contract"
            subprocess.run(
                [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(root / "include"),
                 str(root / "tests/evo_x2/topk_contract_test.cpp"), "-o", str(target)],
                check=True, capture_output=True, text=True,
            )
            result = subprocess.run([str(target)], check=True, capture_output=True, text=True)
        self.assertIn("425 independent oracle cases; HIP NOT RUN", result.stdout)


if __name__ == "__main__":
    unittest.main()

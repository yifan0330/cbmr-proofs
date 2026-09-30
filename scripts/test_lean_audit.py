"""Check that the real Lean audit rejects admitted and extra-axiom proofs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LeanAuditTests(unittest.TestCase):
    def test_untrusted_proofs_are_rejected(self):
        local_lake = ROOT / ".elan/bin/lake"
        lake = str(local_lake) if local_lake.exists() else shutil.which("lake")
        self.assertIsNotNone(lake, "Install Elan and run make lean-deps first")
        env = os.environ.copy()
        if local_lake.exists():
            env["ELAN_HOME"] = str(ROOT / ".elan")
        source = (ROOT / "scripts/LeanAudit.lean").read_text()
        imports, separator, audit = source.partition("run_cmd do")
        self.assertTrue(separator, "Cannot locate the audit command")
        # The isolated rejection fixtures do not need the full mathematical library.
        imports = imports.replace("import CbmrProofs\n", "import Lean.Elab.Command\n", 1)
        cases = [
            (
                "admitted",
                "namespace CbmrProofs.AuditNegative\n"
                "theorem admitted : False := by sorry\n"
                "end CbmrProofs.AuditNegative\n",
                "sorryAx",
            ),
            (
                "extra_axiom",
                "axiom externalUntrustedAssumption : False\n"
                "namespace CbmrProofs.AuditNegative\n"
                "theorem extraAxiom : False := externalUntrustedAssumption\n"
                "end CbmrProofs.AuditNegative\n",
                "externalUntrustedAssumption",
            ),
        ]
        with tempfile.TemporaryDirectory(prefix="cbmr-lean-audit-") as directory:
            for name, declaration, expected in cases:
                with self.subTest(name=name):
                    fixture = Path(directory) / f"{name}.lean"
                    fixture.write_text(imports + declaration + separator + audit)
                    result = subprocess.run(
                        [lake, "env", "lean", str(fixture)], cwd=ROOT, env=env,
                        text=True, capture_output=True, check=False,
                    )
                    output = result.stdout + result.stderr
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn("Untrusted theorem", output, output)
                    self.assertIn(expected, output, output)
                    self.assertNotIn("Lean axiom audit passed", output, output)


if __name__ == "__main__":
    unittest.main()

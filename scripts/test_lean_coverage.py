"""Regression tests for the source-coverage and audited-theorem checks."""

import copy
import hashlib
import unittest

from check_lean_coverage import displayed_math, math_digest, preamble_digest, validate_inventory


class LeanCoverageTests(unittest.TestCase):
    def setUp(self):
        self.source = r"\begin{equation}x=x\end{equation}"
        self.inventory = {
            "display_count": 1,
            "math_sha256": math_digest(displayed_math(self.source)),
            "preamble_sha256": preamble_digest(self.source),
            "formula_macros_sha256": hashlib.sha256(b"").hexdigest(),
            "displays": [{
                "id": "D01", "title": "Identity", "status": "proved",
                "theorems": ["Example.identity"], "remaining": [],
            }],
            "prose": [],
        }
        self.audit = (
            "CHECKED CbmrProofs.Example.identity\n"
            "Lean axiom audit passed for 1 theorem declarations.\n"
        )

    def test_complete_inventory(self):
        self.assertTrue(validate_inventory(
            self.inventory, self.source, self.audit
        )["complete"])

    def test_changed_equation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mathematics changed"):
            validate_inventory(self.inventory, self.source.replace("x=x", "x=y"), self.audit)

    def test_new_display_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "display count changed"):
            validate_inventory(self.inventory, self.source + r"\[y=y\]", self.audit)

    def test_commented_display_is_ignored(self):
        self.assertTrue(validate_inventory(
            self.inventory, self.source + "\n% " + r"\[y=y\]", self.audit
        )["complete"])

    def test_macro_change_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "formula macros changed"):
            validate_inventory(self.inventory, self.source, self.audit, r"\def\x{0}")

    def test_notation_macro_change_is_rejected(self):
        source = r"\newcommand{\E}{\mathrm{E}}\begin{document}" + self.source
        self.inventory["preamble_sha256"] = preamble_digest(source)
        with self.assertRaisesRegex(ValueError, "notation preamble changed"):
            validate_inventory(
                self.inventory, source.replace(r"\mathrm{E}", r"\mathrm{median}"),
                self.audit,
            )

    def test_control_word_spacing_changes_digest(self):
        first = displayed_math(r"\[\mu x\]")
        second = displayed_math(r"\[\mux\]")
        self.assertNotEqual(math_digest(first), math_digest(second))

    def test_unaudited_theorem_is_rejected(self):
        self.inventory["displays"][0]["theorems"] = ["Example.missing"]
        with self.assertRaisesRegex(ValueError, "not audited"):
            validate_inventory(self.inventory, self.source, self.audit)

    def test_failed_audit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "successful Lean"):
            validate_inventory(self.inventory, self.source, "error: untrusted theorem\n")

    def test_error_after_summary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "successful Lean"):
            validate_inventory(self.inventory, self.source, self.audit + "error: later failure\n")

    def test_partial_inventory_is_not_complete(self):
        entry = self.inventory["displays"][0]
        entry["status"] = "partial"
        entry["remaining"] = ["Another implication"]
        self.assertFalse(validate_inventory(
            self.inventory, self.source, self.audit
        )["complete"])

    def test_proved_requires_evidence_and_no_gaps(self):
        for changes in ({"theorems": []}, {"remaining": ["Missing proof"]}):
            with self.subTest(changes=changes):
                inventory = copy.deepcopy(self.inventory)
                inventory["displays"][0].update(changes)
                with self.assertRaisesRegex(ValueError, "needs proofs"):
                    validate_inventory(inventory, self.source, self.audit)

    def test_duplicate_display_is_rejected(self):
        self.inventory["displays"].append(copy.deepcopy(self.inventory["displays"][0]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_inventory(self.inventory, self.source, self.audit)


if __name__ == "__main__":
    unittest.main()

"""Check the reviewed appendix-to-theorem inventory against an actual Lean audit."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DISPLAY = re.compile(
    r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}|\\\[(.*?)\\\]", re.S
)
AUDIT_SUMMARY = re.compile(
    r"^Lean axiom audit passed for (\d+) theorem declarations\.$", re.M
)


def strip_comments(source):
    lines = []
    for line in source.splitlines(keepends=True):
        for index, char in enumerate(line):
            if char != "%":
                continue
            previous = index - 1
            while previous >= 0 and line[previous] == "\\":
                previous -= 1
            if (index - previous - 1) % 2 == 0:
                line = line[:index] + ("\n" if line.endswith("\n") else "")
                break
        lines.append(line)
    return "".join(lines)


def displayed_math(source):
    source = strip_comments(source)
    return [
        {
            "line": source.count("\n", 0, match.start()) + 1,
            "math": match.group(2) if match.group(1) else match.group(3),
        }
        for match in DISPLAY.finditer(source)
    ]


def math_digest(displays):
    canonical = "\n".join(re.sub(r"\s+", " ", item["math"]).strip() for item in displays)
    return hashlib.sha256(canonical.encode()).hexdigest()


def preamble_digest(source):
    preamble, separator, _ = strip_comments(source).partition(r"\begin{document}")
    canonical = re.sub(r"\s+", " ", preamble).strip() if separator else ""
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_inventory(inventory, source, audit, formula_macros=""):
    displays = displayed_math(source)
    if inventory["display_count"] != len(displays):
        raise ValueError("The appendix display count changed; review its Lean coverage.")
    if inventory["math_sha256"] != math_digest(displays):
        raise ValueError("The displayed mathematics changed; review the theorem mapping.")
    if inventory["preamble_sha256"] != preamble_digest(source):
        raise ValueError("The notation preamble changed; review the theorem mapping.")
    if inventory["formula_macros_sha256"] != hashlib.sha256(formula_macros.encode()).hexdigest():
        raise ValueError("The generated formula macros changed; review the theorem mapping.")
    expected_ids = {f"D{number:02}" for number in range(1, len(displays) + 1)}
    entries = inventory["displays"]
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("Each appendix display must have exactly one inventory entry.")
    summaries = AUDIT_SUMMARY.findall(audit)
    checked = re.findall(r"^CHECKED (\S+)$", audit, re.M)
    if (len(summaries) != 1 or int(summaries[0]) != len(checked)
            or re.search(r"(?m)^.*\berror:|^FAILED\b", audit)):
        raise ValueError("A complete successful Lean axiom audit is required.")
    if len(checked) != len(set(checked)):
        raise ValueError("The Lean audit contains duplicate theorem names.")
    checked = set(checked)
    prose_ids = [entry["id"] for entry in inventory["prose"]]
    if len(prose_ids) != len(set(prose_ids)) or set(prose_ids) & expected_ids:
        raise ValueError("Prose obligations must have unique, separate identifiers.")
    for entry in entries + inventory["prose"]:
        status = entry["status"]
        if status not in {"proved", "partial", "pending"}:
            raise ValueError(f"{entry['id']}: invalid coverage status {status!r}")
        if status == "proved" and (not entry["theorems"] or entry["remaining"]):
            raise ValueError(f"{entry['id']}: a proved entry needs proofs and no gaps.")
        if status != "proved" and not entry["remaining"]:
            raise ValueError(f"{entry['id']}: explain the outstanding obligation.")
        for theorem in entry["theorems"]:
            name = f"CbmrProofs.{theorem}"
            if name not in checked:
                raise ValueError(f"{entry['id']}: theorem was not audited: {name}")
    return {
        "math_sha256": inventory["math_sha256"],
        "preamble_sha256": inventory["preamble_sha256"],
        "formula_macros_sha256": inventory["formula_macros_sha256"],
        "audited_theorem_declarations": len(checked),
        "display_coverage": dict(Counter(entry["status"] for entry in entries)),
        "prose_coverage": dict(
            Counter(entry["status"] for entry in inventory["prose"])
        ),
        "complete": all(
            entry["status"] == "proved" for entry in entries + inventory["prose"]
        ),
        "displays": [
            dict(entry, source_line=displays[int(entry["id"][1:]) - 1]["line"])
            for entry in entries
        ],
        "prose": inventory["prose"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    inventory = json.loads((ROOT / "docs/lean_coverage.json").read_text())
    try:
        report = validate_inventory(
            inventory, (ROOT / "docs/cbmr_appendix.tex").read_text(),
            args.audit.read_text(),
            (ROOT / "docs/generated/appendix_equations.tex").read_text(),
        )
    except ValueError as error:
        parser.exit(1, f"Lean coverage check failed: {error}\n")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for label, key in (("Displayed mathematics", "display_coverage"),
                       ("Additional prose derivations", "prose_coverage")):
        counts = report[key]
        print(
            f"{label}: {counts.get('proved', 0)} proved, "
            f"{counts.get('partial', 0)} partial, {counts.get('pending', 0)} pending."
        )
    if report["complete"]:
        print("Every inventoried derivation has an audited theorem mapping.")
    else:
        print("Full appendix formalisation remains incomplete; see docs/lean_coverage.json.")
        if args.require_complete:
            parser.exit(1)


if __name__ == "__main__":
    main()

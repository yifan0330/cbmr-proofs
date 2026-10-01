"""Load actual source snapshots without requiring or replacing installed NiMARE.

The supplied original uses absolute ``nimare.meta.cbmr`` imports. A restricted
loader redirects ONLY those import nodes into a private package namespace.
Likelihood, optimizer and covariance bodies are compiled unchanged from disk.
No files or installed NiMARE modules are modified.
"""

import ast
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PACKAGE = "_comparison_cbmr_original"


class _Imports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        prefix = "nimare.meta.cbmr"
        if node.module == prefix or (node.module and node.module.startswith(prefix + ".")):
            node.module = PRIVATE_PACKAGE + node.module[len(prefix):]
        return node


class _OriginalLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        tree = ast.parse(self.get_data(self.path), filename=self.path)
        return compile(_Imports().visit(tree), self.path, "exec")


class _OriginalFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != PRIVATE_PACKAGE and not fullname.startswith(PRIVATE_PACKAGE + "."):
            return None
        relative = fullname[len(PRIVATE_PACKAGE):].lstrip(".").replace(".", "/")
        source = ROOT / "cbmr_code" / (relative + ".py" if relative else "__init__.py")
        if not source.is_file():
            return None
        loader = _OriginalLoader(fullname, str(source))
        return importlib.util.spec_from_file_location(
            fullname, source, loader=loader,
            submodule_search_locations=[str(source.parent)] if not relative else None,
        )


@lru_cache(None)
def modules(implementation):
    if implementation not in ("original", "improved"):
        raise ValueError("implementation must be 'original' or 'improved'.")
    prefix = "cbmr_improved"
    if implementation == "original":
        prefix = PRIVATE_PACKAGE
        if not any(isinstance(finder, _OriginalFinder) for finder in sys.meta_path):
            sys.meta_path.insert(0, _OriginalFinder())
    return SimpleNamespace(**{
        name: importlib.import_module(f"{prefix}.{name}")
        for name in ("model", "predictor", "distributions", "information", "covariance")
    })


@dataclass(frozen=True)
class _Term:
    name: str
    spatial: bool
    exposure: bool = False
    is_derived_exposure: bool = False

    def __str__(self):
        return self.name


class _BoundDesign:
    def __init__(self, case):
        self.exposure = case.exposure
        self.blocks = [SimpleNamespace(
            term=_Term("spatial", True), block=case.spatial, n_columns=case.spatial.shape[1],
        )]
        if case.n_global:
            self.blocks.append(SimpleNamespace(
                term=_Term("global", False), block=case.global_design, n_columns=case.n_global,
            ))
        self.terms = tuple(block.term for block in self.blocks)

    def parameter_slices(self, n_bases):
        n_spatial = self.blocks[0].n_columns * n_bases
        result = {"spatial": slice(0, n_spatial)}
        if len(self.blocks) > 1:
            result["global"] = slice(n_spatial, n_spatial + self.blocks[1].n_columns)
        return result


def package_model(case, implementation):
    """Instantiate the real CPU model at identical coefficients and fixed dispersion."""
    if case.model == "independent_nb":
        raise ValueError("Independent-cell NB is absent from both source packages.")
    source = modules(implementation)
    names = {
        "poisson": "poisson",
        "aggregated_nb": "negativebinomial",
        "clustered_nb": "clusterednegativebinomial",
    }
    predictor = source.predictor.CBMRPredictor(_BoundDesign(case), case.bases)
    model = source.model.CBMRModel(predictor, names[case.model])
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(case.coefficients, dtype=torch.float64))
        if model.nuisance is not None:
            model.nuisance.fill_(np.log(case.dispersion))
    return model


def package_objective(case, implementation, coefficients=None, model=None):
    """Actual unpenalized source likelihood plus an explicitly external test penalty."""
    model = package_model(case, implementation) if model is None else model
    vector = model.coefficients if coefficients is None else coefficients
    value = -model.log_likelihood(case.counts, flat=vector)
    if case.penalty is not None:
        penalty = torch.as_tensor(case.penalty, dtype=torch.float64)
        value = value + 0.5 * vector @ penalty @ vector
    return value


def package_information(case, implementation, model=None):
    model = package_model(case, implementation) if model is None else model
    information = model.information_matrix(case.counts)
    return information if case.penalty is None else information + case.penalty


def source_manifest():
    """Fingerprint the exact on-disk statistical implementations under comparison."""
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("cbmr_code", "cbmr_improved")
        for path in sorted((ROOT / directory).glob("*.py"))
    }

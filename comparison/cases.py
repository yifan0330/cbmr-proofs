"""Small seeded designs shared by numerical checks and isolated experiments."""

from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np


MODELS = ("poisson", "independent_nb", "aggregated_nb", "clustered_nb")


@dataclass
class Case:
    name: str
    model: str
    design: str
    bases: np.ndarray
    spatial: np.ndarray
    global_design: np.ndarray
    exposure: np.ndarray
    counts: np.ndarray
    coefficients: np.ndarray
    dispersion: float = 0.3
    penalty: np.ndarray | None = None

    @property
    def n_spatial(self):
        return self.spatial.shape[1] * self.bases.shape[1]

    @property
    def n_global(self):
        return self.global_design.shape[1]

    @property
    def n_parameters(self):
        return self.n_spatial + self.n_global

    @cached_property
    def patterns(self):
        return np.unique(self.spatial, axis=0, return_inverse=True)

    def cell_design(self):
        spatial = np.einsum("ic,vk->ivck", self.spatial, self.bases).reshape(
            len(self.spatial), len(self.bases), self.n_spatial
        )
        global_design = np.broadcast_to(
            self.global_design[:, None, :], (*self.counts.shape, self.n_global)
        )
        return np.concatenate((spatial, global_design), axis=2)

    def means(self, coefficients=None):
        coefficients = self.coefficients if coefficients is None else coefficients
        beta = coefficients[:self.n_spatial].reshape(self.spatial.shape[1], self.bases.shape[1])
        eta = (self.spatial @ beta) @ self.bases.T
        eta += (self.global_design @ coefficients[self.n_spatial:])[:, None]
        return self.exposure[:, None] * np.exp(eta)

    def with_coefficients(self, coefficients):
        return replace(self, coefficients=np.asarray(coefficients, dtype=float))


def make_case(model="poisson", design="gc", seed=731, groups=2, experiments=16, voxels=7,
              bases=2, dispersion=0.3, penalty=False):
    """Make an identifiable GC or overlapping SV design with unequal group sizes.

    SV slopes repeat within groups so the original marginal NB implementations
    have multiple experiments per spatial pattern. Continuous unique SV rows
    are tested separately for the cellwise models.
    """
    if model not in MODELS or design not in ("gc", "sv", "overlap"):
        raise ValueError("Unknown model or design.")
    if groups < 1 or experiments < 4 * groups or voxels < bases or bases < 1:
        raise ValueError("Require groups>=1, experiments>=4*groups, and voxels>=bases>=1.")
    if not np.isfinite(dispersion) or dispersion <= 0:
        raise ValueError("dispersion must be finite and positive.")
    rng = np.random.default_rng(seed)
    sizes = np.full(groups, experiments // groups)
    sizes[:experiments % groups] += 1
    if groups > 1 and sizes[-1] > 4:
        sizes[0] += 1
        sizes[-1] -= 1
    assignment = np.repeat(np.arange(groups), sizes)
    spatial = np.eye(groups)[assignment]
    if design != "gc":
        slope = np.concatenate([np.tile([-0.7, 0.7], (size + 1) // 2)[:size] for size in sizes])
        spatial = np.column_stack((spatial, slope))
    grid = np.linspace(-1, 1, voxels)
    basis = np.polynomial.legendre.legvander(grid, bases - 1)
    global_design = rng.normal(size=(experiments, 1))
    exposure = rng.uniform(0.7, 1.4, experiments)
    n_spatial = spatial.shape[1] * bases
    coefficients = rng.uniform(-0.15, 0.15, n_spatial + 1)
    coefficients[:n_spatial:bases] = 0.25
    counts = rng.poisson(1.6, size=(experiments, voxels)).astype(float)
    matrix = None
    if penalty:
        difference = np.diff(np.eye(bases), axis=0)
        matrix = np.zeros((len(coefficients), len(coefficients)))
        matrix[:n_spatial, :n_spatial] = np.kron(
            np.eye(spatial.shape[1]), 0.4 * difference.T @ difference
        )
    return Case(
        f"{model}-{design}", model, design, basis, spatial, global_design,
        exposure, counts, coefficients, dispersion, matrix,
    )

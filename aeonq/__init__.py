"""Framework-independent amplitude-estimation semantics."""

from aeonq.problem import AEProblem
from aeonq.runtime.result import AEResult


def estimate(
    problem: AEProblem,
    *,
    algorithm: str,
    backend: object,
    epsilon: float = 0.01,
    alpha: float = 0.05,
    **options: object,
) -> AEResult:
    """Run an implemented estimator; reject unsupported algorithm names."""
    from aeonq.algorithms.canonical_qae import CanonicalQAE
    from aeonq.algorithms.faster_ae import FasterAE
    from aeonq.algorithms.iqae import IQAE
    from aeonq.algorithms.mlqae import MLQAE

    choices = {
        "iqae": IQAE,
        "mlqae": MLQAE,
        "canonical_qae": CanonicalQAE,
        "faster_ae": FasterAE,
    }
    if algorithm not in choices:
        raise ValueError(
            f"unsupported algorithm {algorithm!r}; available: {', '.join(choices)}"
        )
    solver = choices[algorithm](epsilon=epsilon, alpha=alpha, **options)
    return solver.run(problem, backend)


__all__ = ["AEProblem", "AEResult", "estimate"]

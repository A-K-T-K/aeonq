"""Branch-safe iterative AE with exact per-round binomial intervals.

The simultaneous confidence statement follows from alpha spending and the
union bound under ideal independent Bernoulli shots. Power selection uses only
previous observations, so conditional binomial coverage applies each round.
"""

from dataclasses import dataclass
from typing import Any
from math import pi, sin
from time import perf_counter

from aeonq.adapters.base import BackendAdapter
from aeonq.algorithms.base import CompiledMetrics, Experiment, qualified_guarantee
from aeonq.inference.alias import candidate_branches, unambiguous_branch
from aeonq.inference.intervals import clopper_pearson, phase_interval
from aeonq.problem import AEProblem
from aeonq.runtime.result import AEResult
from aeonq.runtime.checkpoint import (
    IncompatibleCheckpoint,
    identity,
    load_checkpoint,
    save_checkpoint,
)
from aeonq.scheduling.base import Schedule, ScheduleState
from pathlib import Path


@dataclass(frozen=True)
class IQAE:
    epsilon: float
    alpha: float = 0.05
    shots_per_round: int = 256
    max_rounds: int = 20
    max_power: int = 32
    max_resamples_per_round: int = 1
    scheduler: Schedule | None = None
    shot_options: tuple[int, ...] | None = None
    seed: int = 0
    checkpoint: str | Path | None = None

    def __post_init__(self) -> None:
        if not 0 < self.epsilon < 1 or not 0 < self.alpha < 1:
            raise ValueError("epsilon and alpha must be in (0,1)")
        if (
            self.shots_per_round < 1
            or self.max_rounds < 1
            or self.max_power < 0
            or self.max_resamples_per_round < 0
        ):
            raise ValueError(
                "positive shot/round budget and nonnegative power required"
            )
        if self.shot_options is not None and (
            not self.shot_options or any(n < 1 for n in self.shot_options)
        ):
            raise ValueError("shot options must be positive")

    def run(self, problem: AEProblem, backend: BackendAdapter) -> AEResult:
        start = perf_counter()
        experiment = Experiment(problem, backend)
        theta = (0.0, pi / 2)
        powers: list[int] = []
        per_power: dict[int, int] = {}
        branch_history: list[int] = []
        alias_history: list[dict[str, Any]] = []
        observation_history: list[dict[str, int]] = []
        resampling_triggered = False
        status = "INSUFFICIENT_BUDGET"
        elapsed_before_resume = 0.0
        checkpoint_identity = None
        if self.checkpoint is not None:
            configuration = {
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "shots_per_round": self.shots_per_round,
                "max_rounds": self.max_rounds,
                "max_power": self.max_power,
                "max_resamples_per_round": self.max_resamples_per_round,
                "scheduler": repr(self.scheduler),
                "shot_options": self.shot_options,
                "seed": self.seed,
                "algorithm_version": 1,
            }
            checkpoint_identity = identity(problem, backend, "IQAE", configuration)
            if Path(self.checkpoint).exists():
                state = load_checkpoint(self.checkpoint, checkpoint_identity)
                try:
                    restored_theta = tuple(float(x) for x in state["theta_interval"])
                    if len(restored_theta) != 2:
                        raise ValueError("theta interval must have two endpoints")
                    theta = (restored_theta[0], restored_theta[1])
                    powers = list(map(int, state["powers"]))
                    per_power = {
                        int(k): int(v) for k, v in state["shots_per_power"].items()
                    }
                    branch_history = list(map(int, state["branch_history"]))
                    alias_history = list(state["alias_history"])
                    observation_history = list(state["observation_history"])
                    resampling_triggered = bool(state["resampling_triggered"])
                    status = str(state["status"])
                    experiment.shots = int(state["total_shots"])
                    experiment.oracle_queries = int(state["total_oracle_queries"])
                    experiment.evaluations = int(state["total_circuit_evaluations"])
                    experiment.compile_time = float(state["compile_time"])
                    experiment.execution_time = float(state["execution_time"])
                    experiment.resource_metrics = state["resource_metrics"]
                    elapsed_before_resume = float(state["wall_clock_time"])
                except (KeyError, TypeError, ValueError, AttributeError) as exc:
                    raise IncompatibleCheckpoint(
                        "malformed IQAE checkpoint state"
                    ) from exc
                if (
                    len(theta) != 2
                    or not 0 <= theta[0] <= theta[1] <= pi / 2
                    or len(branch_history) > self.max_rounds
                    or status
                    not in ("INSUFFICIENT_BUDGET", "CONVERGED", "INCONCLUSIVE")
                    or experiment.shots < 0
                    or experiment.evaluations < 0
                    or experiment.oracle_queries < 0
                ):
                    raise IncompatibleCheckpoint("inconsistent IQAE checkpoint state")

        def persist() -> None:
            if self.checkpoint is None or checkpoint_identity is None:
                return
            save_checkpoint(
                self.checkpoint,
                checkpoint_identity,
                {
                    "theta_interval": theta,
                    "powers": powers,
                    "shots_per_power": per_power,
                    "branch_history": branch_history,
                    "alias_history": alias_history,
                    "observation_history": observation_history,
                    "resampling_triggered": resampling_triggered,
                    "status": status,
                    "total_shots": experiment.shots,
                    "total_oracle_queries": experiment.oracle_queries,
                    "total_circuit_evaluations": experiment.evaluations,
                    "compile_time": experiment.compile_time,
                    "execution_time": experiment.execution_time,
                    "resource_metrics": experiment.resource_metrics,
                    "wall_clock_time": elapsed_before_resume + perf_counter() - start,
                    "completed_job_ids": [],
                    "fisher_state": None,
                    "scheduler_state": {"next_iteration": len(branch_history)},
                },
            )

        if status not in ("CONVERGED", "INCONCLUSIVE"):
            iterations = range(len(branch_history), self.max_rounds)
        else:
            iterations = range(0)
        for iteration in iterations:
            choices = [
                (k, unambiguous_branch(theta, k)) for k in range(self.max_power + 1)
            ]
            if self.scheduler is None:
                power, branch = max(
                    ((k, b) for k, b in choices if b is not None),
                    key=lambda item: item[0],
                )
                round_shots = self.shots_per_round
            else:
                schedule_state = ScheduleState(
                    theta,
                    iteration,
                    self.max_power,
                    self.shot_options or (self.shots_per_round,),
                    resource_metrics_by_power=CompiledMetrics(experiment),
                )
                selected = self.scheduler.select(schedule_state)
                selected_branch = selected.branch
                if selected_branch is None:
                    raise ValueError("scheduler returned no phase branch")
                power, branch, round_shots = (
                    selected.power,
                    selected_branch,
                    selected.shots,
                )
                if (
                    power < 0
                    or power > self.max_power
                    or unambiguous_branch(theta, power) != branch
                ):
                    raise ValueError("scheduler returned an unsafe power or branch")
            assert branch is not None
            alias_history.append(
                {
                    "chosen_power": power,
                    "branch": branch,
                    "candidate_branches": candidate_branches(theta, power),
                    "rejected_powers": tuple(k for k, b in choices if b is None),
                }
            )
            updated = None
            for attempt in range(self.max_resamples_per_round + 1):
                observed = experiment.sample(
                    power,
                    round_shots,
                    seed=self.seed
                    + iteration * (self.max_resamples_per_round + 1)
                    + attempt,
                )
                powers.append(power)
                per_power[power] = per_power.get(power, 0) + round_shots
                observation_history.append(
                    {
                        "power": power,
                        "shots": round_shots,
                        "successes": observed,
                        "seed": self.seed
                        + iteration * (self.max_resamples_per_round + 1)
                        + attempt,
                    }
                )
                # Finite-horizon Bonferroni allocation includes possible retries.
                p_interval = clopper_pearson(
                    observed,
                    round_shots,
                    self.alpha / (self.max_rounds * (self.max_resamples_per_round + 1)),
                )
                candidate = phase_interval(p_interval, power, branch)
                proposed = (max(theta[0], candidate[0]), min(theta[1], candidate[1]))
                if proposed[0] <= proposed[1] + 1e-14:
                    # Intervals that only touch within rounding may cross by
                    # up to 1e-14; collapse to their midpoint (kept inside the
                    # current interval) rather than store lower > upper.
                    if proposed[0] > proposed[1]:
                        middle = min(
                            max((proposed[0] + proposed[1]) / 2, theta[0]), theta[1]
                        )
                        proposed = (middle, middle)
                    updated = proposed
                    break
                if attempt < self.max_resamples_per_round:
                    resampling_triggered = True
            branch_history.append(branch)
            if updated is None:
                status = "INCONCLUSIVE"
                persist()
                break
            theta = updated
            a_interval = (sin(theta[0]) ** 2, sin(theta[1]) ** 2)
            if a_interval[1] - a_interval[0] <= 2 * self.epsilon:
                status = "CONVERGED"
                persist()
                break
            persist()
        a_interval = (sin(theta[0]) ** 2, sin(theta[1]) ** 2)
        estimate = (a_interval[0] + a_interval[1]) / 2
        assumptions = (
            "ideal independent Bernoulli shots",
            "exact state preparation, reflections, and objective measurement",
            "backend counts faithfully normalized",
        )
        guarantee, warnings = qualified_guarantee(
            backend,
            "THEORETICAL_GUARANTEE_SATISFIED"
            if status == "CONVERGED"
            else "INSUFFICIENT_BUDGET"
            if status == "INSUFFICIENT_BUDGET"
            else "INCONCLUSIVE",
        )
        return AEResult(
            estimate,
            a_interval,
            "frequentist confidence interval",
            self.epsilon,
            self.alpha,
            status,
            "finite-sample frequentist",
            guarantee,
            assumptions,
            experiment.shots,
            experiment.oracle_queries,
            experiment.evaluations,
            tuple(powers),
            per_power,
            max(powers, default=0),
            len(branch_history),
            experiment.compile_time,
            experiment.execution_time,
            elapsed_before_resume + perf_counter() - start,
            type(backend).__name__,
            type(backend).__module__.split(".")[-1],
            next(iter(backend.capabilities().metadata.values()), "unknown"),
            experiment.resource_metrics or {},
            warnings,
            {
                "theta_interval": theta,
                "branch_history": tuple(branch_history),
                "alias_risk": sum(len(row["rejected_powers"]) for row in alias_history),
                "candidate_branches_last": candidate_branches(theta, powers[-1])
                if powers
                else (),
                "alias_history": tuple(alias_history),
                "branch_selected": tuple(branch_history),
                "branch_confidence": 1 - self.alpha,
                "resampling_triggered": resampling_triggered,
                "observation_history": tuple(observation_history),
            },
        )

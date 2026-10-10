from aeonq.adapters.reference import ReferenceAdapter
from aeonq.algorithms.base import Experiment
from aeonq.benchmarks.problems import bernoulli


class CountingReference(ReferenceAdapter):
    def __init__(self) -> None:
        self.compile_calls = 0

    def compile(self, native_program, options=None):
        self.compile_calls += 1
        return super().compile(native_program, options)

    def resource_metrics(self, compiled_program):
        metrics = super().resource_metrics(compiled_program)
        metrics["depth"] = 5
        metrics["two_qubit_gates"] = 2
        return metrics


def test_experiment_compiles_each_power_once_but_resamples_with_new_seeds():
    backend = CountingReference()
    experiment = Experiment(bernoulli(0.25).problem, backend)
    observations = [experiment.sample(0, 64, seed=seed) for seed in range(5)]
    experiment.sample(1, 64, seed=7)
    assert backend.compile_calls == 2
    assert experiment.evaluations == 6
    assert experiment.shots == 384
    assert experiment.resource_metrics["total_compiled_depth"] == 30
    assert experiment.resource_metrics["total_two_qubit_gates"] == 12
    assert len(set(observations)) > 1

"""Advertised statevector access must correspond to the adapter's public API."""

import numpy as np
import pytest

from aeonq.adapters.reference import ReferenceAdapter
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.ir import Gate, Sequence


@pytest.mark.parametrize(
    "factory", [ReferenceAdapter, QiskitAdapter, CirqAdapter, PennyLaneAdapter]
)
def test_advertised_statevector_retains_relative_phase_and_logical_order(factory):
    adapter = factory()
    assert adapter.capabilities().supports_statevector
    ir = Sequence((Gate("H", (1,), 2), Gate("RZ", (1,), 2, np.pi / 2)), 2)
    expected = np.array([np.exp(-1j * np.pi / 4), 0, np.exp(1j * np.pi / 4), 0])
    assert np.allclose(adapter.statevector(ir), expected / np.sqrt(2), atol=1e-9)


def test_qsharp_does_not_advertise_unimplemented_statevector_access():
    assert not QSharpAdapter().capabilities().supports_statevector

"""Tests for QuantumHardwareTarget, fail-closed execution receipts, and hash-locked Nexus dispatch."""

import pytest

from qccd_showcase.canonical import canonical_sha256
from qccd_showcase.nexus_gate import (
    ContractViolationError,
    HashMismatchError,
    NexusSubmissionGate,
    QNexusGateError,
    QuantumExecutionReceipt,
    QuantumHardwareTarget,
)


class _TransientError(Exception):
    def __init__(self, status_code: int, retry_after: float = 0.0) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


class _MockQNexusAdapter:
    def __init__(self, *, mutate_compiled: bool = False, fail_auth: bool = False, transient_failures: int = 0) -> None:
        self.mutate_compiled = mutate_compiled
        self.fail_auth = fail_auth
        self.transient_remaining = transient_failures
        self.execute_called = False

    def upload_circuit(self, circ: dict, name: str) -> str:
        if self.fail_auth:
            raise _TransientError(401)
        if self.transient_remaining > 0:
            self.transient_remaining -= 1
            raise _TransientError(503, retry_after=0.0)
        return f"ref:{name}"

    def compile_circuit(self, ref: str, device_name: str, optimisation_level: int) -> tuple[str, dict]:
        assert optimisation_level == 0
        compiled = {"gates": ["ZZPhase"] * (33 if self.mutate_compiled else 32)}
        return "compile-job-001", compiled

    def execute_circuit(self, compiled_circ: dict, device_name: str, shots: int) -> tuple[str, dict]:
        self.execute_called = True
        return "exec-job-h2-001", {"0" * 16: shots}


def test_hardware_targets_and_simulated_vs_hardware_job_id_invariant() -> None:
    helios = QuantumHardwareTarget.quantinuum_helios("SIMULATED")
    h2 = QuantumHardwareTarget.quantinuum_h2("H2-1", "HARDWARE")
    assert helios.validate() is True
    assert h2.validate() is True

    # SIMULATED receipt MUST reject a fabricated provider_job_id
    with pytest.raises(ContractViolationError, match="fabricated provider_job_id"):
        QuantumExecutionReceipt(
            backend_id="quantinuum_helios_selene",
            provider="Quantinuum",
            shots=1024,
            result_sha256="a" * 64,
            evidence_class="SIMULATED",
            provider_job_id="fake-job-123",
        )

    # HARDWARE receipt MUST require a real provider_job_id
    with pytest.raises(ContractViolationError, match="authoritative provider_job_id"):
        QuantumExecutionReceipt(
            backend_id="H2-1",
            provider="Quantinuum",
            shots=1024,
            result_sha256="a" * 64,
            evidence_class="HARDWARE",
            provider_job_id=None,
        )

    rec = QuantumExecutionReceipt(
        backend_id="H2-1",
        provider="Quantinuum",
        shots=1000,
        result_sha256="b" * 64,
        evidence_class="HARDWARE",
        provider_job_id="qnx-job-999",
    ).sign("test-secret")
    assert rec.verify_integrity("test-secret") is True
    assert rec.verify_integrity("wrong-secret") is False


def test_nexus_gate_retries_transient_503_and_aborts_before_execute_on_isa_drift() -> None:
    pre_circ = {"stage": "pre"}
    isa_circ = {"gates": ["ZZPhase"] * 32}
    manifest = {
        "hash_lock": True,
        "compiler_lock": True,
        "optimisation_level": 0,
        "precompile_sha256": canonical_sha256(pre_circ),
        "isa_sha256": canonical_sha256(isa_circ),
    }
    gate = NexusSubmissionGate(manifest["precompile_sha256"], manifest["isa_sha256"], canonical_sha256)

    # 1. Happy path on H2-1 with 2 transient 503 retries
    ok_adapter = _MockQNexusAdapter(transient_failures=2)
    res = gate.dispatch_hash_locked(
        pre_circuit=pre_circ,
        isa_circuit=isa_circ,
        manifest=manifest,
        qnx_adapter=ok_adapter,
        device_name="H2-1",
        shots=1000,
        sleep_fn=lambda _: None,
    )
    assert res["status"] == "EXECUTED"
    assert res["execute_job_id"] == "exec-job-h2-001"
    assert ok_adapter.execute_called is True

    # 2. Server-side ISA drift aborts BEFORE execute_circuit is ever called (no HQC spent)
    drift_adapter = _MockQNexusAdapter(mutate_compiled=True)
    with pytest.raises(HashMismatchError, match="aborting execute"):
        gate.dispatch_hash_locked(
            pre_circuit=pre_circ,
            isa_circuit=isa_circ,
            manifest=manifest,
            qnx_adapter=drift_adapter,
            device_name="H2-1",
            shots=1000,
            sleep_fn=lambda _: None,
        )
    assert drift_adapter.execute_called is False

    # 3. Non-retryable 401 fails closed immediately
    auth_adapter = _MockQNexusAdapter(fail_auth=True)
    with pytest.raises(QNexusGateError, match="Non-retryable"):
        gate.dispatch_hash_locked(
            pre_circuit=pre_circ,
            isa_circuit=isa_circ,
            manifest=manifest,
            qnx_adapter=auth_adapter,
            device_name="H2-1SC",
            sleep_fn=lambda _: None,
        )

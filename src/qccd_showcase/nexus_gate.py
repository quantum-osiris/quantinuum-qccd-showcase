"""Hash-locked Quantinuum Nexus (`qnexus`) dispatch gate & auditable execution receipts."""

from __future__ import annotations

import hashlib
import hmac
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

from .canonical import canonical_json_dumps, canonical_sha256, is_valid_sha256_hex

T = TypeVar("T")

TRANSIENT_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})
NON_RETRYABLE_HTTP_CODES = frozenset({400, 401, 403, 404, 422})


class ContractViolationError(ValueError):
    """Raised when a hardware target or execution receipt violates a fail-closed contract."""


class HashMismatchError(RuntimeError):
    """Raised when a pre-compile or compiled ISA circuit digest drifts from the frozen manifest."""


class QNexusGateError(RuntimeError):
    """Raised when a qnexus API call fails authentication or exhausts transient retries."""


@dataclass
class QuantumHardwareTarget:
    """Governed capability specification for Quantinuum QCCD and planar transmon backends."""

    provider: str
    backend_id: str
    native_gateset: List[str]
    topology_type: str
    supports_dynamic_circuits: bool
    max_shots: int
    max_qubits: int = 98
    two_qubit_error_rate: float = 0.00079
    evidence_class: str = "SIMULATED"

    VALID_TOPOLOGIES = frozenset({"QCCD_ALL_TO_ALL", "HEAVY_HEX", "FULL_EMULATION"})

    def validate(self) -> bool:
        try:
            if not isinstance(self.provider, str) or not self.provider.strip():
                return False
            if not isinstance(self.backend_id, str) or not self.backend_id.strip():
                return False
            if not self.native_gateset or any(not isinstance(g, str) or not g.strip() for g in self.native_gateset):
                return False
            if self.topology_type not in self.VALID_TOPOLOGIES:
                return False
            if not isinstance(self.supports_dynamic_circuits, bool):
                return False
            if isinstance(self.max_shots, bool) or self.max_shots <= 0:
                return False
            if isinstance(self.max_qubits, bool) or self.max_qubits <= 0:
                return False
            if not (0.0 <= float(self.two_qubit_error_rate) <= 1.0):
                return False
            if self.evidence_class not in ("SIMULATED", "HARDWARE"):
                return False
            return True
        except Exception:
            return False

    @classmethod
    def quantinuum_helios(cls, evidence_class: str = "SIMULATED") -> "QuantumHardwareTarget":
        return cls(
            provider="Quantinuum",
            backend_id="quantinuum_helios_selene" if evidence_class == "SIMULATED" else "Helios-1",
            native_gateset=["RZ", "PhasedX", "ZZPhase", "ZZMax", "Measure", "Reset"],
            topology_type="QCCD_ALL_TO_ALL",
            supports_dynamic_circuits=True,
            max_shots=100_000,
            max_qubits=98,
            two_qubit_error_rate=0.00079,
            evidence_class=evidence_class,
        )

    @classmethod
    def quantinuum_h2(cls, device_name: str = "H2-1", evidence_class: str = "HARDWARE") -> "QuantumHardwareTarget":
        return cls(
            provider="Quantinuum",
            backend_id=device_name,
            native_gateset=["Rz", "PhasedX", "ZZMax", "ZZPhase", "TK2", "Measure", "Reset"],
            topology_type="QCCD_ALL_TO_ALL",
            supports_dynamic_circuits=True,
            max_shots=100_000,
            max_qubits=56,
            two_qubit_error_rate=0.001,
            evidence_class=evidence_class,
        )


@dataclass
class QuantumExecutionReceipt:
    """Tamper-evident execution receipt enforcing SIMULATED vs HARDWARE job-ID provenance."""

    backend_id: str
    provider: str
    shots: int
    result_sha256: str
    native_gateset: str = "RZ,PhasedX,ZZPhase,ZZMax,Measure,Reset"
    queue_duration_sec: float = 0.0
    execution_duration_sec: float = 0.0
    calibration_sha256: str = "0" * 64
    execution_id: str = field(default_factory=lambda: f"qexec-{uuid.uuid4().hex[:12]}")
    execution_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    receipt_id: str = ""
    nonce: str = ""
    evidence_class: str = "SIMULATED"
    provider_job_id: Optional[str] = None
    receipt_sha256: str = ""
    signature_hmac_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.shots, bool) or int(self.shots) <= 0:
            raise ContractViolationError("QuantumExecutionReceipt.shots must be > 0")
        if self.queue_duration_sec < 0.0 or self.execution_duration_sec < 0.0:
            raise ContractViolationError("Receipt durations must be non-negative")
        if not is_valid_sha256_hex(self.calibration_sha256) or not is_valid_sha256_hex(self.result_sha256):
            raise ContractViolationError("calibration_sha256 and result_sha256 must be 64-hex SHA-256")
        if self.evidence_class == "SIMULATED" and self.provider_job_id is not None:
            raise ContractViolationError(
                "SIMULATED QuantumExecutionReceipt must not carry a fabricated provider_job_id"
            )
        if self.evidence_class == "HARDWARE" and not self.provider_job_id:
            raise ContractViolationError(
                "HARDWARE QuantumExecutionReceipt requires an authoritative provider_job_id"
            )

        seed = f"{self.execution_id}:{self.backend_id}:{self.execution_timestamp}:{self.result_sha256}"
        if not self.receipt_id:
            self.receipt_id = "qrec-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
        if not self.nonce:
            self.nonce = hashlib.sha256((seed + ":nonce").encode("utf-8")).hexdigest()[:32]
        if not self.receipt_sha256:
            self.receipt_sha256 = self.sha256()

    def deterministic_core(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "backend_id": self.backend_id,
            "provider": self.provider,
            "native_gateset": self.native_gateset,
            "queue_duration_sec": round(float(self.queue_duration_sec), 6),
            "execution_duration_sec": round(float(self.execution_duration_sec), 6),
            "calibration_sha256": self.calibration_sha256.lower(),
            "shots": int(self.shots),
            "execution_timestamp": self.execution_timestamp,
            "result_sha256": self.result_sha256.lower(),
            "evidence_class": self.evidence_class,
            "provider_job_id": self.provider_job_id,
        }

    def signable_body(self) -> Dict[str, Any]:
        body = self.deterministic_core()
        body["receipt_id"] = self.receipt_id
        body["nonce"] = self.nonce
        body["timestamp"] = self.execution_timestamp
        return body

    def sha256(self) -> str:
        return canonical_sha256(self.deterministic_core())

    def sign(self, secret_key: str) -> "QuantumExecutionReceipt":
        self.receipt_sha256 = self.sha256()
        payload = canonical_json_dumps(self.signable_body()).encode("utf-8")
        self.signature_hmac_sha256 = hmac.new(
            secret_key.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()
        return self

    def verify_integrity(self, secret_key: Optional[str] = None) -> bool:
        try:
            if not self.execution_id or not self.backend_id or not self.provider or int(self.shots) <= 0:
                return False
            if self.evidence_class == "SIMULATED" and self.provider_job_id is not None:
                return False
            if self.evidence_class == "HARDWARE" and not self.provider_job_id:
                return False
            if not hmac.compare_digest(self.sha256(), self.receipt_sha256):
                return False
            if secret_key is not None:
                if not self.signature_hmac_sha256:
                    return False
                expected = hmac.new(
                    secret_key.encode("utf-8"),
                    canonical_json_dumps(self.signable_body()).encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                if not hmac.compare_digest(self.signature_hmac_sha256, expected):
                    return False
            return True
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        out = self.signable_body()
        out["receipt_sha256"] = self.receipt_sha256
        out["signature_hmac_sha256"] = self.signature_hmac_sha256
        return out


def call_with_retry_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay_sec: float = 0.05,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Execute `fn` with Retry-After / jittered exponential backoff on transient HTTP codes."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in NON_RETRYABLE_HTTP_CODES or "AuthenticationError" in type(exc).__name__:
                raise QNexusGateError(f"Non-retryable Nexus API error ({status}): {exc}") from exc
            if status not in TRANSIENT_HTTP_CODES or attempt >= max_retries:
                raise QNexusGateError(f"Nexus API call failed after {attempt} retries: {exc}") from exc
            retry_after = getattr(exc, "retry_after", None)
            delay = float(retry_after) if retry_after is not None else base_delay_sec * (2**attempt)
            delay += random.uniform(0.0, 0.25 * base_delay_sec)
            sleep_fn(delay)
            attempt += 1


class NexusSubmissionGate:
    """Pre/post-compile hash-locked submission gate for Quantinuum Nexus (`H2-1SC` / `H2-1E` / `H2-1`)."""

    def __init__(
        self,
        expected_precompile_sha256: str,
        expected_isa_sha256: str,
        hash_fn: Callable[[Any], str],
    ) -> None:
        self.expected_precompile_sha256 = expected_precompile_sha256
        self.expected_isa_sha256 = expected_isa_sha256
        self.hash_fn = hash_fn

    def verify_manifest_lock(self, manifest: Dict[str, Any]) -> None:
        if not manifest.get("hash_lock", False) or not manifest.get("compiler_lock", False):
            raise ContractViolationError("Manifest must enforce both hash_lock=True and compiler_lock=True")
        if int(manifest.get("optimisation_level", -1)) != 0:
            raise ContractViolationError("Manifest must lock Nexus server-side optimisation_level=0")
        if manifest.get("precompile_sha256") != self.expected_precompile_sha256:
            raise HashMismatchError("Manifest precompile_sha256 deviates from frozen baseline")
        if manifest.get("isa_sha256") != self.expected_isa_sha256:
            raise HashMismatchError("Manifest isa_sha256 deviates from frozen H2-1 ISA baseline")

    def dispatch_hash_locked(
        self,
        *,
        pre_circuit: Any,
        isa_circuit: Any,
        manifest: Dict[str, Any],
        qnx_adapter: Any,
        device_name: str = "H2-1SC",
        shots: int = 1000,
        syntax_check_only: bool = False,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> Dict[str, Any]:
        """Verify pre/ISA hashes, compile on Nexus at opt_level=0, verify zero ISA drift, then execute."""
        self.verify_manifest_lock(manifest)

        obs_pre = self.hash_fn(pre_circuit)
        if obs_pre != self.expected_precompile_sha256:
            raise HashMismatchError(f"Pre-compile hash drift: {obs_pre} != {self.expected_precompile_sha256}")

        obs_isa = self.hash_fn(isa_circuit)
        if obs_isa != self.expected_isa_sha256:
            raise HashMismatchError(f"Offline H2-1 ISA hash drift: {obs_isa} != {self.expected_isa_sha256}")

        uploaded_ref = call_with_retry_backoff(
            lambda: qnx_adapter.upload_circuit(isa_circuit, name="wen_plaquette_h2_isa"),
            sleep_fn=sleep_fn,
        )
        compile_job_id, compiled_circ = call_with_retry_backoff(
            lambda: qnx_adapter.compile_circuit(uploaded_ref, device_name=device_name, optimisation_level=0),
            sleep_fn=sleep_fn,
        )

        nexus_isa_hash = self.hash_fn(compiled_circ)
        if nexus_isa_hash != self.expected_isa_sha256:
            raise HashMismatchError(
                f"Nexus compile job {compile_job_id} altered frozen H2-1 ISA ({nexus_isa_hash}); aborting execute!"
            )

        if syntax_check_only or device_name.endswith("SC"):
            return {
                "status": "SYNTAX_CHECK_PASSED",
                "device_name": device_name,
                "compile_job_id": compile_job_id,
                "execute_job_id": None,
                "isa_sha256": nexus_isa_hash,
            }

        exec_job_id, counts = call_with_retry_backoff(
            lambda: qnx_adapter.execute_circuit(compiled_circ, device_name=device_name, shots=shots),
            sleep_fn=sleep_fn,
        )
        receipt = QuantumExecutionReceipt(
            backend_id=device_name,
            provider="Quantinuum",
            shots=shots,
            result_sha256=canonical_sha256(counts),
            evidence_class="HARDWARE" if device_name == "H2-1" else "SIMULATED",
            provider_job_id=exec_job_id if device_name == "H2-1" else None,
        )
        return {
            "status": "EXECUTED",
            "device_name": device_name,
            "compile_job_id": compile_job_id,
            "execute_job_id": exec_job_id,
            "isa_sha256": nexus_isa_hash,
            "counts": counts,
            "receipt": receipt.to_dict(),
        }

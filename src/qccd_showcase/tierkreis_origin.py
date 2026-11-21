"""Typed Tierkreis computational graph serializer & Quantum Origin Mermin-M3 entropy verifier.

Implements:
  1. `TierkreisGraph` (`TKRRef[T]`, `guppy_worker`, `pytket_worker`, `nexus_worker`, `quantinuum_worker`)
     with canonical JSON export and deterministic SHA-256 hashing.
  2. Fail-closed 3-qubit GHZ Mermin-M3 device-independent min-entropy verifier
     (`M3 = <XXX> - <XYY> - <YXY> - <YYX>`, Foreman et al., Quantum 7, 969, 2023)
     enforcing M3_lower_bound > 2.0 at 5 sigma and SHAKE-256 seed extraction.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, List, Mapping, Optional, TypeVar

from .canonical import canonical_json_dumps, canonical_sha256

T = TypeVar("T")

MERMIN_REQUIRED_BASES = ("XXX", "XYY", "YXY", "YYX")
VALID_3BIT_OUTCOMES = frozenset(f"{i:03b}" for i in range(8))
CLASSICAL_MERMIN_BOUND = 2.0
DEFAULT_SIGNING_KEY = "qccd-showcase-default-hmac-key"


class TierkreisWorkerKind(str, Enum):
    BUILTIN = "builtin"
    GUPPY_WORKER = "guppy_worker"
    PYTKET_WORKER = "pytket_worker"
    NEXUS_WORKER = "nexus_worker"
    QUANTINUUM_WORKER = "quantinuum_worker"
    AER_WORKER = "aer_worker"


@dataclass(frozen=True)
class TKRRef(Generic[T]):
    node_id: str
    port_name: str = "value"
    type_name: str = "Any"

    def to_dict(self) -> Dict[str, str]:
        return {"node_id": self.node_id, "port_name": self.port_name, "type_name": self.type_name}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TKRRef[Any]":
        return cls(
            node_id=str(data["node_id"]),
            port_name=str(data.get("port_name", "value")),
            type_name=str(data.get("type_name", "Any")),
        )


def _encode_val(val: Any) -> Any:
    if isinstance(val, TKRRef):
        return {"__tkr_ref__": val.to_dict()}
    if isinstance(val, dict):
        return {str(k): _encode_val(v) for k, v in sorted(val.items())}
    if isinstance(val, list):
        return [_encode_val(v) for v in val]
    return val


def _decode_val(val: Any) -> Any:
    if isinstance(val, dict):
        if set(val.keys()) == {"__tkr_ref__"} and isinstance(val["__tkr_ref__"], dict):
            return TKRRef.from_dict(val["__tkr_ref__"])
        return {str(k): _decode_val(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_decode_val(v) for v in val]
    return val


@dataclass
class TierkreisTaskNode:
    node_id: str
    worker: TierkreisWorkerKind
    function_name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "worker": self.worker.value,
            "function_name": self.function_name,
            "inputs": {str(k): _encode_val(self.inputs[k]) for k in sorted(self.inputs.keys())},
            "metadata": {str(k): _encode_val(self.metadata[k]) for k in sorted(self.metadata.keys())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TierkreisTaskNode":
        raw_in = data.get("inputs") or {}
        return cls(
            node_id=str(data["node_id"]),
            worker=TierkreisWorkerKind(str(data["worker"])),
            function_name=str(data["function_name"]),
            inputs={str(k): _decode_val(v) for k, v in raw_in.items()},
            metadata=dict(data.get("metadata") or {}),
        )


class TierkreisGraph:
    """Deterministic Tierkreis hybrid quantum-classical DAG (`tierkreis-graph/v1`)."""

    SCHEMA_VERSION = "tierkreis-graph/v1"

    def __init__(self, name: str = "qccd_hybrid_workflow") -> None:
        self.name = name
        self.tasks: Dict[str, TierkreisTaskNode] = {}
        self.outputs: Dict[str, TKRRef[Any]] = {}
        self._counter = 0

    def input_ref(self, param_name: str, type_name: str = "Any") -> TKRRef[Any]:
        return TKRRef(node_id="__inputs__", port_name=param_name, type_name=type_name)

    def task(
        self,
        worker: TierkreisWorkerKind,
        function_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TKRRef[Any]:
        self._counter += 1
        nid = node_id or f"tkr_{self._counter}_{function_name}"
        if nid in self.tasks:
            raise ValueError(f"Duplicate Tierkreis node_id '{nid}'")
        self.tasks[nid] = TierkreisTaskNode(
            node_id=nid,
            worker=worker,
            function_name=function_name,
            inputs=dict(inputs or {}),
            metadata=dict(metadata or {}),
        )
        return TKRRef(node_id=nid, port_name="output", type_name="dict")

    def finish_with_outputs(self, **outputs: TKRRef[Any]) -> "TierkreisGraph":
        self.outputs = dict(outputs)
        return self

    def to_dict(self) -> Dict[str, Any]:
        sorted_ids = sorted(self.tasks.keys())
        edges: List[Dict[str, str]] = []
        for nid in sorted_ids:
            tnode = self.tasks[nid]
            for in_key in sorted(tnode.inputs.keys()):
                val = tnode.inputs[in_key]
                if isinstance(val, TKRRef) and val.node_id != "__inputs__":
                    edges.append(
                        {
                            "upstream": val.node_id,
                            "downstream": nid,
                            "input_key": str(in_key),
                            "port_name": val.port_name,
                        }
                    )
        return {
            "schema": self.SCHEMA_VERSION,
            "name": self.name,
            "nodes": [self.tasks[nid].to_dict() for nid in sorted_ids],
            "edges": edges,
            "tasks": {nid: self.tasks[nid].to_dict() for nid in sorted_ids},
            "outputs": {k: self.outputs[k].to_dict() for k in sorted(self.outputs.keys())},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_canonical_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_json(cls, raw_json: str) -> "TierkreisGraph":
        data = json.loads(raw_json)
        g = cls(name=str(data["name"]))
        for nid, tdata in sorted((data.get("tasks") or {}).items()):
            g.tasks[nid] = TierkreisTaskNode.from_dict(tdata)
        g._counter = len(g.tasks)
        g.outputs = {k: TKRRef.from_dict(v) for k, v in sorted((data.get("outputs") or {}).items())}
        return g


class QuantumOriginValidationError(ValueError):
    """Raised when Quantum Origin Mermin-M3 evidence fails closed during verification."""


def evaluate_mermin_m3_counts(
    raw_counts: Mapping[str, Any],
    *,
    confidence_z: float = 5.0,
    min_shots_per_basis: int = 64,
) -> Dict[str, Any]:
    """Evaluate 3-qubit GHZ Mermin-M3 correlators directly from raw un-postselected counts."""
    if not isinstance(raw_counts, Mapping) or not raw_counts:
        raise QuantumOriginValidationError("Missing or empty counts mapping")
    missing = [b for b in MERMIN_REQUIRED_BASES if b not in raw_counts]
    if missing:
        raise QuantumOriginValidationError(f"Missing Mermin bases: {missing}")
    extra = [k for k in raw_counts.keys() if k not in MERMIN_REQUIRED_BASES]
    if extra:
        raise QuantumOriginValidationError(f"Unexpected measurement bases: {extra}")

    correlators: Dict[str, float] = {}
    variances: Dict[str, float] = {}
    shots_by_basis: Dict[str, int] = {}

    for basis in MERMIN_REQUIRED_BASES:
        b_counts = raw_counts[basis]
        if not isinstance(b_counts, Mapping) or not b_counts:
            raise QuantumOriginValidationError(f"Empty counts for basis '{basis}'")
        total_shots = 0
        weighted_sum = 0
        for bitstr, cnt in b_counts.items():
            if bitstr not in VALID_3BIT_OUTCOMES:
                raise QuantumOriginValidationError(f"Invalid 3-bit string '{bitstr}' in '{basis}'")
            if isinstance(cnt, bool) or not isinstance(cnt, int) or cnt < 0:
                raise QuantumOriginValidationError(f"Invalid shot count for '{basis}:{bitstr}'")
            total_shots += cnt
            weighted_sum += (-1 if (bitstr.count("1") % 2 == 1) else 1) * cnt
        if total_shots < min_shots_per_basis:
            raise QuantumOriginValidationError(f"Insufficient shots ({total_shots}) in '{basis}'")
        exp_val = weighted_sum / float(total_shots)
        correlators[basis] = exp_val
        variances[basis] = max(1e-12, (1.0 - exp_val * exp_val) / float(total_shots))
        shots_by_basis[basis] = total_shots

    m3_signed = correlators["XXX"] - correlators["XYY"] - correlators["YXY"] - correlators["YYX"]
    m3_abs = abs(m3_signed)
    std_err = math.sqrt(sum(variances.values()))
    m3_lower = m3_abs - confidence_z * std_err
    z_sigma = (m3_abs - CLASSICAL_MERMIN_BOUND) / max(1e-12, std_err)

    if m3_lower <= CLASSICAL_MERMIN_BOUND:
        certified = False
        h_min = 0.0
    else:
        norm_v = min(1.0, max(0.0, (m3_lower - CLASSICAL_MERMIN_BOUND) / 2.0))
        p_guess = 0.5 + 0.5 * math.sqrt(max(0.0, 1.0 - norm_v * norm_v))
        h_min = max(0.0, min(1.0, -math.log2(max(0.5, min(1.0, p_guess)))))
        certified = h_min > 0.0

    return {
        "correlators": {k: round(v, 6) for k, v in correlators.items()},
        "shots_by_basis": shots_by_basis,
        "total_shots": sum(shots_by_basis.values()),
        "m3_raw": round(m3_abs, 6),
        "m3_std_error": round(std_err, 6),
        "m3_lower_bound": round(m3_lower, 6),
        "confidence_z": round(confidence_z, 3),
        "z_sigma_above_classical": round(z_sigma, 3),
        "certified_min_entropy_per_round": round(h_min, 6),
        "certified": certified,
    }


class QuantumOriginIngestor:
    """Fail-closed builder and verifier for Quantinuum Quantum Origin Mermin-M3 bundles."""

    SUPPORTED_PROVIDERS = frozenset(
        {"Quantinuum Quantum Origin", "Quantinuum", "Quantinuum Selene Emulator"}
    )

    @staticmethod
    def synthesize_ghz_counts(shots_per_setting: int = 2048, visibility: float = 0.965) -> Dict[str, Dict[str, int]]:
        if shots_per_setting < 64 or not (0.0 <= visibility <= 1.0):
            raise ValueError("Invalid shots_per_setting or visibility")
        good_total = int(round(shots_per_setting * (1.0 + visibility) / 2.0))
        bad_total = shots_per_setting - good_total

        def split4(tot: int) -> List[int]:
            q, rem = divmod(tot, 4)
            return [q + (1 if i < rem else 0) for i in range(4)]

        g, b = split4(good_total), split4(bad_total)
        evens, odds = ["000", "011", "101", "110"], ["001", "010", "100", "111"]
        counts: Dict[str, Dict[str, int]] = {
            "XXX": {**{evens[i]: g[i] for i in range(4)}, **{odds[i]: b[i] for i in range(4)}}
        }
        for basis in ("XYY", "YXY", "YYX"):
            counts[basis] = {**{odds[i]: g[i] for i in range(4)}, **{evens[i]: b[i] for i in range(4)}}
        return counts

    @classmethod
    def build_bundle(
        cls,
        raw_counts: Mapping[str, Mapping[str, int]],
        *,
        provider: str = "Quantinuum Quantum Origin",
        evidence_class: str = "SIMULATED",
        job_id: Optional[str] = None,
        seed_bytes_length: int = 64,
        confidence_z: float = 5.0,
        signing_key: str = DEFAULT_SIGNING_KEY,
    ) -> Dict[str, Any]:
        if evidence_class == "SIMULATED" and job_id is not None:
            raise QuantumOriginValidationError("SIMULATED bundle must not carry a fabricated job_id")
        if evidence_class == "HARDWARE" and (not job_id or not str(job_id).strip()):
            raise QuantumOriginValidationError("HARDWARE bundle requires a non-empty job_id")

        eval_res = evaluate_mermin_m3_counts(raw_counts, confidence_z=confidence_z)
        if not eval_res["certified"]:
            raise QuantumOriginValidationError(f"Uncertified counts: M3_lower={eval_res['m3_lower_bound']} <= 2.0")

        canonical_counts = {
            basis: {k: int(v) for k, v in sorted(raw_counts[basis].items())}
            for basis in MERMIN_REQUIRED_BASES
        }
        raw_counts_sha256 = canonical_sha256(canonical_counts)
        seed = hashlib.shake_256(f"QUANTUM_ORIGIN_M3:{provider}:{raw_counts_sha256}".encode("utf-8")).digest(
            seed_bytes_length
        )
        seed_sha256 = hashlib.sha256(seed).hexdigest()

        receipt_body: Dict[str, Any] = {
            "schema": "qccd-quantum-origin-receipt/v1",
            "provider": provider,
            "job_id": job_id,
            "evidence_class": evidence_class,
            "raw_counts_sha256": raw_counts_sha256,
            "seed_num_bytes": seed_bytes_length,
            "seed_sha256": seed_sha256,
            "m3_raw": eval_res["m3_raw"],
            "m3_lower_bound": eval_res["m3_lower_bound"],
            "confidence_z": eval_res["confidence_z"],
            "certified_min_entropy_per_round": eval_res["certified_min_entropy_per_round"],
        }
        receipt_sha256 = canonical_sha256(receipt_body)
        sig_hex = hmac.new(
            signing_key.encode("utf-8"),
            canonical_json_dumps(receipt_body).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        receipt = {**receipt_body, "receipt_sha256": receipt_sha256, "signature_hmac_sha256": sig_hex}
        return {
            "provider": provider,
            "job_id": job_id,
            "counts": canonical_counts,
            "receipt": receipt,
            "signature": sig_hex,
        }

    @classmethod
    def ingest(cls, bundle: Mapping[str, Any], *, signing_key: str = DEFAULT_SIGNING_KEY) -> Dict[str, Any]:
        if not isinstance(bundle, Mapping):
            raise QuantumOriginValidationError("Bundle must be a mapping")
        provider = bundle.get("provider")
        if provider not in cls.SUPPORTED_PROVIDERS:
            raise QuantumOriginValidationError(f"Unsupported provider: {provider!r}")
        counts = bundle.get("counts") or bundle.get("raw_counts")
        receipt = bundle.get("receipt")
        sig = bundle.get("signature")
        if not isinstance(counts, Mapping) or not isinstance(receipt, Mapping) or not isinstance(sig, str):
            raise QuantumOriginValidationError("Bundle requires 'counts', 'receipt', and 'signature'")

        job_id = bundle.get("job_id")
        ev_class = str(receipt.get("evidence_class", ""))
        if ev_class == "SIMULATED" and job_id is not None:
            raise QuantumOriginValidationError("SIMULATED bundle must not carry a fabricated job_id")
        if ev_class == "HARDWARE" and not job_id:
            raise QuantumOriginValidationError("HARDWARE bundle requires job_id")
        if receipt.get("job_id") != job_id or receipt.get("provider") != provider:
            raise QuantumOriginValidationError("Envelope/receipt metadata mismatch")

        eval_res = evaluate_mermin_m3_counts(counts, confidence_z=float(receipt.get("confidence_z", 5.0)))
        if not eval_res["certified"]:
            raise QuantumOriginValidationError("Mermin-M3 bound failed")

        canonical_counts = {
            basis: {k: int(v) for k, v in sorted(counts[basis].items())}
            for basis in MERMIN_REQUIRED_BASES
        }
        if not hmac.compare_digest(canonical_sha256(canonical_counts), str(receipt.get("raw_counts_sha256", ""))):
            raise QuantumOriginValidationError("Tampered raw_counts_sha256")
        if abs(float(receipt.get("m3_lower_bound", 0.0)) - eval_res["m3_lower_bound"]) > 1e-5:
            raise QuantumOriginValidationError("Tampered m3_lower_bound")

        signable = {
            k: v for k, v in receipt.items() if k not in ("receipt_sha256", "signature_hmac_sha256", "signature")
        }
        if not hmac.compare_digest(canonical_sha256(signable), str(receipt.get("receipt_sha256", ""))):
            raise QuantumOriginValidationError("Tampered receipt_sha256")

        expected_sig = hmac.new(
            signing_key.encode("utf-8"),
            canonical_json_dumps(signable).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            raise QuantumOriginValidationError("Invalid HMAC signature")

        return {"verified": True, "provider": provider, "evidence_class": ev_class, "evaluation": eval_res}


def validate_origin_receipt(bundle: Mapping[str, Any], *, signing_key: str = DEFAULT_SIGNING_KEY) -> bool:
    try:
        QuantumOriginIngestor.ingest(bundle, signing_key=signing_key)
        return True
    except Exception:
        return False

"""pytket CustomPass for 20-qubit Wen-plaquette feed-forward preparation & offline H2-1 ISA lowering.

Implements the [[16, 2, 4]] periodic checkerboard Wen-plaquette state-preparation protocol
(prior art: Iqbal et al., arXiv:2302.01917 on Quantinuum H1; Quantinuum feed-forward tutorial)
as a hardened, hash-locked pytket compilation pass targeting Quantinuum H2-1 (56 qubits):
  - 16 data qubits (q[0..15]) + 4 recycled ancilla qubits (q[16..19]) = 20 physical qubits.
  - 24 mid-circuit/terminal Measure ops, 8 mid-circuit Reset ops, 20 classical Conditional Z ops.
  - Explicit big-endian syndrome -> pytket little-endian condition_value encoding & audit log.
  - Strictly distinguishes single-shot codespace state preparation from multi-round QEC memory.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple

PACKAGE_VERSION = "1.1.1"
FROZEN_PRECOMPILE_SHA256 = "b86aeabd70eecb7f5fbca5d0e1688e06d71dc358fd9fbaaab7fd8e4f9634230b"
FROZEN_H2_ISA_SHA256 = "02d5cb72849f9b473c022e68023486fdecc8c9f7fc3da9a9a9299c3482f31062"
EXPECTED_H2_DEPTH = 19
EXPECTED_H2_ZZPHASE = 32

N_DATA = 16
N_ANCILLA = 4
N_TOTAL = N_DATA + N_ANCILLA  # 20

RESERVED_Q_REGISTERS = frozenset({"q", "qr", "quantum"})
RESERVED_C_REGISTERS = frozenset(
    {"c", "creg", "cl", "classical", "c0_1st", "c1_1st", "c2_1st", "c3_1st", "c4_2nd", "c5_end"}
)

LOCKED_COMPILER_METADATA: Dict[str, Any] = {
    "package_version": PACKAGE_VERSION,
    "compiler_lock": True,
    "hash_locked": True,
}


class EndiannessPolicy(str, Enum):
    """Explicit bit-ordering contract for classical condition registers."""

    TKET_LITTLE_ENDIAN = "tket_little_endian"  # c_reg[0] is LSB (2^0)
    QISKIT_BIG_ENDIAN = "qiskit_big_endian"    # string[0] is MSB


@dataclass(frozen=True)
class PauliStabilizer:
    pauli_type: Literal["X", "Z"]
    support: Tuple[int, ...]

    def __repr__(self) -> str:
        return ".".join(f"{self.pauli_type}{i}" for i in self.support)


@dataclass
class EndiannessAuditEntry:
    syndrome_be: str
    tket_condition_value: int
    target_qubits: List[int]
    parity_valid: bool
    register_name: str


@dataclass
class CompilationProvenanceCapsule:
    capsule_version: str = "qccd.tket.v1"
    pass_name: str = "WenPlaquetteFeedForwardPass"
    package_version: str = PACKAGE_VERSION
    compiler_lock: bool = True
    hash_locked: bool = True
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    target_device: str = "H2-1"
    measurement_setting: Literal["X", "Z"] = "X"
    n_data_qubits: int = N_DATA
    n_ancilla_qubits: int = N_ANCILLA
    pre_pass_circuit_sha256: str = ""
    post_pass_circuit_sha256: str = ""
    classical_registers: Dict[str, int] = field(default_factory=dict)
    endianness_policy: str = EndiannessPolicy.TKET_LITTLE_ENDIAN.value
    endianness_audit: List[EndiannessAuditEntry] = field(default_factory=list)
    conditional_op_count: int = 0
    reset_op_count: int = 0
    capsule_sha256: str = ""

    def finalize(self) -> "CompilationProvenanceCapsule":
        payload = asdict(self)
        payload.pop("capsule_sha256", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.capsule_sha256 = hashlib.sha256(canonical).hexdigest()
        return self


WEN_STABILIZERS: List[PauliStabilizer] = [
    PauliStabilizer("Z", (0, 1, 4, 5)),      # 0
    PauliStabilizer("X", (1, 2, 5, 6)),      # 1  (Round 2)
    PauliStabilizer("Z", (2, 3, 6, 7)),      # 2
    PauliStabilizer("X", (3, 0, 7, 4)),      # 3  (Round 2)
    PauliStabilizer("X", (4, 5, 8, 9)),      # 4  (Round 1)
    PauliStabilizer("Z", (5, 6, 9, 10)),     # 5
    PauliStabilizer("X", (6, 7, 10, 11)),    # 6  (Round 1)
    PauliStabilizer("Z", (7, 4, 11, 8)),     # 7
    PauliStabilizer("Z", (8, 9, 12, 13)),    # 8
    PauliStabilizer("X", (9, 10, 13, 14)),   # 9  (Round 2)
    PauliStabilizer("Z", (10, 11, 14, 15)),  # 10
    PauliStabilizer("X", (11, 8, 15, 12)),   # 11 (Round 2)
    PauliStabilizer("X", (12, 13, 0, 1)),    # 12 (Round 1)
    PauliStabilizer("Z", (13, 14, 1, 2)),    # 13
    PauliStabilizer("X", (14, 15, 2, 3)),    # 14 (Round 1)
    PauliStabilizer("Z", (15, 12, 3, 0)),    # 15
]

ROUND_1_MAP: Dict[int, Tuple[int, int, int]] = {
    4: (5, 16, 0),
    6: (7, 17, 1),
    12: (13, 18, 2),
    14: (15, 19, 3),
}

ROUND_2_MAP: Dict[int, Tuple[int, int]] = {
    1: (16, 0),
    3: (17, 1),
    9: (18, 2),
    11: (19, 3),
}

ROUND_2_LUT_BE: Dict[str, List[int]] = {
    "0011": [10, 11],
    "0101": [4, 8],
    "0110": [0, 13],
    "1001": [2, 15],
    "1010": [5, 9],
    "1100": [6, 7],
    "1111": [2, 3, 14, 15],
}


def encode_syndrome_condition(
    syndrome_be: str,
    reg_size: int,
    policy: EndiannessPolicy = EndiannessPolicy.TKET_LITTLE_ENDIAN,
    enforce_even_parity: bool = True,
) -> Tuple[int, bool]:
    """Encode a big-endian syndrome string into pytket's little-endian condition_value."""
    if len(syndrome_be) != reg_size or any(ch not in ("0", "1") for ch in syndrome_be):
        raise ValueError(f"Invalid syndrome '{syndrome_be}': expected {reg_size}-bit binary string.")
    ones_count = syndrome_be.count("1")
    parity_valid = (ones_count % 2) == 0
    if enforce_even_parity and not parity_valid:
        raise ValueError(
            f"Topological parity violation: syndrome '{syndrome_be}' has odd anyon count ({ones_count})."
        )
    cond_val = int(syndrome_be[::-1], 2) if policy == EndiannessPolicy.TKET_LITTLE_ENDIAN else int(syndrome_be, 2)
    return cond_val, parity_valid


def verify_lut_and_commutativity() -> Dict[str, Any]:
    """Pure-Python algebraic audit of the 16 Wen-plaquette stabilizers and 7-entry Round-2 LUT."""
    for i, s_a in enumerate(WEN_STABILIZERS):
        for j, s_b in enumerate(WEN_STABILIZERS):
            if i >= j or s_a.pauli_type == s_b.pauli_type:
                continue
            overlap = len(set(s_a.support) & set(s_b.support))
            if overlap % 2 != 0:
                raise ValueError(f"Stabilizers {i} and {j} anticommute (overlap={overlap})")

    r2_plaquettes = [WEN_STABILIZERS[idx] for idx in (1, 3, 9, 11)]
    for syn_be, z_targets in ROUND_2_LUT_BE.items():
        z_set = set(z_targets)
        induced = "".join(
            "1" if (len(set(p.support) & z_set) % 2 == 1) else "0"
            for p in r2_plaquettes
        )
        if induced != syn_be:
            raise ValueError(f"LUT mismatch for {syn_be}: targets {z_targets} induce {induced}")
        # Also confirm zero side-effect on Round-1 X plaquettes (4, 6, 12, 14)
        for r1_idx in (4, 6, 12, 14):
            if len(set(WEN_STABILIZERS[r1_idx].support) & z_set) % 2 != 0:
                raise ValueError(f"LUT targets {z_targets} disturb Round-1 plaquette {r1_idx}")

    return {
        "stabilizer_count": len(WEN_STABILIZERS),
        "all_commute": True,
        "round2_lut_entries": len(ROUND_2_LUT_BE),
        "lut_syndromes_verified": True,
    }


def _normalize_phase_str(raw_phase: Any) -> str:
    try:
        val = float(raw_phase)
    except (TypeError, ValueError):
        return str(raw_phase)
    if not math.isfinite(val):
        raise ValueError(f"Non-finite circuit phase: {raw_phase}")
    mod_val = val % 2.0
    if abs(mod_val) < 1e-12 or abs(mod_val - 2.0) < 1e-12:
        return "0.0"
    return f"{mod_val:.12g}"


def normalize_circuit_metadata(circ: Any, compiler_lock: bool = True) -> Any:
    """Normalize mutable pytket Circuit name/phase/metadata to prevent digest drift."""
    if circ.name is not None and circ.name != "":
        circ.name = ""
    try:
        p_val = float(circ.phase) % 2.0
        if abs(p_val) < 1e-12 or abs(p_val - 2.0) < 1e-12:
            p_val = 0.0
        if float(circ.phase) != p_val:
            circ.add_phase(p_val - float(circ.phase))
    except Exception:
        pass
    target_meta = dict(LOCKED_COMPILER_METADATA) if compiler_lock else {}
    try:
        setattr(circ, "metadata", target_meta)
    except (AttributeError, TypeError):
        pass
    return circ


def canonical_circuit_dict(circ: Any) -> Dict[str, Any]:
    d = dict(circ.to_dict())
    d.pop("name", None)
    d.pop("metadata", None)
    if "phase" in d:
        d["phase"] = _normalize_phase_str(d["phase"])
    return d


def hash_tket_circuit(circ: Any) -> str:
    """Deterministic SHA-256 digest of a pytket Circuit."""
    canonical = json.dumps(
        canonical_circuit_dict(circ), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class WenPlaquetteFeedForwardPass:
    """Preparation-only pytket CustomPass builder for 20-qubit Wen-plaquette codespace preparation."""

    def __init__(
        self,
        measurement_setting: Literal["X", "Z"] = "X",
        target_device: str = "H2-1",
        policy: EndiannessPolicy = EndiannessPolicy.TKET_LITTLE_ENDIAN,
        compiler_lock: bool = True,
        strict_registers: bool = False,
    ) -> None:
        self.measurement_setting = measurement_setting
        self.target_device = target_device
        self.policy = policy
        self.compiler_lock = compiler_lock
        self.strict_registers = strict_registers
        self.last_capsule: CompilationProvenanceCapsule | None = None

    def _check_input_contract(self, circ: Any) -> None:
        n_cmd = len(circ.get_commands())
        if n_cmd != 0:
            raise ValueError(
                f"WenPlaquetteFeedForwardPass is preparation-only: input must be empty, got {n_cmd} command(s)."
            )
        for qreg in circ.q_registers:
            if qreg.name == "q":
                if self.strict_registers:
                    raise ValueError('Input already declares quantum register "q" under strict_registers=True.')
                if qreg.size != N_TOTAL:
                    raise ValueError(f'Register "q" has size {qreg.size}; expected {N_TOTAL}.')
            elif qreg.name in RESERVED_Q_REGISTERS:
                raise ValueError(f'Colliding quantum register "{qreg.name}".')
        for creg in circ.c_registers:
            if creg.name in RESERVED_C_REGISTERS:
                raise ValueError(f'Colliding classical register "{creg.name}".')

    @staticmethod
    def _measure_stabilizer_nondestructive(
        circ: Any,
        q_reg: Any,
        stab: PauliStabilizer,
        c_bit: Any,
        anc_qubit: Any,
    ) -> None:
        from pytket import OpType  # type: ignore

        for i in stab.support:
            if stab.pauli_type == "X":
                circ.H(q_reg[i])
                circ.CX(q_reg[i], anc_qubit)
                circ.H(q_reg[i])
            elif stab.pauli_type == "Z":
                circ.CX(q_reg[i], anc_qubit)
        circ.Measure(anc_qubit, c_bit)
        circ.add_gate(OpType.Reset, [anc_qubit])

    def _transform(self, circ: Any) -> Any:
        from pytket import Circuit, OpType  # type: ignore
        from pytket.circuit import BitRegister, QubitRegister  # type: ignore

        self._check_input_contract(circ)
        pre_hash = hash_tket_circuit(circ)
        out = Circuit()

        q_reg = QubitRegister("q", N_TOTAL)
        out.add_q_register(q_reg)

        c_regs_1st = []
        for idx in range(4):
            reg = BitRegister(f"c{idx}_1st", 1)
            out.add_c_register(reg)
            c_regs_1st.append(reg)

        c_reg_2nd = BitRegister("c4_2nd", 4)
        out.add_c_register(c_reg_2nd)
        c_reg_end = BitRegister("c5_end", N_TOTAL)
        out.add_c_register(c_reg_end)

        audit_log: List[EndiannessAuditEntry] = []

        for stab_idx, (_, anc_idx, c_idx) in ROUND_1_MAP.items():
            self._measure_stabilizer_nondestructive(
                out, q_reg, WEN_STABILIZERS[stab_idx], c_regs_1st[c_idx][0], q_reg[anc_idx]
            )

        for _, (target_q, _, c_idx) in ROUND_1_MAP.items():
            cond_val, _ = encode_syndrome_condition(
                "1", reg_size=1, policy=self.policy, enforce_even_parity=False
            )
            out.Z(q_reg[target_q], condition_bits=c_regs_1st[c_idx], condition_value=cond_val)
            audit_log.append(
                EndiannessAuditEntry("1", cond_val, [target_q], True, c_regs_1st[c_idx].name)
            )

        for stab_idx, (anc_idx, c_bit_idx) in ROUND_2_MAP.items():
            self._measure_stabilizer_nondestructive(
                out, q_reg, WEN_STABILIZERS[stab_idx], c_reg_2nd[c_bit_idx], q_reg[anc_idx]
            )

        for syndrome_be, target_qubits in ROUND_2_LUT_BE.items():
            cond_val, parity_ok = encode_syndrome_condition(
                syndrome_be, reg_size=4, policy=self.policy, enforce_even_parity=True
            )
            for q_idx in target_qubits:
                out.Z(q_reg[q_idx], condition_bits=c_reg_2nd, condition_value=cond_val)
            audit_log.append(
                EndiannessAuditEntry(syndrome_be, cond_val, list(target_qubits), parity_ok, c_reg_2nd.name)
            )

        out.add_barrier([q_reg[i] for i in range(N_TOTAL)])
        for q_idx in range(N_DATA):
            if self.measurement_setting == "X":
                out.H(q_reg[q_idx])
            out.Measure(q_reg[q_idx], c_reg_end[q_idx])

        cond_ops = [cmd for cmd in out.get_commands() if cmd.op.type == OpType.Conditional]
        reset_ops = [cmd for cmd in out.get_commands() if cmd.op.type == OpType.Reset]
        expected_cond = len(ROUND_1_MAP) + sum(len(qs) for qs in ROUND_2_LUT_BE.values())
        if len(cond_ops) != expected_cond:
            raise RuntimeError(f"Conditional op count mismatch: expected {expected_cond}, got {len(cond_ops)}.")

        normalize_circuit_metadata(out, compiler_lock=self.compiler_lock)
        self.last_capsule = CompilationProvenanceCapsule(
            compiler_lock=self.compiler_lock,
            hash_locked=self.compiler_lock,
            target_device=self.target_device,
            measurement_setting=self.measurement_setting,
            pre_pass_circuit_sha256=pre_hash,
            post_pass_circuit_sha256=hash_tket_circuit(out),
            classical_registers={reg.name: reg.size for reg in out.c_registers},
            endianness_policy=self.policy.value,
            endianness_audit=audit_log,
            conditional_op_count=len(cond_ops),
            reset_op_count=len(reset_ops),
        ).finalize()
        return out

    def to_tket_pass(self) -> Any:
        from pytket.passes import CustomPass  # type: ignore

        def _apply(c: Any) -> Any:
            built = self._transform(c)
            normalize_circuit_metadata(built, compiler_lock=self.compiler_lock)
            return built

        return CustomPass(_apply, label="WenPlaquetteFeedForwardPass")


def compile_h2_offline(optimisation_level: int = 2) -> Dict[str, Any]:
    """Build the 20-qubit Wen-plaquette circuit and compile offline to Quantinuum H2-1 ISA.

    Uses `QuantinuumBackend("H2-1", api_handler=QuantinuumAPIOffline())` (no credentials or
    network calls) and verifies bit-for-bit parity against the frozen manifest hashes.
    """
    from pytket import Circuit  # type: ignore
    from pytket.extensions.quantinuum import QuantinuumBackend  # type: ignore
    from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPIOffline  # type: ignore

    builder = WenPlaquetteFeedForwardPass(measurement_setting="X", target_device="H2-1")
    pre_circ = Circuit()
    builder.to_tket_pass().apply(pre_circ)
    pre_hash = hash_tket_circuit(pre_circ)

    backend = QuantinuumBackend("H2-1", api_handler=QuantinuumAPIOffline())
    isa_circ = backend.get_compiled_circuit(pre_circ.copy(), optimisation_level=optimisation_level)
    isa_hash = hash_tket_circuit(isa_circ)
    gate_counts = dict(sorted(Counter(cmd.op.type.name for cmd in isa_circ.get_commands()).items()))

    return {
        "target_backend": "H2-1",
        "mode": "QuantinuumAPIOffline",
        "n_qubits": isa_circ.n_qubits,
        "n_clbits": isa_circ.n_bits,
        "precompile_sha256": pre_hash,
        "precompile_matches_frozen": pre_hash == FROZEN_PRECOMPILE_SHA256,
        "isa_sha256": isa_hash,
        "isa_matches_frozen": isa_hash == FROZEN_H2_ISA_SHA256,
        "depth": isa_circ.depth(),
        "zzphase_count": gate_counts.get("ZZPhase", 0),
        "gate_counts": gate_counts,
    }

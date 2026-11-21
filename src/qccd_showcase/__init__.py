"""qccd_showcase: Fault-tolerant QCCD compilation, decoding, and provenance showcase."""

from .canonical import (
    canonical_json_dumps,
    canonical_sha256,
    terhal_vollbrecht_ef_ebits,
    wilson_lower_bound_95,
)
from .guppy_selene import run_hp004_qccd_benchmark
from .nexus_gate import (
    ContractViolationError,
    HashMismatchError,
    NexusSubmissionGate,
    QNexusGateError,
    QuantumExecutionReceipt,
    QuantumHardwareTarget,
)
from .qldpc_bposd import (
    BPOSDResult,
    build_bivariate_bicycle_72_12_6,
    build_helios_workload_summary,
    build_steane_7_1_3,
    build_tesseract_16_6_4,
    css_logical_operators,
    decode_bp_osd,
    estimate_hqc,
)
from .tierkreis_origin import (
    QuantumOriginIngestor,
    QuantumOriginValidationError,
    TKRRef,
    TierkreisGraph,
    TierkreisWorkerKind,
    evaluate_mermin_m3_counts,
    validate_origin_receipt,
)
from .tket_passes import (
    FROZEN_H2_ISA_SHA256,
    FROZEN_PRECOMPILE_SHA256,
    ROUND_2_LUT_BE,
    WEN_STABILIZERS,
    WenPlaquetteFeedForwardPass,
    compile_h2_offline,
    encode_syndrome_condition,
)

__version__ = "1.0.0"

__all__ = [
    "BPOSDResult",
    "ContractViolationError",
    "FROZEN_H2_ISA_SHA256",
    "FROZEN_PRECOMPILE_SHA256",
    "HashMismatchError",
    "NexusSubmissionGate",
    "QNexusGateError",
    "QuantumExecutionReceipt",
    "QuantumHardwareTarget",
    "QuantumOriginIngestor",
    "QuantumOriginValidationError",
    "ROUND_2_LUT_BE",
    "TKRRef",
    "TierkreisGraph",
    "TierkreisWorkerKind",
    "WEN_STABILIZERS",
    "WenPlaquetteFeedForwardPass",
    "build_bivariate_bicycle_72_12_6",
    "build_helios_workload_summary",
    "build_steane_7_1_3",
    "build_tesseract_16_6_4",
    "canonical_json_dumps",
    "canonical_sha256",
    "compile_h2_offline",
    "css_logical_operators",
    "decode_bp_osd",
    "encode_syndrome_condition",
    "estimate_hqc",
    "evaluate_mermin_m3_counts",
    "run_hp004_qccd_benchmark",
    "terhal_vollbrecht_ef_ebits",
    "validate_origin_receipt",
    "wilson_lower_bound_95",
]

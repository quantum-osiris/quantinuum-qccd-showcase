"""Tests for CSS/qLDPC algebra, Sum-Product BP+OSD-CS decoding, and Helios/H2 HQC estimation."""

import numpy as np
import pytest

from qccd_showcase.qldpc_bposd import (
    build_bivariate_bicycle_72_12_6,
    build_helios_workload_summary,
    build_steane_7_1_3,
    build_tesseract_16_6_4,
    css_logical_operators,
    decode_bp_osd,
    estimate_hqc,
)


def test_css_codes_algebra_and_helios_96q_footprint() -> None:
    summary = build_helios_workload_summary()
    codes = summary["verified_css_codes"]

    assert codes["Steane_7_1_3"]["k"] == 1
    assert codes["Steane_7_1_3"]["exhaustive_weight1_recovery"] == "7/7"

    assert codes["Tesseract_16_6_4"]["k"] == 6
    assert codes["Tesseract_16_6_4"]["exhaustive_weight1_recovery"] == "16/16"

    bb = codes["BivariateBicycle_72_12_6"]
    assert bb["n"] == 72 and bb["k"] == 12
    assert bb["rank_hx"] == 30 and bb["rank_hz"] == 30
    assert bb["max_check_weight"] == 6 and bb["max_qubit_degree"] == 3
    assert bb["exhaustive_weight1_recovery"] == "72/72"

    # 72 data + 24 recycled ancillas = 96 ions <= 98 physical ions on Helios-1
    wp3b = [w for w in summary["workloads"] if w["work_package"] == "WP3b"][0]
    assert wp3b["physical_qubits"] == 96
    assert summary["total_quantinuum_circuits"] == 191
    assert summary["total_quantinuum_shots"] == 67124
    assert summary["total_estimated_hqc"] == pytest.approx(21169.8, abs=0.5)


def test_bivariate_bicycle_weight2_bp_osd_recovery() -> None:
    hx, hz = build_bivariate_bicycle_72_12_6()
    _, lz = css_logical_operators(hx, hz)
    rng = np.random.default_rng(42)

    # Deterministic sample of 120 distinct weight-2 X faults on [[72, 12, 6]] (d=6 corrects all weight <= 2)
    pairs = set()
    while len(pairs) < 120:
        i, j = sorted(rng.choice(72, size=2, replace=False).tolist())
        pairs.add((i, j))

    for i, j in sorted(pairs):
        err = np.zeros(72, dtype=np.uint8)
        err[i] = 1
        err[j] = 1
        syn = (hx @ err) % 2
        res = decode_bp_osd(hx, syn, channel_probs=0.002)
        residual = (err ^ res.correction) % 2
        assert np.all((hx @ residual) % 2 == 0)
        assert np.all((lz @ residual) % 2 == 0)


def test_hqc_formula_virtual_rz_cost() -> None:
    # 0 gates, 0 meas -> base 5.0 HQC
    assert estimate_hqc(0, 0, 0, shots=1000) == pytest.approx(5.0)
    # 500 shots * (100 + 10*20 + 5*20) / 5000 = 5 + 500 * 400 / 5000 = 45.0 HQC
    assert estimate_hqc(100, 20, 20, shots=500) == pytest.approx(45.0)

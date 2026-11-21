"""Tests for 8Q 9-basis QCCD vs Heavy-Hex stabilizer benchmark & Terhal-Vollbrecht E_F bounds."""

import pytest

from qccd_showcase.canonical import terhal_vollbrecht_ef_ebits
from qccd_showcase.guppy_selene import run_hp004_qccd_benchmark


def test_terhal_vollbrecht_d4_concordance_table() -> None:
    # Exact numerical concordance with published Zenodo 10.5281/zenodo.24287505 (E-94):
    assert terhal_vollbrecht_ef_ebits(0.25) == 0.0
    assert terhal_vollbrecht_ef_ebits(0.633423) == pytest.approx(0.845131, abs=2e-6)
    assert terhal_vollbrecht_ef_ebits(0.683910) == pytest.approx(1.000000, abs=5e-5)
    assert terhal_vollbrecht_ef_ebits(0.723700) == pytest.approx(1.124434, abs=2e-6)
    assert terhal_vollbrecht_ef_ebits(0.730713) == pytest.approx(1.146530, abs=2e-6)
    assert terhal_vollbrecht_ef_ebits(1.0) == pytest.approx(2.0, abs=1e-9)


def test_hp004_five_arms_and_negative_controls() -> None:
    res = run_hp004_qccd_benchmark(shots_per_basis=512, mode="builtin", seed=20261119)
    assert res["evidence_class"] == "SIMULATED"
    assert res["hardware"] is False

    arm_a = res["arms"]["Arm A"]
    arm_b = res["arms"]["Arm B"]
    arm_ctrl = res["arms"]["Arm B-Ctrl"]
    arm_def = res["arms"]["Arm B-Def"]
    arm_held = res["arms"]["Arm B-Held"]

    # Positive arms witness SN >= 3 and E_F > 1 ebit with 37.5%-42.1% fewer 2Q gates than Heavy-Hex
    assert arm_a["swap_gate_reduction_pct"] == pytest.approx(42.11, abs=0.02)
    assert arm_b["swap_gate_reduction_pct"] == pytest.approx(37.50, abs=0.02)
    assert arm_a["f_joint_qccd_emulated"] > 0.95
    assert arm_b["f_joint_qccd_emulated"] > 0.95
    assert arm_b["ef_lower_bound_qccd_ebits"] > 1.80

    # Negative controls remain strictly below the 0.5000 Schmidt-number-2 ceiling
    for neg in (arm_ctrl, arm_def, arm_held):
        assert neg["schmidt_number_ge_3"] is False
        assert neg["exceeds_1ebit_threshold"] is False
    assert arm_held["f_joint_qccd_emulated"] < 0.12

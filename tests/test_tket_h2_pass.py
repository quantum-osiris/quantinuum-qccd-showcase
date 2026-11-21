"""Tests for Wen-plaquette stabilizer/LUT algebra, endianness encoding, and offline H2-1 compilation."""

import pytest

from qccd_showcase.canonical import min_successes_for_wilson_threshold
from qccd_showcase.tket_passes import (
    EXPECTED_H2_DEPTH,
    EXPECTED_H2_ZZPHASE,
    FROZEN_H2_ISA_SHA256,
    FROZEN_PRECOMPILE_SHA256,
    ROUND_2_LUT_BE,
    WenPlaquetteFeedForwardPass,
    compile_h2_offline,
    encode_syndrome_condition,
    hash_tket_circuit,
    verify_lut_and_commutativity,
)


def test_wen_stabilizer_commutativity_and_lut_algebra() -> None:
    res = verify_lut_and_commutativity()
    assert res["stabilizer_count"] == 16
    assert res["all_commute"] is True
    assert res["round2_lut_entries"] == 7
    assert res["lut_syndromes_verified"] is True


def test_big_endian_to_tket_little_endian_encoding() -> None:
    val, parity_ok = encode_syndrome_condition("0011", reg_size=4)
    assert val == 12  # int("1100", 2)
    assert parity_ok is True
    assert len(ROUND_2_LUT_BE) == 7

    with pytest.raises(ValueError, match="odd anyon count"):
        encode_syndrome_condition("0001", reg_size=4, enforce_even_parity=True)


def test_wilson_95_hardware_acceptance_thresholds() -> None:
    expected = {512: 512, 1000: 997, 2000: 1989, 5000: 4964, 10000: 9920}
    for n, k_exp in expected.items():
        assert min_successes_for_wilson_threshold(n, threshold=0.99) == k_exp


def test_pytket_pass_and_offline_h2_isa_parity_when_installed() -> None:
    pytket = pytest.importorskip("pytket")
    pytest.importorskip("pytket.extensions.quantinuum")

    # Nonempty circuit rejected fail-closed
    nonempty = pytket.Circuit(20)
    nonempty.H(0)
    builder = WenPlaquetteFeedForwardPass()
    with pytest.raises(ValueError, match="preparation-only"):
        builder.to_tket_pass().apply(nonempty)

    # Colliding classical register rejected fail-closed
    colliding = pytket.Circuit()
    colliding.add_c_register("c4_2nd", 4)
    with pytest.raises(ValueError, match="Colliding classical register"):
        builder.to_tket_pass().apply(colliding)

    # Clean empty circuit reproduces exact Zenodo-archived precompile & H2-1 ISA SHA-256s
    res = compile_h2_offline(optimisation_level=2)
    assert res["precompile_sha256"] == FROZEN_PRECOMPILE_SHA256
    assert res["isa_sha256"] == FROZEN_H2_ISA_SHA256
    assert res["depth"] == EXPECTED_H2_DEPTH
    assert res["zzphase_count"] == EXPECTED_H2_ZZPHASE

    # Metadata normalization keeps SHA-256 invariant
    c = pytket.Circuit()
    builder.to_tket_pass().apply(c)
    assert hash_tket_circuit(c) == FROZEN_PRECOMPILE_SHA256

"""Tests for Tierkreis graph canonical hashing, Quantum Origin Mermin-M3 verifier, and CLI export."""

from pathlib import Path
import json
import pytest

from qccd_showcase.cli import main as cli_main
from qccd_showcase.tierkreis_origin import (
    DEFAULT_SIGNING_KEY,
    QuantumOriginIngestor,
    QuantumOriginValidationError,
    TierkreisGraph,
    TierkreisWorkerKind,
    validate_origin_receipt,
)


def test_tierkreis_graph_roundtrip_and_canonical_sha256() -> None:
    g1 = TierkreisGraph(name="test_dag")
    inp = g1.input_ref("shots", "int")
    t1 = g1.task(TierkreisWorkerKind.PYTKET_WORKER, "compile", {"shots": inp, "opt": 2}, node_id="n1")
    t2 = g1.task(TierkreisWorkerKind.GUPPY_WORKER, "run_selene", {"circ": t1}, node_id="n2")
    g1.finish_with_outputs(result=t2)

    restored = TierkreisGraph.from_json(g1.to_json())
    assert restored.sha256() == g1.sha256()
    assert restored.to_canonical_json() == g1.to_canonical_json()


def test_quantum_origin_mermin_m3_fail_closed_invariants() -> None:
    # 1. Valid high-visibility GHZ counts pass (> 2.0 at 5 sigma)
    good_counts = QuantumOriginIngestor.synthesize_ghz_counts(shots_per_setting=2048, visibility=0.96)
    bundle = QuantumOriginIngestor.build_bundle(good_counts, signing_key=DEFAULT_SIGNING_KEY)
    assert validate_origin_receipt(bundle, signing_key=DEFAULT_SIGNING_KEY) is True

    # 2. Classical LHV counts (visibility=0.40 -> M3_raw=1.60 <= 2.0) rejected fail-closed
    classical_counts = QuantumOriginIngestor.synthesize_ghz_counts(shots_per_setting=2048, visibility=0.40)
    with pytest.raises(QuantumOriginValidationError, match="M3_lower"):
        QuantumOriginIngestor.build_bundle(classical_counts)

    # 3. Tampered counts rejected
    tampered = json.loads(json.dumps(bundle))
    tampered["counts"]["XXX"]["000"] += 1
    assert validate_origin_receipt(tampered, signing_key=DEFAULT_SIGNING_KEY) is False

    # 4. Fabricated job_id on SIMULATED bundle rejected
    with pytest.raises(QuantumOriginValidationError, match="fabricated job_id"):
        QuantumOriginIngestor.build_bundle(good_counts, evidence_class="SIMULATED", job_id="fake-123")


def test_cli_verify_all_and_export_artifacts(tmp_path: Path) -> None:
    rc = cli_main(["verify-all", "--export-dir", str(tmp_path)])
    assert rc == 0
    for expected in (
        "h2_wen_plaquette_manifest.json",
        "hp004_qccd_vs_heavy_hex.json",
        "helios_hqc_bposd_workload.json",
        "tierkreis_origin_capsule.json",
    ):
        fp = tmp_path / expected
        assert fp.exists()
        data = json.loads(fp.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and "schema" in data or "hqc_formula" in data

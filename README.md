# `quantinuum-qccd-showcase`

**Fault-Tolerant QCCD Compilation, Hash-Locked `qnexus` Dispatch, `Guppy`/`Selene` Benchmarking, `BP+OSD` qLDPC Decoding, and `Tierkreis` / `Quantum Origin` Provenance**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Zenodo QEC-002](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24424726-0F766E)](https://doi.org/10.5281/zenodo.24424726)
[![Zenodo HP-004](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24287505-1D4ED8)](https://doi.org/10.5281/zenodo.24287505)

A standalone, zero-credential reference repository demonstrating end-to-end compiler, error-correction, and provenance tooling across Quantinuum's modern software and QCCD hardware stack (`pytket`, `pytket-quantinuum` offline `H2-1` ISA lowering, `qnexus`, `guppylang` / `selene-sim`, `Tierkreis` DAG serialization, `BP+OSD` hypergraph decoding for `Helios-1`, and `Quantum Origin` Mermin-$M_3$ entropy verification).

---

## Methodological & Provenance Boundaries

1. **Strict `SIMULATED` vs. `HARDWARE` Evidence Separation:** Every execution receipt (`QuantumExecutionReceipt` and `QuantumOriginIngestor`) enforces fail-closed provenance invariants—`SIMULATED` receipts reject non-null `provider_job_id`s, and `HARDWARE` receipts require an authoritative provider job ID. All QCCD execution receipts committed in [`artifacts/`](artifacts/) are explicitly marked `SIMULATED` or `QuantinuumAPIOffline`; archived physical QPU baselines (`ibm_fez` and `ibm_marrakesh`, 978,944 un-postselected shots across [`10.5281/zenodo.24287505`](https://doi.org/10.5281/zenodo.24287505) and [`10.5281/zenodo.24293756`](https://doi.org/10.5281/zenodo.24293756)) are cited with their exact job IDs.
2. **Codespace Preparation $\neq$ Fault-Tolerant Memory:** The 20-qubit `[[16, 2, 4]]` toroidal Wen-plaquette pass (`WenPlaquetteFeedForwardPass`, implementing the two-round mid-circuit X-measurement + 7-entry LUT protocol of Iqbal *et al.*, [`arXiv:2302.01917`](https://arxiv.org/abs/2302.01917)) is a **state-preparation pass**, not a multi-round QEC memory. Pre-preparation weight-1 $Z$ errors on all 16 data qubits are corrected (`16/16`), whereas post-preparation errors are detected (`16/16` for $Y, Z$) without subsequent syndrome-extraction rounds.
3. **Retired Legacy API Handling:** Following the sunset of the legacy `pytket-quantinuum` job-submission API (`>= 0.56`), `pytket-quantinuum` is used strictly for offline `H2-1` ISA compilation (`QuantinuumAPIOffline`), while cloud compilation/execution is routed through `qnexus` (`NexusSubmissionGate`) with server-side `optimisation_level=0` ISA-drift aborts.

---

## Architecture & Module Map

| Module | Stack Target | Technical Scope & Verified Invariants |
| :--- | :--- | :--- |
| [`tket_passes.py`](src/qccd_showcase/tket_passes.py) | `pytket` `CustomPass` & `QuantinuumAPIOffline` (`H2-1`) | Hardened 20Q/28-clbit `[[16, 2, 4]]` Wen-plaquette preparation pass (`24 Measure`, `8 Reset`, `20 Conditional`), big-endian $\to$ little-endian LUT audit, register-collision guards, and frozen `H2-1` ISA parity (`depth = 19`, `32 ZZPhase`, `72 PhasedX`, SHA-256 `02d5cb72...`). |
| [`nexus_gate.py`](src/qccd_showcase/nexus_gate.py) | `qnexus` (`H2-1SC`, `H2-1E`, `H2-1`, `Helios-1`) | Pre/post-compile SHA-256 lock gate aborting before `execute` if server compilation mutates the frozen ISA; `Retry-After` jittered backoff on transient HTTP `{408, 429, 500, 502, 503, 504}`; HMAC/SHA-256 execution receipts. |
| [`guppy_selene.py`](src/qccd_showcase/guppy_selene.py) | `@guppy` (HUGR) + `selene-sim` (`Stim`) | 8-qubit, 9-basis (`{X,Y,Z}_1 \times {X,Y,Z}_2`) Hayden–Preskill Bell-stabilizer benchmark across 5 pre-registered arms. Contrasts QCCD zero-SWAP 2Q gate counts (`20` vs. `32` Heavy-Hex CZs = **37.5% reduction**; `22` vs. `38` = **42.1% reduction**) and Terhal–Vollbrecht $E_F(\rho) \ge \gamma_4(F)$ bounds against `ibm_marrakesh` (`0.845 ebits`) and `ibm_fez` (`1.147 ebits`). |
| [`qldpc_bposd.py`](src/qccd_showcase/qldpc_bposd.py) | `Helios-1` (98Q QCCD) & `H2-1` | Exact GF(2) CSS constructors and logical-operator verifiers for Steane `[[7,1,3]]`, 4D Tesseract `[[16,6,4]]`, and Bravyi *et al.* Bivariate-Bicycle `[[72,12,6]]` (72 data + 24 recycled ancillas = **96/98 physical ions on Helios-1**), paired with a pure-NumPy Sum-Product `BP+OSD-CS` decoder (`72/72` weight-1 and `2,556/2,556` weight-2 recovery) and HQC estimator (`21,169.8 HQC` across 191 circuits). |
| [`tierkreis_origin.py`](src/qccd_showcase/tierkreis_origin.py) | `Tierkreis` DAG & `Quantum Origin` | Deterministic `TierkreisGraph` (`TKRRef[T]`, `pytket_worker`, `guppy_worker`, `nexus_worker`, `quantinuum_worker`) canonical JSON serializer and fail-closed 3-qubit GHZ Mermin-$M_3$ certifier ($M_{3,\text{lower}} > 2.0$ at $5\sigma$, Foreman *et al.*, *Quantum* 7, 969, 2023) with SHAKE-256 seed extraction. |

---

## Quickstart

### 1. Install

```bash
# Core + test suite (pure Python + NumPy; runs anywhere in <2s)
python -m pip install -e ".[dev]"

# Optional: enable live offline H2-1 ISA compilation & qnexus client bindings
python -m pip install -e ".[quantinuum]"

# Optional: enable native @guppy HUGR compilation + selene_sim.Stim execution
python -m pip install -e ".[guppy]"
```

### 2. Run the CLI & Regenerate Verified Artifacts

```bash
# Execute all 4 pipelines and export JSON receipts to artifacts/
qccd-showcase verify-all --export-dir artifacts

# Individual subcommands
qccd-showcase h2-compile
qccd-showcase benchmark-hp004
qccd-showcase bposd-hqc
qccd-showcase origin-capsule
```

### 3. Run the Test Suite

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest -v
```

---

## Reproduced Reference Artifacts ([`artifacts/`](artifacts/))

- [`artifacts/h2_wen_plaquette_manifest.json`](artifacts/h2_wen_plaquette_manifest.json) — Frozen pre-compile (`b86aeabd...`) and `H2-1` ISA (`02d5cb72...`) SHA-256 digests, `QuantinuumAPIOffline` gate counts, and 95% Wilson score acceptance gates (`> 0.99`).
- [`artifacts/hp004_qccd_vs_heavy_hex.json`](artifacts/hp004_qccd_vs_heavy_hex.json) — 5-arm 8Q 9-basis QCCD stabilizer correlators and Terhal–Vollbrecht $E_F$ lower bounds alongside archived `ibm_marrakesh` (`dbti1fbsbsbs73cb2vu0`) and `ibm_fez` (`dbtj61bnlcms73ajc380`) hardware references.
- [`artifacts/helios_hqc_bposd_workload.json`](artifacts/helios_hqc_bposd_workload.json) — 6-work-package `H2-1` / `Helios-1` HQC budget (`191` circuits, `67,124` shots, `21,169.8 HQC`) and verified CSS/`BP+OSD` parameters.
- [`artifacts/tierkreis_origin_capsule.json`](artifacts/tierkreis_origin_capsule.json) — Canonical `TierkreisGraph` DAG (`tierkreis-graph/v1`), signed QCCD execution receipt, and verified 3-qubit GHZ Mermin-$M_3$ `Quantum Origin` bundle.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

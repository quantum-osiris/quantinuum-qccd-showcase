"""Routing-free CSS/qLDPC codes, Sum-Product BP+OSD-CS decoder, and Quantinuum HQC cost model.

Implements:
  1. GF(2) CSS verification and parity-check constructors for Steane [[7,1,3]],
     4D Tesseract [[16,6,4]], and Bravyi et al. (2024) Bivariate-Bicycle [[72,12,6]]
     (72 data + 24 recycled ancilla ions = 96 physical ions inside 98-ion Helios-1).
  2. Pure-NumPy log-domain Sum-Product Belief Propagation + Combination-Sweep OSD (`BP+OSD-CS`).
  3. Quantinuum H-Series / Helios Hardware Quantum Credit (HQC) estimator:
       HQC = 5.0 + shots * (N_1q + 10 * N_2q + 5 * N_m) / 5000.0   (virtual RZ = 0 HQC).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def gf2_rank(matrix: np.ndarray) -> int:
    """Exact rank of a binary matrix over GF(2)."""
    a = (np.asarray(matrix, dtype=np.uint8) & 1).copy()
    rows, cols = a.shape
    r = 0
    for c in range(cols):
        pivots = np.nonzero(a[r:, c])[0]
        if pivots.size == 0:
            continue
        piv = r + int(pivots[0])
        if piv != r:
            a[[r, piv]] = a[[piv, r]]
        for rr in np.nonzero(a[:, c])[0]:
            if rr != r:
                a[rr] ^= a[r]
        r += 1
        if r == rows:
            break
    return r


def gf2_nullspace(h_mat: np.ndarray) -> np.ndarray:
    """Return basis rows for ker(H) over GF(2)."""
    a = (np.asarray(h_mat, dtype=np.uint8) & 1).copy()
    rows, cols = a.shape
    pivot_cols: List[int] = []
    r = 0
    for c in range(cols):
        pivots = np.nonzero(a[r:, c])[0]
        if pivots.size == 0:
            continue
        piv = r + int(pivots[0])
        if piv != r:
            a[[r, piv]] = a[[piv, r]]
        for rr in np.nonzero(a[:, c])[0]:
            if rr != r:
                a[rr] ^= a[r]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break

    free_cols = [c for c in range(cols) if c not in pivot_cols]
    basis = np.zeros((len(free_cols), cols), dtype=np.uint8)
    for idx, fc in enumerate(free_cols):
        basis[idx, fc] = 1
        for pr, pc in enumerate(pivot_cols):
            if a[pr, fc]:
                basis[idx, pc] = 1
    return basis


def css_logical_operators(hx: np.ndarray, hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Verify CSS orthogonality (HX @ HZ^T == 0 mod 2) and extract logical X and Z bases."""
    if np.any((hx @ hz.T) % 2 != 0):
        raise ValueError("CSS orthogonality violated: HX @ HZ^T != 0 mod 2")

    def quotient_basis(ker_basis: np.ndarray, stab_rows: np.ndarray) -> np.ndarray:
        cur = (stab_rows.astype(np.uint8) & 1).copy()
        cur_rank = gf2_rank(cur)
        logicals: List[np.ndarray] = []
        for cand in ker_basis:
            trial = np.vstack([cur, cand[None, :]]) if cur.size else cand[None, :]
            nr = gf2_rank(trial)
            if nr > cur_rank:
                cur = trial
                cur_rank = nr
                logicals.append(cand.copy())
        return np.array(logicals, dtype=np.uint8)

    lz = quotient_basis(gf2_nullspace(hx), hz)
    lx = quotient_basis(gf2_nullspace(hz), hx)
    return lx, lz


def build_steane_7_1_3() -> Tuple[np.ndarray, np.ndarray]:
    """Steane [[7, 1, 3]] color code (non-planar CNOT Tanner graph; 0 SWAPs on QCCD)."""
    h = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    return h.copy(), h.copy()


def build_tesseract_16_6_4() -> Tuple[np.ndarray, np.ndarray]:
    """4D hypercube RM(1,4) CSS [[16, 6, 4]] Tesseract code (5 independent weight-8 checks)."""
    coords = np.array([[(i >> b) & 1 for b in range(4)] for i in range(16)], dtype=np.uint8)
    facets = []
    for axis in range(4):
        facets.append((coords[:, axis] == 0).astype(np.uint8))
        facets.append((coords[:, axis] == 1).astype(np.uint8))
    h_indep = np.vstack([facets[0], facets[1], facets[2], facets[4], facets[6]])
    return h_indep.copy(), h_indep.copy()


def build_bivariate_bicycle_72_12_6() -> Tuple[np.ndarray, np.ndarray]:
    """Bravyi et al. (Nature 627, 778, 2024) [[72, 12, 6]] weight-6 Bivariate-Bicycle qLDPC code."""
    def shift(size: int, power: int) -> np.ndarray:
        m = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            m[i, (i + power) % size] = 1
        return m

    ix = np.eye(6, dtype=np.uint8)
    iy = np.eye(6, dtype=np.uint8)
    x = np.kron(shift(6, 1), iy).astype(np.uint8)
    y = np.kron(ix, shift(6, 1)).astype(np.uint8)
    x2 = (x @ x) % 2
    x3 = (x2 @ x) % 2
    y2 = (y @ y) % 2
    y3 = (y2 @ y) % 2

    a_poly = (x3 ^ y ^ y2).astype(np.uint8)
    b_poly = (y3 ^ x ^ x2).astype(np.uint8)
    hx = np.hstack([a_poly, b_poly]).astype(np.uint8)
    hz = np.hstack([b_poly.T, a_poly.T]).astype(np.uint8)
    return hx, hz


@dataclass(frozen=True)
class BPOSDResult:
    correction: np.ndarray
    bp_converged: bool
    bp_iterations: int
    used_osd: bool
    weight: int


def decode_bp_osd(
    h_mat: np.ndarray,
    syndrome: np.ndarray,
    channel_probs: Union[np.ndarray, float] = 0.01,
    max_iter: int = 30,
    damping: float = 0.20,
    osd_order: int = 12,
) -> BPOSDResult:
    """Log-domain Sum-Product BP + Combination-Sweep Ordered Statistics Decoder (`BP+OSD-CS`)."""
    h = np.asarray(h_mat, dtype=np.uint8) & 1
    s = (np.asarray(syndrome, dtype=np.uint8) & 1).reshape(-1)
    m_checks, n_vars = h.shape
    if s.shape[0] != m_checks:
        raise ValueError(f"Syndrome length {s.shape[0]} != number of checks {m_checks}")

    if np.all(s == 0):
        return BPOSDResult(np.zeros(n_vars, dtype=np.uint8), True, 0, False, 0)

    p = (
        np.full(n_vars, float(channel_probs), dtype=np.float64)
        if np.ndim(channel_probs) == 0
        else np.asarray(channel_probs, dtype=np.float64).reshape(n_vars)
    )
    p = np.clip(p, 1e-12, 0.499999)
    llr0 = np.log((1.0 - p) / p)

    check_nbrs = [np.nonzero(h[i])[0] for i in range(m_checks)]
    var_nbrs = [np.nonzero(h[:, j])[0] for j in range(n_vars)]

    m_cv = np.zeros((m_checks, n_vars), dtype=np.float64)
    m_vc = np.zeros((m_checks, n_vars), dtype=np.float64)
    for j in range(n_vars):
        m_vc[var_nbrs[j], j] = llr0[j]

    posterior_llr = llr0.copy()
    bp_cand: Optional[np.ndarray] = None
    bp_conv_iter = max_iter

    for it in range(1, max_iter + 1):
        for i in range(m_checks):
            nbrs = check_nbrs[i]
            if nbrs.size == 0:
                continue
            t_vals = np.tanh(0.5 * np.clip(m_vc[i, nbrs], -30.0, 30.0))
            t_vals = np.where(np.abs(t_vals) < 1e-15, 1e-15, t_vals)
            total_prod = (-1.0 if s[i] else 1.0) * float(np.prod(t_vals))
            for k_loc, j in enumerate(nbrs):
                extrinsic = np.clip(total_prod / float(t_vals[k_loc]), -0.999999999999, 0.999999999999)
                m_cv[i, j] = 2.0 * math.atanh(extrinsic)

        for j in range(n_vars):
            checks = var_nbrs[j]
            total_llr = llr0[j] + float(np.sum(m_cv[checks, j]))
            posterior_llr[j] = total_llr
            for i in checks:
                new_msg = total_llr - m_cv[i, j]
                m_vc[i, j] = (1.0 - damping) * new_msg + damping * m_vc[i, j]

        e_hat = (posterior_llr < 0.0).astype(np.uint8)
        if np.array_equal((h @ e_hat) % 2, s):
            bp_cand = e_hat.copy()
            bp_conv_iter = it
            if int(e_hat.sum()) <= 1:
                return BPOSDResult(bp_cand, True, it, False, int(bp_cand.sum()))
            break

    col_order = np.argsort(posterior_llr)
    h_perm = h[:, col_order].copy()
    aug = np.hstack([h_perm, s[:, None]]).astype(np.uint8)

    pivot_cols: List[int] = []
    r = 0
    for c in range(n_vars):
        pivots = np.nonzero(aug[r:, c])[0]
        if pivots.size == 0:
            continue
        piv = r + int(pivots[0])
        if piv != r:
            aug[[r, piv]] = aug[[piv, r]]
        for rr in np.nonzero(aug[:, c])[0]:
            if rr != r:
                aug[rr] ^= aug[r]
        pivot_cols.append(c)
        r += 1
        if r == m_checks:
            break

    rhs_piv = aug[:r, n_vars].copy()
    free_cols = [c for c in range(n_vars) if c not in pivot_cols]
    dep = aug[:r, free_cols] if free_cols else np.zeros((r, 0), dtype=np.uint8)
    prior_cost_perm = llr0[col_order]

    def eval_free(active: Sequence[int]) -> np.ndarray:
        e_perm = np.zeros(n_vars, dtype=np.uint8)
        piv_vals = rhs_piv.copy()
        for f_idx in active:
            e_perm[free_cols[f_idx]] = 1
            piv_vals ^= dep[:, f_idx]
        for p_idx, pc in enumerate(pivot_cols):
            e_perm[pc] = piv_vals[p_idx]
        return e_perm

    best_perm = eval_free(())
    best_nll = float(np.dot(best_perm, prior_cost_perm))

    k1 = min(len(free_cols), max(0, osd_order))
    k2 = min(len(free_cols), min(8, max(0, osd_order)))
    candidates: List[Tuple[int, ...]] = [(i,) for i in range(k1)]
    for i in range(k2):
        for j in range(i + 1, k2):
            candidates.append((i, j))

    for active in candidates:
        cand_perm = eval_free(active)
        nll = float(np.dot(cand_perm, prior_cost_perm))
        if nll < best_nll - 1e-12:
            best_perm = cand_perm
            best_nll = nll

    e_osd = np.zeros(n_vars, dtype=np.uint8)
    e_osd[col_order] = best_perm

    if bp_cand is not None:
        if float(np.dot(bp_cand, llr0)) <= best_nll + 1e-12:
            return BPOSDResult(bp_cand, True, bp_conv_iter, False, int(bp_cand.sum()))
        return BPOSDResult(e_osd, True, bp_conv_iter, True, int(e_osd.sum()))

    return BPOSDResult(e_osd, False, max_iter, True, int(e_osd.sum()))


def estimate_hqc(n_1q: float, n_2q: float, n_m: float, shots: int) -> float:
    """Compute Quantinuum H-Series / Helios Hardware Quantum Credits (virtual RZ = 0 HQC)."""
    if shots <= 0 or n_1q < 0 or n_2q < 0 or n_m < 0:
        raise ValueError("shots must be > 0 and gate/measurement counts non-negative")
    return 5.0 + (shots * (float(n_1q) + 10.0 * float(n_2q) + 5.0 * float(n_m))) / 5000.0


@dataclass(frozen=True)
class WorkloadRow:
    work_package: str
    protocol: str
    target_system: str
    physical_qubits: int
    circuits: int
    shots_per_circuit: int
    total_shots: int
    mean_n_1q: float
    mean_n_2q: float
    mean_n_m: float
    routing_swaps: int
    hqc_per_circuit: float
    total_hqc: float


WORKLOAD_SPEC_ROWS: Tuple[Tuple[str, str, str, int, int, int, float, float, float], ...] = (
    ("WP1a", "8Q k=2 9-basis joint stabilizer replication (zero SWAP)", "Quantinuum H2-1 / Helios-1", 8, 81, 300, 99.67, 33.33, 12.0),
    ("WP1b", "12Q k=3 all-to-all Schmidt-rank extension", "Quantinuum H2-1 / Helios-1", 12, 54, 300, 53.0, 30.0, 18.0),
    ("WP2a", "50Q 4-layer nonlocal debiased transport stability", "Quantinuum H2-1 / Helios-1", 50, 12, 512, 300.0, 100.0, 100.0),
    ("WP2b", "96Q 4-layer nonlocal debiased transport stability", "Quantinuum Helios-1 (98Q QCCD)", 96, 12, 512, 574.0, 192.0, 192.0),
    ("WP3a", "Steane [[7,1,3]] & Tesseract [[16,6,4]] R=3 mid-circuit QEC", "Quantinuum H2-1 / Helios-1", 21, 24, 512, 152.5, 156.0, 75.0),
    ("WP3b", "Bivariate-Bicycle [[72,12,6]] qLDPC R=2 (96Q Helios ancilla reuse)", "Quantinuum Helios-1 (98Q QCCD)", 96, 8, 256, 792.0, 864.0, 456.0),
)


def verify_exhaustive_weight1_recovery(hx: np.ndarray, lz: np.ndarray) -> Tuple[int, int]:
    """Verify BP+OSD recovers 100% of single-qubit X errors without logical Z flips."""
    n_qubits = hx.shape[1]
    recovered = 0
    for q in range(n_qubits):
        err = np.zeros(n_qubits, dtype=np.uint8)
        err[q] = 1
        syn = (hx @ err) % 2
        res = decode_bp_osd(hx, syn, channel_probs=0.002)
        residual = (err ^ res.correction) % 2
        if np.all((hx @ residual) % 2 == 0) and np.all((lz @ residual) % 2 == 0):
            recovered += 1
    return recovered, n_qubits


def build_helios_workload_summary() -> Dict[str, Any]:
    """Verify CSS codes + exhaustive weight-1 BP+OSD recovery and build the H2-1/Helios-1 HQC budget."""
    codes_summary: Dict[str, Any] = {}
    for name, builder in (
        ("Steane_7_1_3", build_steane_7_1_3),
        ("Tesseract_16_6_4", build_tesseract_16_6_4),
        ("BivariateBicycle_72_12_6", build_bivariate_bicycle_72_12_6),
    ):
        hx, hz = builder()
        lx, lz = css_logical_operators(hx, hz)
        rec, tot = verify_exhaustive_weight1_recovery(hx, lz)
        codes_summary[name] = {
            "n": int(hx.shape[1]),
            "k": int(lx.shape[0]),
            "rank_hx": gf2_rank(hx),
            "rank_hz": gf2_rank(hz),
            "max_check_weight": int(np.max(np.sum(hx, axis=1))),
            "max_qubit_degree": int(np.max(np.sum(hx, axis=0))),
            "exhaustive_weight1_recovery": f"{rec}/{tot}",
        }

    rows: List[Dict[str, Any]] = []
    for wp, proto, target, nq, n_circ, shots, n1q, n2q, nm in WORKLOAD_SPEC_ROWS:
        raw_hqc = estimate_hqc(n1q, n2q, nm, shots)
        hqc_c = round(raw_hqc, 3 if wp == "WP3b" else 2)
        tot_hqc = round(hqc_c * n_circ, 1)
        rows.append(
            asdict(
                WorkloadRow(
                    work_package=wp,
                    protocol=proto,
                    target_system=target,
                    physical_qubits=nq,
                    circuits=n_circ,
                    shots_per_circuit=shots,
                    total_shots=n_circ * shots,
                    mean_n_1q=n1q,
                    mean_n_2q=n2q,
                    mean_n_m=nm,
                    routing_swaps=0,
                    hqc_per_circuit=hqc_c,
                    total_hqc=tot_hqc,
                )
            )
        )

    return {
        "hqc_formula": "HQC = 5 + C * (N_1q + 10 * N_2q + 5 * N_m) / 5000 (rz virtual = 0)",
        "total_quantinuum_circuits": sum(r["circuits"] for r in rows),
        "total_quantinuum_shots": sum(r["total_shots"] for r in rows),
        "total_estimated_hqc": round(sum(r["total_hqc"] for r in rows), 1),
        "recommended_initial_hqc_allocation": 21500,
        "workloads": rows,
        "verified_css_codes": codes_summary,
    }

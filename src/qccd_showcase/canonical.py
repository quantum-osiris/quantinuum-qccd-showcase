"""Deterministic JSON canonicalization, SHA-256 digests, and statistical bounds."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def canonical_json_bytes(obj: Any, *, prefer_rfc8785: bool = False) -> bytes:
    """Serialize `obj` deterministically (RFC 8785 JCS when requested & installed, else sorted compact JSON)."""
    if prefer_rfc8785:
        try:
            import rfc8785  # type: ignore

            return rfc8785.dumps(obj)
        except ImportError:
            pass
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_dumps(obj: Any) -> str:
    """Return compact, sorted-key canonical JSON string (rejects NaN/Inf)."""
    return canonical_json_bytes(obj, prefer_rfc8785=False).decode("utf-8")


def canonical_sha256(obj: Any, *, prefer_rfc8785: bool = False) -> str:
    """Compute SHA-256 hex digest over deterministic canonical JSON."""
    return hashlib.sha256(canonical_json_bytes(obj, prefer_rfc8785=prefer_rfc8785)).hexdigest()


def is_valid_sha256_hex(value: Any) -> bool:
    """Return True iff `value` is a 64-character hexadecimal SHA-256 string."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in "0123456789abcdefABCDEF" for ch in value)


def wilson_lower_bound_95(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> float:
    """Compute the 95% one-sided/two-sided normal-z Wilson score lower confidence bound."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError(f"Invalid binomial counts: successes={successes}, total={total}")
    p_hat = successes / float(total)
    z2 = z * z
    denom = 1.0 + z2 / total
    center = p_hat + z2 / (2.0 * total)
    half_width = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total)
    return max(0.0, (center - half_width) / denom)


def min_successes_for_wilson_threshold(
    total: int,
    threshold: float = 0.99,
    z: float = 1.959963984540054,
) -> int:
    """Return minimum integer k in [0, total] whose Wilson lower bound strictly exceeds `threshold`."""
    for k in range(total + 1):
        if wilson_lower_bound_95(k, total, z=z) > threshold:
            return k
    raise ValueError(f"Threshold {threshold} unreachable at n={total}")


def _h4_isotropic(fidelity: float) -> float:
    """Raw Terhal-Vollbrecht entropy function g_4(F) for d=4 isotropic overlap F in [0.25, 1.0]."""
    if fidelity <= 0.25:
        return 0.0
    if fidelity >= 1.0:
        return 2.0
    p0 = 0.25 * (math.sqrt(fidelity) + math.sqrt(3.0 * (1.0 - fidelity))) ** 2
    p0 = min(1.0, max(0.25, p0))
    if p0 >= 1.0 - 1e-15:
        return 0.0
    rem = (1.0 - p0) / 3.0
    return -p0 * math.log2(p0) - 3.0 * rem * math.log2(rem)


def terhal_vollbrecht_ef_ebits(fidelity: float) -> float:
    """Convex-roof entanglement of formation lower bound gamma_4(F) (ebits) for 4x4 bipartite states.

    References:
      - Terhal & Vollbrecht, Phys. Rev. Lett. 85, 2625 (2000)
      - Vollbrecht & Werner, Phys. Rev. A 64, 062307 (2001)

    For d=4, g_4(F) = H_2(p_0(F), (1-p_0)/3 x 3) is convex on [1/4, F_tang] (F_tang ~= 0.858111)
    and continues along its tangent line from (F_tang, g_4(F_tang)) to (1.0, 2.0) above F_tang.
    """
    f = float(fidelity)
    if f <= 0.25:
        return 0.0
    if f >= 1.0:
        return 2.0
    # Tangency point where (2.0 - g_4(F_t)) / (1.0 - F_t) == g_4'(F_t) for d=4:
    f_tang = 0.858110885
    if f <= f_tang:
        return _h4_isotropic(f)
    g_tang = _h4_isotropic(f_tang)
    return g_tang + (2.0 - g_tang) * (f - f_tang) / (1.0 - f_tang)

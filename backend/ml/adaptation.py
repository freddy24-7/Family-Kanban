"""Per-household effort adjustment: label-shift correction with Bayes' rule.
See docs/learning/06-retraining.md, section 7.

For one household and category, from its most recent sprint reviews:
  pi_model(e)      = mean model probability of effort e on those reviewed tickets
  pi_household(e)  = actual efforts, smoothed toward pi_model with `prior_strength`
                     pseudo-counts (two reviews can't flip everything)
and a new prediction is re-weighted: p_adj(e) ∝ p_model(e) * pi_household(e) / pi_model(e).

Always computed from RAW model probabilities (before any adjustment), or the adjustment
would feed on itself.
"""

EFFORTS = ("S", "M", "L")
MIN_REVIEWS = 3
PRIOR_STRENGTH = 4.0
EPS = 1e-3


def segment_ratio(
    reviews: list[tuple[dict[str, float], str]], prior_strength: float = PRIOR_STRENGTH
) -> dict[str, float] | None:
    """reviews: (raw model probabilities, actual effort) for one household+category.
    Returns the multiplicative ratio per effort, or None with too few reviews."""
    if len(reviews) < MIN_REVIEWS:
        return None
    n = len(reviews)
    pi_model = {e: max(sum(p.get(e, 0.0) for p, _ in reviews) / n, EPS) for e in EFFORTS}
    total = sum(pi_model.values())
    pi_model = {e: v / total for e, v in pi_model.items()}
    counts = {e: sum(1 for _, actual in reviews if actual == e) for e in EFFORTS}
    pi_household = {
        e: (counts[e] + prior_strength * pi_model[e]) / (n + prior_strength) for e in EFFORTS
    }
    return {e: pi_household[e] / pi_model[e] for e in EFFORTS}


def adjust(probabilities: dict[str, float], ratio: dict[str, float]) -> dict[str, float]:
    raw = {e: probabilities.get(e, 0.0) * ratio.get(e, 1.0) for e in EFFORTS}
    total = sum(raw.values()) or 1.0
    return {e: v / total for e, v in raw.items()}

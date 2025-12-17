from __future__ import annotations

from app.models.track import TrackPublic


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _overlap_score(value: int, lo: int, hi: int) -> float:
    """
    Returns 1.0 if value in [lo, hi], else decreases as distance grows.
    """
    if lo <= value <= hi:
        return 1.0
    dist = min(abs(value - lo), abs(value - hi))
    # 0 at distance 100
    return max(0.0, 1.0 - dist / 100.0)


def pick_best_track(
    tracks: list[TrackPublic],
    goal: str,
    duration_min: int,
    energy: int,
    ambience: int,
    nature: str,
) -> TrackPublic:
    """
    Deterministic scoring to select best free track.
    No randomness so results are stable and cacheable.
    """
    g = goal.strip().lower().replace(" ", "_")
    n = nature.strip().lower()

    duration_sec = _clamp(duration_min, 1, 360) * 60

    best: TrackPublic | None = None
    best_score = -1.0

    for t in tracks:
        # Base
        score = 0.0

        # Goal match
        goal_tags = [x.lower() for x in (t.goalTags or [])]
        if g in goal_tags:
            score += 3.0
        elif goal_tags:
            score += 0.5  # has some tagging but not matching
        else:
            score += 0.2  # no tags, still eligible

        # Nature match
        nature_tags = [x.lower() for x in (t.natureTags or [])]
        if n in nature_tags:
            score += 2.0
        elif nature_tags:
            score += 0.3

        # Energy/Ambience fit
        score += 2.0 * _overlap_score(energy, t.energyMin, t.energyMax)
        score += 1.5 * _overlap_score(ambience, t.ambienceMin, t.ambienceMax)

        # Duration fit (prefer >= requested)
        # If track shorter than requested, penalize
        if t.durationSec >= duration_sec:
            score += 1.0
        else:
            score += max(0.0, 1.0 - (duration_sec - t.durationSec) / duration_sec)

        # Slight preference for more recent dates (lexicographic YYYY-MM-DD works)
        # (Small weight so it doesn’t dominate)
        score += 0.05 * _date_recency_bonus(t.date)

        if score > best_score:
            best_score = score
            best = t

    # Fallback
    if best is None:
        raise ValueError("No tracks available")
    return best


def _date_recency_bonus(date_str: str) -> float:
    """
    Cheap recency boost: maps YYYY-MM-DD to a float by simple parsing.
    Not critical; safe if parsing fails.
    """
    try:
        y, m, d = date_str.split("-")
        return float(int(y) * 10000 + int(m) * 100 + int(d))
    except Exception:
        return 0.0

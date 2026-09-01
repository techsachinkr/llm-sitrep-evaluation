"""Phase 4 metrics: operational completeness, visibility ceiling, figures, CIs, judge agreement.

Every function here is **pure** (no I/O, no config, no model calls) so the paper's headline
numbers can be unit-tested exactly — CLAUDE.md rule 5 ("metrics tested before trusted").
`src/judge_slots.py` supplies the judgement records; this module only does arithmetic.

Vocabulary
----------
*judgement*  one (sitrep, slot) verdict in {absent, partial, present} (see `VERDICT_SCORE`).
*coverage*   mean slot score for one slot over a set of sitreps (0..1).
*completeness*  mean slot score over the slots of one sitrep (or pooled over many).
*ceiling slot*  a slot human sitreps fill reliably and no machine arm fills at all.

Judgement inputs are accepted either as a mapping ``{slot_id: verdict}`` or as a sequence of
mappings with ``slot_id``/``verdict`` keys (extra keys such as ``sitrep_id`` are ignored), so
both a single sitrep and a pooled table can be scored by the same call.
"""
from __future__ import annotations

import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any, Callable

__all__ = [
    "VERDICTS", "VERDICT_SCORE", "verdict_score", "normalise_verdict",
    "operational_completeness", "coverage_by_slot", "SlotCoverage",
    "visibility_ceiling", "SlotCeiling",
    "extract_figures", "figure_extraction_check", "Figure", "FigureCheck",
    "bootstrap_ci", "BootstrapCI",
    "judge_agreement", "cohens_kappa", "Agreement",
]

# ---------------------------------------------------------------------------
# Verdict vocabulary
# ---------------------------------------------------------------------------
VERDICTS: tuple[str, str, str] = ("absent", "partial", "present")
VERDICT_SCORE: dict[str, float] = {"absent": 0.0, "partial": 0.5, "present": 1.0}


def normalise_verdict(verdict: Any) -> str:
    """Lowercase/strip a verdict string and validate it against `VERDICTS`."""
    v = str(verdict).strip().lower()
    if v not in VERDICT_SCORE:
        raise ValueError(f"unknown verdict {verdict!r}; expected one of {VERDICTS}")
    return v


def verdict_score(verdict: Any) -> float:
    """present=1.0, partial=0.5, absent=0.0 (raises ValueError on anything else)."""
    return VERDICT_SCORE[normalise_verdict(verdict)]


def _rows(judgements: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> list[tuple[str, float]]:
    """Normalise judgement input to [(slot_id, score), ...]; raises on malformed rows."""
    out: list[tuple[str, float]] = []
    if isinstance(judgements, Mapping):
        for slot_id, verdict in judgements.items():
            out.append((str(slot_id), verdict_score(verdict)))
        return out
    for i, row in enumerate(judgements):
        if not isinstance(row, Mapping):
            raise TypeError(f"judgement {i} is {type(row).__name__}, expected a mapping")
        if "slot_id" in row:
            slot_id = row["slot_id"]
        elif "slot" in row:
            slot_id = row["slot"]
        else:
            raise KeyError(f"judgement {i} has no 'slot_id'")
        if "verdict" not in row:
            raise KeyError(f"judgement {i} (slot {slot_id!r}) has no 'verdict'")
        out.append((str(slot_id), verdict_score(row["verdict"])))
    return out


# ---------------------------------------------------------------------------
# Operational completeness
# ---------------------------------------------------------------------------
def operational_completeness(
    judgements: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    weights: Mapping[str, float] | None = None,
    *,
    default_weight: float | None = None,
) -> float:
    """Mean slot score (present=1.0, partial=0.5, absent=0.0), optionally prevalence-weighted.

    Args:
        judgements: `{slot_id: verdict}` or rows with `slot_id`/`verdict` keys. Rows that did
            not produce a usable verdict must be filtered out by the caller (see
            `judge_slots.usable_judgements`) — an unknown verdict raises, never scores 0.
        weights: optional per-slot weights (e.g. slot prevalence in the human corpus). The
            result is `sum(w_i * s_i) / sum(w_i)` over the judgements given.
        default_weight: weight for a slot missing from `weights`. `None` (default) raises
            instead, so a stale weight table can never silently drop slots.

    Returns:
        The completeness score in [0, 1].

    Raises:
        ValueError: no judgements, unknown verdict, negative weight, or zero total weight.
        KeyError: a slot is missing from `weights` and `default_weight` is None.
    """
    rows = _rows(judgements)
    if not rows:
        raise ValueError("operational_completeness needs at least one judgement")
    if weights is None:
        return fmean(score for _, score in rows)
    num = den = 0.0
    for slot_id, score in rows:
        if slot_id in weights:
            w = float(weights[slot_id])
        elif default_weight is not None:
            w = float(default_weight)
        else:
            raise KeyError(f"no weight for slot {slot_id!r} (pass default_weight to allow this)")
        if math.isnan(w) or w < 0:
            raise ValueError(f"weight for slot {slot_id!r} must be >= 0, got {w!r}")
        num += w * score
        den += w
    if den <= 0:
        raise ValueError("total weight is zero; cannot compute a weighted mean")
    return num / den


@dataclass(frozen=True)
class SlotCoverage:
    """How well one slot is covered across a set of sitreps."""

    slot_id: str
    n: int
    coverage: float
    n_present: int
    n_partial: int
    n_absent: int

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def coverage_by_slot(judgements: Iterable[Mapping[str, Any]]) -> dict[str, SlotCoverage]:
    """Per-slot mean score over many sitreps -> `{slot_id: SlotCoverage}` (first-seen order)."""
    scores: dict[str, list[float]] = {}
    for slot_id, score in _rows(judgements):
        scores.setdefault(slot_id, []).append(score)
    return {
        slot_id: SlotCoverage(
            slot_id=slot_id,
            n=len(vals),
            coverage=fmean(vals),
            n_present=sum(1 for v in vals if v == 1.0),
            n_partial=sum(1 for v in vals if v == 0.5),
            n_absent=sum(1 for v in vals if v == 0.0),
        )
        for slot_id, vals in scores.items()
    }


# ---------------------------------------------------------------------------
# Visibility ceiling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlotCeiling:
    """Per-slot detail behind the visibility-ceiling claim (one row per slot, CSV-ready)."""

    slot_id: str
    is_ceiling: bool
    human_coverage: float | None
    machine_coverage_max: float | None
    machine_best_arm: str | None
    machine_coverage_by_arm: dict[str, float] = field(default_factory=dict)
    gap: float | None = None
    reason: str = ""

    def as_row(self) -> dict[str, Any]:
        """Flat dict for CSV (arm detail collapsed to an `arm=value;...` string)."""
        row = asdict(self)
        row["machine_coverage_by_arm"] = ";".join(
            f"{arm}={val:.4f}" for arm, val in sorted(self.machine_coverage_by_arm.items())
        )
        return row


def _arm_map(value: Any) -> dict[str, float]:
    """Coerce a per-slot machine score (scalar or {arm: score}) to {arm: score}."""
    if isinstance(value, Mapping):
        return {str(k): float(v) for k, v in value.items()}
    if value is None:
        return {}
    return {"all": float(value)}


def visibility_ceiling(
    machine_scores: Mapping[str, float | Mapping[str, float]],
    human_scores: Mapping[str, float],
    *,
    machine_max: float = 0.1,
    human_min: float = 0.5,
) -> list[SlotCeiling]:
    """Slots that human sitreps fill but **no** machine model/arm does — the visibility ceiling.

    A slot is a ceiling slot when `max(machine coverage over all arms) <= machine_max`
    (both bounds inclusive) **and** `human coverage >= human_min`. A slot that is thin in the
    human sitreps too is *not* a ceiling slot: the ceiling claim is about what social media
    cannot show, not about what nobody bothers to write down.

    Args:
        machine_scores: `{slot_id: coverage}` or `{slot_id: {model_or_arm: coverage}}`.
        human_scores: `{slot_id: coverage}` for held-out human sitreps.
        machine_max: machine coverage at or below this counts as "≈ 0".
        human_min: human coverage at or above this counts as "high".

    Returns:
        One `SlotCeiling` per slot in the union of both inputs (per-slot detail, not just the
        ceiling ids), ceiling slots first, then widest gap first, then slot id.
    """
    if not 0.0 <= machine_max <= 1.0 or not 0.0 <= human_min <= 1.0:
        raise ValueError("machine_max and human_min must be in [0, 1]")
    out: list[SlotCeiling] = []
    for slot_id in sorted(set(machine_scores) | set(human_scores)):
        arms = _arm_map(machine_scores.get(slot_id))
        human = human_scores.get(slot_id)
        human_cov = None if human is None else float(human)
        best_arm = max(arms, key=lambda a: (arms[a], a)) if arms else None
        m_max = arms[best_arm] if best_arm is not None else None
        gap = None if (human_cov is None or m_max is None) else human_cov - m_max
        if not arms:
            is_ceiling, reason = False, "no machine judgements for this slot"
        elif human_cov is None:
            is_ceiling, reason = False, "no human judgements for this slot"
        elif m_max > machine_max:
            is_ceiling = False
            reason = f"machine coverage {m_max:.3f} > machine_max {machine_max:.3f} (arm {best_arm})"
        elif human_cov < human_min:
            is_ceiling = False
            reason = f"human coverage {human_cov:.3f} < human_min {human_min:.3f} (thin in humans too)"
        else:
            is_ceiling = True
            reason = f"machine <= {machine_max:.3f} on all {len(arms)} arm(s), human >= {human_min:.3f}"
        out.append(SlotCeiling(slot_id=slot_id, is_ceiling=is_ceiling, human_coverage=human_cov,
                               machine_coverage_max=m_max, machine_best_arm=best_arm,
                               machine_coverage_by_arm=arms, gap=gap, reason=reason))
    out.sort(key=lambda s: (not s.is_ceiling, -(s.gap if s.gap is not None else -math.inf), s.slot_id))
    return out


# ---------------------------------------------------------------------------
# Figure extraction spot check
# ---------------------------------------------------------------------------
_SCALES: dict[str, float] = {
    "thousand": 1e3, "thousands": 1e3, "million": 1e6, "millions": 1e6,
    "billion": 1e9, "billions": 1e9, "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7,
}
_CASUALTY_WORDS: tuple[str, ...] = (
    "killed", "dead", "death", "deaths", "died", "fatalities", "fatality", "casualties", "casualty",
    "injured", "injuries", "wounded", "missing", "unaccounted",
)
_QUANTITY_WORDS: tuple[str, ...] = (
    "affected", "displaced", "evacuated", "homeless", "destroyed", "damaged", "people", "persons",
    "households", "families", "children", "beneficiaries", "in need", "food insecure", "sheltered",
    "reached", "cases", "admitted", "hectares", "shelters",
)
_MONTHS: tuple[str, ...] = (
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec",
)
_SCALE_ALT = "|".join(sorted(_SCALES, key=len, reverse=True))
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_NUM_RE = re.compile(
    r"(?<![\d.,])(?P<num>\d{1,3}(?:[,\u00A0 ]\d{3})+|\d+(?:\.\d+)?)(?!\d)"
    rf"(?:[\s\u00A0]*(?P<scale>{_SCALE_ALT})\b)?"
    r"(?P<pct>[\s\u00A0]*(?:%|per[\s\u00A0]?cent))?",
    re.IGNORECASE,
)
# Dates are not figures: 2019-03-22, 22/03/2019, "22 March 2019", "March 22, 2019", "March 2019".
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_ALT})\.?(?:,?\s+\d{{4}})?"
    rf"|\b(?:{_MONTH_ALT})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|\b(?:{_MONTH_ALT})\.?\s+\d{{4}}\b",
    re.IGNORECASE,
)
_CLAUSE_BREAKS = ".;\n"
_CONTEXT_CHARS = 80


@dataclass(frozen=True)
class Figure:
    """One number found in a sitrep. Carries no prose - only the numeric token and a lexicon tag.

    Keeping surrounding text out of `Figure` is deliberate: figure tables are written under
    `results/` (committed), and machine sitreps are derived from social-media posts, which must
    never be reproduced verbatim (CLAUDE.md rule 7).
    """

    value: float
    raw: str             # the numeric expression only, e.g. "1,200" or "1.4 million"
    unit: str            # 'count' | 'percent'
    kind: str            # 'casualty' | 'quantity' (coarse lexicon tag, not a parse)
    keyword: str | None  # the lexicon term that classified it (never corpus prose)
    start: int = -1      # character offsets support private manual audit; no prose is persisted
    end: int = -1

    @property
    def key(self) -> tuple[str, float]:
        """Identity used for machine/human overlap: unit + rounded value."""
        return (self.unit, round(self.value, 6))


def _date_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of date expressions, whose digits must not be read as figures."""
    return [(m.start(), m.end()) for m in _DATE_RE.finditer(text)]


def _clause_around(low: str, start: int, end: int) -> str:
    """Lowercased clause containing [start, end): bounded by '.', ';', newline or +-80 chars.

    Clause bounding stops keywords from a neighbouring sentence from mislabelling a number.
    """
    left = max(0, start - _CONTEXT_CHARS)
    right = min(len(low), end + _CONTEXT_CHARS)
    for i in range(start - 1, left - 1, -1):
        if low[i] in _CLAUSE_BREAKS:
            left = i + 1
            break
    for i in range(end, right):
        if low[i] in _CLAUSE_BREAKS:
            right = i
            break
    return low[left:right]


def extract_figures(text: str) -> list[Figure]:
    """Extract casualty/quantity figures from sitrep text (heuristic, regex-based).

    A number is kept only when it is quantitative enough to matter: it carries a scale word
    ("1.4 million"), a percent marker, a thousands separator, or a casualty/quantity keyword in
    its clause. Dates ("22 March 2019", "2019-03-22", "22/03/2019") and bare years (1900-2100)
    are dropped. `kind` is a coarse lexicon tag: a casualty word anywhere in the clause makes a
    figure a casualty figure, so mixed clauses ("1,200 killed and 45% of homes destroyed") can
    tag a quantity as a casualty. Treat it as a spot check, not a parse (PLAN cut-list item 3).
    """
    figures: list[Figure] = []
    if not text:
        return figures
    low = text.lower()
    dates = _date_spans(text)
    for m in _NUM_RE.finditer(text):
        n_start, n_end = m.start("num"), m.end("num")
        if any(d0 <= n_start and n_end <= d1 for d0, d1 in dates):
            continue
        token = m.group("num")
        scale = (m.group("scale") or "").strip().lower()
        unit = "percent" if m.group("pct") else "count"
        if not scale and unit == "count" and token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2100:
            continue  # a bare four-digit year, not a quantity
        value = float(token.replace(",", "").replace("\u00A0", "").replace(" ", ""))
        if scale:
            value *= _SCALES[scale]
        clause = _clause_around(low, m.start(), m.end())
        keyword = next((w for w in _CASUALTY_WORDS if w in clause), None)
        kind = "casualty" if keyword else "quantity"
        if keyword is None:
            keyword = next((w for w in _QUANTITY_WORDS if w in clause), None)
        has_separator = any(sep in token for sep in (",", " ", "\u00A0"))
        if not (scale or unit == "percent" or has_separator or keyword):
            continue
        figures.append(Figure(value=value, raw=m.group(0).strip(), unit=unit, kind=kind, keyword=keyword,
                              start=m.start(), end=m.end()))
    return figures


@dataclass(frozen=True)
class FigureCheck:
    """Result of `figure_extraction_check` — figures on each side and their overlap."""

    machine: list[Figure]
    human: list[Figure]
    overlap: list[Figure]        # human-side figures whose (unit, value) also occur in machine
    only_human: list[Figure]
    only_machine: list[Figure]
    recall: float | None         # |overlap| / |unique human figures|
    precision: float | None      # |overlap| / |unique machine figures|
    jaccard: float | None
    casualty_recall: float | None

    def as_row(self) -> dict[str, Any]:
        """Counts and rates only — safe to write to `results/tables/`."""
        return {
            "n_machine_figures": len({f.key for f in self.machine}),
            "n_human_figures": len({f.key for f in self.human}),
            "n_overlap": len(self.overlap),
            "n_only_human": len(self.only_human),
            "n_only_machine": len(self.only_machine),
            "n_human_casualty_figures": len({f.key for f in self.human if f.kind == "casualty"}),
            "n_machine_casualty_figures": len({f.key for f in self.machine if f.kind == "casualty"}),
            "figure_recall": self.recall,
            "figure_precision": self.precision,
            "figure_jaccard": self.jaccard,
            "casualty_figure_recall": self.casualty_recall,
        }


def _unique(figs: Iterable[Figure]) -> list[Figure]:
    seen: set[tuple[str, float]] = set()
    out: list[Figure] = []
    for f in figs:
        if f.key not in seen:
            seen.add(f.key)
            out.append(f)
    return out


def figure_extraction_check(machine_text: str, human_text: str) -> FigureCheck:
    """Which casualty/quantity numbers a machine sitrep carries vs the human one, and the overlap.

    Rates are `None` when the denominator is empty (no human / no machine figures), never 0.0,
    so an empty sitrep is not silently reported as perfect precision.
    """
    machine = extract_figures(machine_text)
    human = extract_figures(human_text)
    m_u, h_u = _unique(machine), _unique(human)
    m_keys = {f.key for f in m_u}
    h_keys = {f.key for f in h_u}
    overlap = [f for f in h_u if f.key in m_keys]
    only_human = [f for f in h_u if f.key not in m_keys]
    only_machine = [f for f in m_u if f.key not in h_keys]
    union = len(m_keys | h_keys)
    h_cas = [f for f in h_u if f.kind == "casualty"]
    return FigureCheck(
        machine=machine, human=human, overlap=overlap, only_human=only_human, only_machine=only_machine,
        recall=(len(overlap) / len(h_u)) if h_u else None,
        precision=(len(overlap) / len(m_u)) if m_u else None,
        jaccard=(len(m_keys & h_keys) / union) if union else None,
        casualty_recall=(sum(1 for f in h_cas if f.key in m_keys) / len(h_cas)) if h_cas else None,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapCI:
    """Percentile bootstrap CI for a statistic of a sample."""

    point: float
    lo: float
    hi: float
    n: int
    n_boot: int
    alpha: float
    seed: int

    def as_row(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile of an already-sorted sample (q in [0, 1])."""
    if not sorted_vals:
        raise ValueError("empty sample")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[int(pos)])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 17,
    statistic: Callable[[Sequence[float]], float] | None = None,
) -> BootstrapCI:
    """Seeded percentile bootstrap CI for the mean (or any `statistic`) of `values`.

    Deterministic: the same `values`, `n_boot` and `seed` always give the same interval
    (`random.Random(seed)`, stdlib only — no numpy RNG version drift across machines).
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("bootstrap_ci needs at least one value")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    stat = statistic or fmean
    point = float(stat(vals))
    rng = random.Random(seed)
    n = len(vals)
    stats = sorted(float(stat([vals[rng.randrange(n)] for _ in range(n)])) for _ in range(n_boot))
    return BootstrapCI(point=point, lo=_percentile(stats, alpha / 2), hi=_percentile(stats, 1 - alpha / 2),
                       n=n, n_boot=n_boot, alpha=alpha, seed=seed)


# ---------------------------------------------------------------------------
# Judge validation: agreement and Cohen's kappa
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Agreement:
    """Judge-vs-human agreement over a validation sample."""

    n: int
    agreement: float
    kappa: float
    kappa_linear: float
    labels: list[str]
    confusion: list[list[int]]   # confusion[i][j] = judge said labels[i], human said labels[j]
    per_label_agreement: dict[str, float]

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"n": self.n, "raw_agreement": self.agreement, "cohens_kappa": self.kappa,
                               "cohens_kappa_linear": self.kappa_linear}
        for i, li in enumerate(self.labels):
            row[f"agree_{li}"] = self.per_label_agreement[li]
            for j, lj in enumerate(self.labels):
                row[f"n_judge_{li}_human_{lj}"] = self.confusion[i][j]
        return row


def _confusion(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        mat[idx[x]][idx[y]] += 1
    return mat


def cohens_kappa(
    a: Sequence[str],
    b: Sequence[str],
    *,
    labels: Sequence[str] = VERDICTS,
    weighting: str | None = None,
) -> float:
    """Cohen's kappa between two label sequences (`weighting`: None | 'linear' | 'quadratic').

    Returns `nan` when expected disagreement is 0 (both raters used one identical label), where
    kappa is undefined — the same convention scikit-learn uses.
    """
    if len(a) != len(b):
        raise ValueError(f"label sequences differ in length: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("cohens_kappa needs at least one pair")
    if weighting not in (None, "linear", "quadratic"):
        raise ValueError(f"weighting must be None, 'linear' or 'quadratic', got {weighting!r}")
    labels = list(labels)
    unknown = (set(a) | set(b)) - set(labels)
    if unknown:
        raise ValueError(f"labels outside {labels}: {sorted(unknown)}")
    n = len(a)
    k = len(labels)
    mat = _confusion(a, b, labels)
    row = [sum(mat[i]) for i in range(k)]
    col = [sum(mat[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        if weighting is None:
            return 0.0 if i == j else 1.0
        d = abs(i - j) / (k - 1) if k > 1 else 0.0
        return d if weighting == "linear" else d * d

    obs = sum(w(i, j) * mat[i][j] for i in range(k) for j in range(k)) / n
    exp = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    if exp == 0:
        return float("nan")
    return 1.0 - obs / exp


def judge_agreement(
    judge: Sequence[str],
    human: Sequence[str],
    *,
    labels: Sequence[str] = VERDICTS,
) -> Agreement:
    """Raw agreement + Cohen's kappa (unweighted and linear-weighted) for the judge validation.

    PLAN Phase 4 acceptance is `agreement >= 0.7`; kappa is reported alongside it because raw
    agreement is inflated whenever one verdict dominates the sample.
    """
    judge = [normalise_verdict(v) for v in judge]
    human = [normalise_verdict(v) for v in human]
    if len(judge) != len(human):
        raise ValueError(f"judge/human sequences differ in length: {len(judge)} vs {len(human)}")
    if not judge:
        raise ValueError("judge_agreement needs at least one pair")
    labels = list(labels)
    k = len(labels)
    mat = _confusion(judge, human, labels)
    per_label: dict[str, float] = {}
    for i, lab in enumerate(labels):
        # Jaccard-style: hits / (times either rater used the label).
        used = sum(mat[i][j] for j in range(k)) + sum(mat[j][i] for j in range(k)) - mat[i][i]
        per_label[lab] = (mat[i][i] / used) if used else float("nan")
    return Agreement(
        n=len(judge),
        agreement=sum(mat[i][i] for i in range(k)) / len(judge),
        kappa=cohens_kappa(judge, human, labels=labels),
        kappa_linear=cohens_kappa(judge, human, labels=labels, weighting="linear"),
        labels=labels,
        confusion=mat,
        per_label_agreement=per_label,
    )

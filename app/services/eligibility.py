"""Ad eligibility scoring and safety gating.

Responsibility:
    Convert a raw query and a unit-normalized query embedding into an ad
    eligibility score in [0.0, 1.0], with hard safety exits for blocklisted
    or crisis-like queries.

Inputs:
    - query: raw text used by regex/fuzzy/ML blocklist strategies
    - query_embedding: unit-normalized embedding vector used for semantic layers

Outputs:
    - score(): float in [0.0, 1.0]
    - score_with_metadata(): dict with breakdown fields used by API metadata

Layer summary:
    1) Blocklist layer (regex/fuzzy/ML): may hard block to 0.0 before semantic work.
    2) Semantic sensitivity layer — two signals consulted in parallel (OR logic):
       2a) v5 insensitive-query model: hard-blocks above model threshold; provides
           continuous insensitivity probability below threshold.
       2b) Cluster centroid matching (k=3 sub-centroids per cluster): per-cluster hard
           gates fire at lower thresholds than the model catches alone:
             self_harm_crisis=0.60, abuse_trauma=0.62, medical_emergency=0.60,
             addiction_recovery=0.65, grief_loss=0.90, financial_hardship=0.95,
             weapons_violence=0.62, immediate_danger=0.55.
           Continuous penalty = max(model_prob, cluster_penalty) — conservative.
    3) Commercial affinity layer: applies a bounded boost when sensitivity is not extreme.

Invariants:
    - Eligibility, penalties, and boosts are clamped into interpretable ranges.
    - score() and score_with_metadata() always return values safe for downstream
      API consumers that assume probability-like bounds.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.services.embedding import EmbeddingService

# ---------------------------------------------------------------------------
# Data directory and config loading helpers
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SCORING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "eligibility_config.yaml"


def _load_json_data(filename: str) -> Any:
    """Load a JSON data file from data/. Returns None if the file is missing."""
    path = _DATA_DIR / filename
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _load_scoring_config() -> dict:
    """Load the scoring section from eligibility_config.yaml."""
    try:
        import yaml
        with open(_SCORING_CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        return data.get("scoring", {})
    except Exception:
        return {}


_scoring_cfg: dict = _load_scoring_config()

# ---------------------------------------------------------------------------
# Tunable constants — loaded from eligibility_config.yaml (scoring section)
# ---------------------------------------------------------------------------

# How aggressively to penalise sensitivity signal (Layer 2 base formula).
# cosine_sim * SENSITIVITY_SCALE, then clamped to [0, 1].
SENSITIVITY_SCALE: float = float(_scoring_cfg.get("sensitivity_scale", 0.7))

# High noise floor to keep eligibility high for ambiguous/idiomatic phrasing.
SENSITIVITY_MIN_SIM: float = float(_scoring_cfg.get("sensitivity_min_sim", 0.41))

# Commercial boost range: score ∈ [COMMERCIAL_MIN, COMMERCIAL_MIN + COMMERCIAL_RANGE]
# Non-commercial queries get COMMERCIAL_MIN; strongly transactional queries get the full sum.
COMMERCIAL_MIN: float = float(_scoring_cfg.get("commercial_min", 0.87))
COMMERCIAL_RANGE: float = float(_scoring_cfg.get("commercial_range", 0.13))
COMMERCIAL_SCALE: float = float(_scoring_cfg.get("commercial_scale", 2.5))

# ---------------------------------------------------------------------------
# Layer 2a — Per-cluster semantic hard gates
# ---------------------------------------------------------------------------
# Queries whose cosine similarity to a cluster centroid exceeds this threshold
# are immediately scored 0.0 (regardless of blocklist or commercial signal).
_CLUSTER_HARD_THRESHOLDS: dict[str, float] = _scoring_cfg.get(
    "cluster_hard_thresholds",
    {
        "self_harm_crisis":    0.60,
        "abuse_trauma":        0.62,
        "medical_emergency":   0.60,
        "addiction_recovery":  0.65,
        "grief_loss":          0.90,
        "financial_hardship":  0.95,
        "weapons_violence":    0.62,
        "immediate_danger":    0.55,
    },
)

# ---------------------------------------------------------------------------
# Layer 2a — Per-cluster sensitivity scale
# ---------------------------------------------------------------------------
# Multiplier applied to cosine similarity when computing the continuous penalty
# for each cluster. Clusters not listed fall back to the global SENSITIVITY_SCALE.
_CLUSTER_SENSITIVITY_SCALE: dict[str, float] = _scoring_cfg.get(
    "cluster_sensitivity_scale",
    {
        "self_harm_crisis":    1.5,
        "medical_emergency":   1.5,
        "weapons_violence":    1.5,
        "immediate_danger":    1.5,
        "abuse_trauma":        1.4,
        "addiction_recovery":  1.4,
        "grief_loss":          1.3,
        "financial_hardship":  1.1,
    },
)

# ---------------------------------------------------------------------------
# Layer 2b — Co-activation bonus
# ---------------------------------------------------------------------------
CO_ACTIVATION_FLOOR: float = float(_scoring_cfg.get("co_activation_floor", 0.40))
CO_ACTIVATION_BONUS_PER: float = float(_scoring_cfg.get("co_activation_bonus_per", 0.0))

# ---------------------------------------------------------------------------
# Layer 2c — Linguistic distress amplifier
# ---------------------------------------------------------------------------
DISTRESS_AMPLIFIER_SCALE: float = float(_scoring_cfg.get("distress_amplifier_scale", 0.0))
AMPLIFIER_MIN_PENALTY: float = float(_scoring_cfg.get("amplifier_min_penalty", 0.80))
AMPLIFIER_MIN_MARKER_TYPES: int = int(_scoring_cfg.get("amplifier_min_marker_types", 2))

# Toxicity model used to suppress insensitive ad serving.
_DEFAULT_INSENSITIVE_MODEL_PATH = _DATA_DIR / "insensitive_model_v5.joblib"

# ---------------------------------------------------------------------------
# Distress marker regex patterns (compiled once at import time, ~0 overhead)
# ---------------------------------------------------------------------------
# Each pattern targets a distinct LINGUISTIC DIMENSION of distress expression.
# They are intentionally broad within their dimension and rely on the ≥ 2 type
# requirement to avoid false positives.

# Dimension 1: First-person present crisis framing.
_DISTRESS_FIRST_PERSON: re.Pattern[str] = re.compile(
    r"\b("
    r"i am|i'm|i feel|i can't|i cannot|i'm feeling|"
    r"my (?:mom|dad|father|mother|son|daughter|wife|husband|partner|child|children|baby|"
    r"sister|brother|friend|dog|cat|pet) (?:just|has|have|died|passed|left|is gone|was killed)|"
    r"i've been|i'm going through|i was (?:attacked|raped|abused|assaulted|beaten)"
    r")\b",
    re.IGNORECASE,
)

# Dimension 2: Urgency / temporal immediacy.
_DISTRESS_URGENCY: re.Pattern[str] = re.compile(
    r"\b("
    r"right now|tonight|just (?:happened|found out|lost|died)|"
    r"emergency|urgent|need help now|please help|immediately|"
    r"this (?:moment|instant|second|night)|can't wait|call 911|going to (?:hurt|harm|kill)"
    r")\b",
    re.IGNORECASE,
)

# Dimension 3: Depletion / hopelessness language.
_DISTRESS_DEPLETION: re.Pattern[str] = re.compile(
    r"\b("
    r"nothing left|can't cope|can't go on|no hope|no point|"
    r"give up|gave up|lost everything|losing everything|"
    r"can't take it|can't handle|end it|end it all|"
    r"fall(?:ing)? apart|break(?:ing)? down|at the end|hit rock bottom|"
    r"no (?:reason|will|strength|energy) to (?:go on|continue|live|try)"
    r")\b",
    re.IGNORECASE,
)

# Dimension 4: High-valence negative affect vocabulary.
_DISTRESS_NEGATIVE_AFFECT: re.Pattern[str] = re.compile(
    r"\b("
    r"devastated|heartbroken|suicidal|hopeless|worthless|"
    r"destroyed|shattered|broken inside|terrified|helpless|"
    r"in despair|despairing|traumatized|traumatised|overwhelmed|"
    r"inconsolable|unbearable|can't breathe|can't function"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Distress signal computation (Layer 2c)
# ---------------------------------------------------------------------------


def _compute_distress_signal(query: str) -> float:
    """Compute a linguistic distress signal in [0.0, 1.0].

    Checks four orthogonal dimensions of distress language:
        1. First-person present crisis framing
        2. Urgency / temporal immediacy
        3. Depletion / hopelessness vocabulary
        4. High-valence negative affect

    Returns 0.0 if fewer than AMPLIFIER_MIN_MARKER_TYPES dimensions match
    (prevents single-word false positives). Otherwise returns a signal in (0, 1]
    that scales with how many dimensions are present.

    This is intentionally lightweight: four regex checks, ~0.05ms per call.
    """
    types_matched = 0
    if _DISTRESS_FIRST_PERSON.search(query):
        types_matched += 1
    if _DISTRESS_URGENCY.search(query):
        types_matched += 1
    if _DISTRESS_DEPLETION.search(query):
        types_matched += 1
    if _DISTRESS_NEGATIVE_AFFECT.search(query):
        types_matched += 1

    if types_matched < AMPLIFIER_MIN_MARKER_TYPES:
        return 0.0

    total_dims = 4
    return min(
        (types_matched - AMPLIFIER_MIN_MARKER_TYPES + 1) / (total_dims - AMPLIFIER_MIN_MARKER_TYPES + 1),
        1.0,
    )


# ---------------------------------------------------------------------------
# Sensitivity cluster definitions — loaded from data/sensitivity_clusters.json
# ---------------------------------------------------------------------------

# Each cluster is a list of representative sentences. At init these are averaged
# (after embedding) into a single unit-normalised centroid vector.
_SENSITIVITY_CLUSTERS: dict[str, list[str]] = _load_json_data("sensitivity_clusters.json") or {}

# Representative phrases for commercial / transactional intent.
_COMMERCIAL_EXEMPLARS: list[str] = _load_json_data("commercial_exemplars.json") or []

# ---------------------------------------------------------------------------
# Blocklist patterns — loaded from data/blocklist_builtin.txt
# ---------------------------------------------------------------------------


def _load_builtin_patterns() -> list[str]:
    """Load built-in blocklist patterns from data/blocklist_builtin.txt.

    Falls back to a minimal hardcoded set if the file is missing, so safety
    is never completely absent even in stripped environments.
    """
    path = _DATA_DIR / "blocklist_builtin.txt"
    if path.exists():
        patterns: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
        if patterns:
            return patterns
    # Minimal fallback (file missing or empty)
    return [
        r"\b(self[- ]?harm|self[- ]?hurt|hurt myself|kill myself|end my life|suicide|suicidal|want to die)\b",
        r"\b(pipe bomb|make a bomb|how to bomb|how to shoot|mass shooting|school shooting|build a gun illegally)\b",
        r"\bhow to (?:make|build|create|synthesize) (?:a |an )?(?:weapon|bomb|explosive|poison|meth|fentanyl|drug)\b",
        r"\b(child porn(?:ography)?|csam|underage (?:sex|nude|explicit))\b",
        r"\b(ethnic cleansing|genocide|racial extermination)\b",
        r"\b(terrorist?|terrorism) (?:attack|bomb|plot|plan|instructions)\b",
        r"\bhow to (?:murder|kill|assassinate|stab|poison) (?:someone|a person|my)\b",
    ]


_BLOCKLIST_PATTERNS: list[str] = _load_builtin_patterns()


def _load_blocklist(path: str | Path | None) -> list[re.Pattern[str]]:
    """Load blocklist regex patterns, always including built-ins.

    File format: one regex per line. Empty lines and lines starting with # are skipped.
    Precedence rule: built-ins are always included first as a safety floor; file
    patterns are additive overrides for deploy-time policy updates.
    If path is None or the file does not exist, returns only compiled built-ins.
    """
    compiled_builtins = [re.compile(pat, re.IGNORECASE) for pat in _BLOCKLIST_PATTERNS]
    if path is not None:
        p = Path(path)
        if p.exists():
            file_patterns: list[str] = []
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        file_patterns.append(line)
            if file_patterns:
                return compiled_builtins + [re.compile(pat, re.IGNORECASE) for pat in file_patterns]
    return compiled_builtins


@dataclass(frozen=True)
class BlocklistMatch:
    """Represents a blocklist match and why it matched.

    score: 0.0 = hard block (do not show ads); 0 < score < 1.0 = soft cap
    (eligibility is capped at this value but scoring continues).
    """

    strategy: str
    rule: str | None = None
    score: float = 0.0


@dataclass(frozen=True)
class PurchaseIntentionResult:
    """Represents purchase intention check result."""

    strategy: str
    purchase_probability: float
    should_gate: bool  # True if purchase intention too low to show ads
    rule: str | None = None


class InsensitiveQueryModel:
    """Inference wrapper for generate-ads-or-not toxicity v5 artifact.

    The model predicts the probability that showing ads would be insensitive.
    """

    def __init__(self, model_path: str | Path) -> None:
        import joblib

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Trying to unpickle estimator .*",
            )
            artifact = joblib.load(model_path)
        estimator = artifact.get("estimator")
        if estimator is None:
            raise ValueError("Invalid v5 model artifact: missing 'estimator'")

        self._estimator = estimator
        self._calibrator = artifact.get("calibrator")
        self._rules: tuple[str, ...] = tuple(
            str(rule).strip().lower() for rule in (artifact.get("rules") or []) if str(rule).strip()
        )
        self._threshold = float(artifact.get("threshold", 0.5))

    @property
    def threshold(self) -> float:
        return self._threshold

    def predict_insensitive_probability(self, query: str) -> float:
        text = str(query or "")
        low = text.lower()

        # Deterministic rule shortcuts from training bundle.
        if self._rules and any(rule in low for rule in self._rules):
            return 1.0

        if hasattr(self._estimator, "predict_proba"):
            prob = float(self._estimator.predict_proba([text])[0][1])
        elif hasattr(self._estimator, "decision_function"):
            raw = float(self._estimator.decision_function([text])[0])
            prob = 1.0 / (1.0 + math.exp(-raw))
        else:
            prob = float(self._estimator.predict([text])[0])

        if self._calibrator is not None:
            prob = float(self._calibrator.predict([prob])[0])

        return float(np.clip(prob, 0.0, 1.0))


class BlocklistStrategy(Protocol):
    """Strategy interface for blocklist matching."""

    def match(self, query: str) -> BlocklistMatch | None: ...


class RegexBlocklistStrategy:
    """Regex-based blocklist strategy (current default behavior)."""

    def __init__(self, patterns: list[re.Pattern[str]]) -> None:
        self._patterns = patterns

    def match(self, query: str) -> BlocklistMatch | None:
        for pattern in self._patterns:
            if pattern.search(query):
                return BlocklistMatch(strategy="regex", rule=pattern.pattern)
        return None


# ---------------------------------------------------------------------------
# Fuzzy Blocklist Strategy — Deterministic (v5)
# ---------------------------------------------------------------------------
# Key improvements over regex strategy:
#   - Leetspeak normalisation (0→o, 1→i, 3→e, 4→a, 5→s, 7→t)
#   - Collapsed-spaced-letter detection (catches "s u i c i d e" obfuscations)
#   - Compact-form matching (strips non-alphanum for run-together evasions)
#   - Violence / bomb idiom guards (won't block "killing it" or "bomb tacos")
#   - PII and doxxing detection
#   - Spam / scam heuristics

_FZ_NONWORD_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_FZ_WS_RE = re.compile(r"\s+")
_LEETSPEAK_TABLE = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})

_FZ_FOOD_NOUNS: frozenset[str] = frozenset({
    "food", "tacos", "ramen", "pizza", "biryani", "burger",
    "coffee", "restaurant", "meal", "dinner", "lunch", "breakfast",
})
_FZ_BOMB_SLANG: frozenset[str] = frozenset({
    "bomb ass", "the bomb", "bomb af", "bomb asf",
    "thats bomb", "that's bomb", "bomb.com",
})
_FZ_HATE_SLURS: frozenset[str] = frozenset({
    "nigger", "nigga", "faggot", "kike", "spic", "chink", "wetback", "raghead", "pajeet",
})
_FZ_VIOLENCE_IDIOMS: frozenset[str] = frozenset({
    "kill it", "killing it", "killed it", "dead tired",
    "im dead", "i'm dead", "slay it", "slayed it", "deadass",
})
_FZ_PHOTO_IDIOMS: frozenset[str] = frozenset({
    "photo shoot", "shoot photos", "shooting photos", "shoot a photo",
})
_FZ_DRUG_TERMS: frozenset[str] = frozenset({
    "drugs", "cocaine", "weed", "heroin", "meth", "fentanyl", "opioids", "xanax",
})
_FZ_EXPLICIT_BODY: frozenset[str] = frozenset({"tits", "boobs", "pussy", "dick", "cock", "cum"})
_FZ_MINOR_TERMS: frozenset[str] = frozenset({"underage", "minor", "child", "kid", "teen"})
_FZ_SEX_TERMS: frozenset[str] = frozenset({
    "sex", "nude", "nudes", "porn", "explicit", "sexual", "fuck", "suck", "blowjob", "handjob",
})
_FZ_EXTREMISM_TERMS: frozenset[str] = frozenset({
    "isis", "isil", "alqaeda", "al qaeda", "taliban", "hezbollah", "extremist", "extremist group",
})
_FZ_PROFANITY: frozenset[str] = frozenset({"fuck", "fucking", "shit", "ass", "arse", "bitch"})
_FZ_SCAM_PHRASES: frozenset[str] = frozenset({
    "free money", "get rich quick", "guaranteed profit", "double your money", "100% free",
})
_FZ_HATE_SLUR_RES: list[re.Pattern[str]] = [
    re.compile(r"n\W*i\W*g\W*g\W*e\W*r", re.IGNORECASE),
    re.compile(r"f\W*a\W*g\W*g\W*o\W*t", re.IGNORECASE),
    re.compile(r"k\W*i\W*k\W*e", re.IGNORECASE),
]
_FZ_SELF_HARM_RE = re.compile(
    r"\b(thoughts of|thinking of|want to|how to|ways to|plan to|teach me|show me|"
    r"i want to|i am|i'm|gonna|going to)\b"
    r".*\b(kill myself|end my life|end myself|suicid(?:e|al)|self[-\s]?harm|hurt myself|cut myself|die|"
    r"don't want to be alive|do not want to be alive)\b",
    re.IGNORECASE,
)
_FZ_SELF_HARM_SUPPORT_RE = re.compile(
    r"\b(988|hotline|helpline|crisis line|prevention|support|resources?|how to help|"
    r"help a friend|counsel(?:or|ing)|therapy)\b",
    re.IGNORECASE,
)
_FZ_DRUG_INTENT_RE = re.compile(
    r"\b(buy|get|score|order|need|want|make|synthesize|manufacture|produce|cook|brew|"
    r"pls i need|please i need|where can i get|how do i buy|how to make|how to produce)\b"
    r".*\b(drugs|cocaine|weed|heroin|meth|fentanyl|opioids|xanax)\b",
    re.IGNORECASE,
)
# First-person active addiction/use crisis — distinct from procurement intent above.
# Catches: "I've been using heroin and I can't stop", "I'm addicted to meth",
# "I can't stop taking fentanyl", "hooked on heroin", "dependent on opioids".
# NOTE: matched against the *normalized* form (apostrophes stripped → "i've"→"i ve",
# "can't"→"can t", "I'm"→"i m"), so patterns are written accordingly.
# Hard drug list excludes weed (already handled by soft cap in drug terms check).
_FZ_DRUG_PERSONAL_USE_RE = re.compile(
    r"\b("
    r"i ve been (?:using|taking|on)|"           # "I've been" (normalized)
    r"i have been (?:using|taking|on)|"
    r"i m (?:on|using|addicted to|hooked on|dependent on|strung out on)|"  # "I'm" (normalized)
    r"i am (?:on|using|addicted to|hooked on|dependent on|strung out on)|"
    r"i can t stop(?: using| taking)?|"         # "I can't stop" (normalized)
    r"i cannot stop(?: using| taking)?|"
    r"can t stop(?: using| taking)?|"           # "can't stop" (normalized)
    r"cannot stop(?: using| taking)?"
    r")\b"
    r".*\b(heroin|meth(?:amphetamine)?|cocaine|fentanyl|opioids?|crack|smack|dope|xanax|oxy(?:contin)?)\b",
    re.IGNORECASE,
)
_FZ_DRUG_SUPPORT_RE = re.compile(
    r"\b(treatment|rehab|recovery|withdrawal|sobriety|support group|help)\b",
    re.IGNORECASE,
)
_FZ_EXPLOSIVES_RE = re.compile(
    r"\b(how to|teach me|instructions|guide|recipe)\b"
    r".*\b(pipe bomb|molotov|ied|bomb|explosive|detonator)\b",
    re.IGNORECASE,
)
_FZ_MAKE_BOMB_RE = re.compile(
    r"\b(make|build|create|construct|assemble)\s+(a|an)\s+(bomb|explosive|ied|grenade)\b",
    re.IGNORECASE,
)
_FZ_VIOLENCE_THREAT_RE = re.compile(
    r"\b(i will|im going to|i'm going to|gonna)\s+(kill|murder|stab|shoot)\b",
    re.IGNORECASE,
)
_FZ_NSFW_INSTRUCTION_RE = re.compile(
    r"\b(show me how|teach me how|how to)\b.*\b(fuck|suck|blowjob|handjob)\b",
    re.IGNORECASE,
)
_FZ_EXTREMISM_INTENT_RE = re.compile(
    r"\b(join|support|fund|donate|recruit)\b.*\b("
    r"isis|isil|al\s*qaeda|taliban|hezbollah|extremist(?:\s+group)?"
    r")\b",
    re.IGNORECASE,
)
_FZ_VIOLENCE_INTENT_RE = re.compile(
    r"\b(how to|how can i|how do i|ways to|best way to|plan to)\b.*\b("
    r"kill|murder|stab|shoot|poison|kidnap|abduct|torture|hide (?:a\s+)?(?:dead\s+)?body|ambush|attack|stalk|"
    r"blackmail|extort|terror attack|arson|launder money|make counterfeit money"
    r")\b",
    re.IGNORECASE,
)
_FZ_WEAPON_INTENT_RE = re.compile(
    r"\b(how to|how can i|how do i|where can i|ways to)\b.*\b("
    r"untraceable gun|illegal weapon|illegal weapons|ghost gun|silencer|suppressor"
    r")\b",
    re.IGNORECASE,
)
_FZ_CYBERCRIME_INTENT_RE = re.compile(
    r"\b(how to|how can i|how do i|teach me|show me)\b.*\b("
    r"hack|break into|ddos(?: attack)?|phish|steal credit card|stolen credit card"
    r")\b",
    re.IGNORECASE,
)
_FZ_MINOR_EXPLICIT_RE = re.compile(
    r"\b(minor|minors|underage|child|teen)\b.*\b(explicit|sexual|nude|porn|image|images|video|videos)\b",
    re.IGNORECASE,
)
_FZ_LOW_SENSITIVITY_RE = re.compile(
    r"\b("
    r"funeral|bereavement|passed away|died today|child loss|miscarriage|"
    r"i was sexually assaulted|i was raped|my partner assaulted me|my (?:husband|wife) beat me|"
    r"hate speech|hate crime|overdose|stops? breathing|medical emergency|stroke|heart attack|"
    r"emergency domestic violence hotline|"
    r"988(?:\s+(?:crisis|mental health|suicide))?(?:\s+line|\s+number)?|"
    r"suicide hotline|suicide prevention hotline"
    r")\b",
    re.IGNORECASE,
)
_FZ_MEDIUM_SENSITIVITY_RE = re.compile(
    r"\b("
    r"stress(?:ed)?|anxiety|unemployment|lost my job|eviction|bankruptcy|debt|"
    r"legal aid|rent assistance|food assistance|financial hardship|relapse|addiction|"
    r"trauma survivors?|partner yells at me|abusive partner|"
    r"domestic violence shelter|sexual assault survivor|rape crisis|grief counseling|medical bills"
    r")\b",
    re.IGNORECASE,
)
_V5_FALSE_POSITIVE_WORD_RE = re.compile(
    r"\b(?:"
    r"advice|aid|all|and|anxiety|apartment|assistance|by|care|children(?:'s)?|"
    r"debt|depression|domestic|end|help|landlord|lawyer|legal|life|me|no|"
    r"options|rent|resources|stage|support|can|am"
    r")\b",
    re.IGNORECASE,
)
_FZ_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_FZ_PHONE_RE = re.compile(r"\b\+?1?\s*\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", re.IGNORECASE)
_FZ_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_FZ_DOXX_RE = re.compile(
    r"\b(dox|doxx|doxxing|here is (his|her|their) (number|address))\b", re.IGNORECASE
)
_FZ_URL_RE = re.compile(r"\bhttps?://\S+\b", re.IGNORECASE)
_FZ_REPEAT_CHAR_RE = re.compile(r"(.)\1\1\1+")
_REQUEST_INTENT_PREFIX_RE = re.compile(
    r"^(?:\s*(?:"
    r"i\s+(?:want|need|am\s+looking\s+for|m\s+looking\s+for)"
    r"|can\s+you\s+(?:suggest|recommend|find)"
    r"|suggest|recommend|show\s+me|help\s+me\s+find|looking\s+for"
    r")\s+(?:me\s+)?(?:a|an|some|the)?\s*)+",
    re.IGNORECASE,
)
_REQUEST_INTENT_GUARD_RE = re.compile(
    r"\b("
    r"kill|suicid(?:e|al)?|self[-\s]?harm|hurt myself|end my life|"
    r"bomb|explosive|murder|shoot|poison|terror|weapon|"
    r"drugs?|meth|fentanyl|heroin|cocaine|"
    r"hack|ddos|phish|underage|porn|nude|get"
    r")\b",
    re.IGNORECASE,
)



def _fz_normalize(s: str) -> str:
    s = s.strip().lower().translate(_LEETSPEAK_TABLE)
    s = _FZ_NONWORD_RE.sub(" ", s)
    return _FZ_WS_RE.sub(" ", s).strip()


def _strip_request_intent_prefix(query: str) -> str:
    """Remove generic request-intent scaffolding from the query prefix.

    Applied immediately after regex+fuzzy blocklist checks so those safety
    checks see the raw user text first.
    """
    raw = str(query or "")
    if _REQUEST_INTENT_GUARD_RE.search(raw):
        return raw.strip()
    stripped = _REQUEST_INTENT_PREFIX_RE.sub("", raw, count=1).strip()
    return stripped or raw.strip()


def _filter_v5_false_positive_terms(query: str) -> str:
    """Guardrails for false positive triggers."""
    raw = str(query or "")
    filtered = _V5_FALSE_POSITIVE_WORD_RE.sub(" ", raw)
    return _FZ_WS_RE.sub(" ", filtered).strip()


def _fz_collapse_spaced(tokens: list[str]) -> list[str]:
    """Merge runs of 3+ single-letter tokens to catch 's u i c i d e' style evasions."""
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            if len(buf) >= 3:
                out.append("".join(buf))
            else:
                out.extend(buf)
            buf.clear()

    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
        else:
            flush()
            out.append(t)
    flush()
    return out


def _fz_is_bomb_food_slang(norm: str, tokens: list[str]) -> bool:
    if re.search(r"\bmake\s+(a|an)\s+bomb\b", norm):
        return False
    if any(p in norm for p in _FZ_BOMB_SLANG) and any(t in _FZ_FOOD_NOUNS for t in tokens):
        return True
    food_pat = "|".join(sorted(_FZ_FOOD_NOUNS))
    if re.search(rf"\b(make|cook)\s+bomb\s+(ass\s+)?({food_pat})\b", norm):
        return True
    return False


def _fz_score(query: str) -> tuple[float, list[str]]:
    """Return (score, triggers). Score 0.0 → hard block."""
    norm = _fz_normalize(query)
    tokens = _fz_collapse_spaced(norm.split())
    norm = " ".join(tokens).strip()
    compact = re.sub(r"[^a-z0-9]+", "", norm)
    triggers: list[str] = []
    support_self_harm_context = False

    # --- Hard blocks ---

    # Hate slurs
    if any(t in _FZ_HATE_SLURS for t in tokens) or any(
        s.replace(" ", "") in compact for s in _FZ_HATE_SLURS
    ):
        triggers.append("hate:slur_token")
        return 0.0, triggers
    for rx in _FZ_HATE_SLUR_RES:
        if rx.search(query):
            triggers.append("hate:slur_regex")
            return 0.0, triggers

    # PII / doxxing
    digits = re.sub(r"\D", "", query)
    phone_like = len(digits) == 10 or (len(digits) == 11 and digits.startswith("1"))
    if (
        _FZ_EMAIL_RE.search(query)
        or _FZ_PHONE_RE.search(query)
        or _FZ_SSN_RE.search(query)
        or _FZ_DOXX_RE.search(query)
        or phone_like
    ):
        triggers.append("pii:doxxing")
        return 0.0, triggers

    # Minors + sex
    if any(t in _FZ_MINOR_TERMS for t in tokens) and any(t in _FZ_SEX_TERMS for t in tokens):
        triggers.append("nsfw:minors")
        return 0.0, triggers
    if _FZ_MINOR_EXPLICIT_RE.search(norm):
        triggers.append("nsfw:minors_explicit")
        return 0.0, triggers

    # Self-harm
    self_harm_hit = _FZ_SELF_HARM_RE.search(norm) or any(
        p in norm
        for p in (
            "kill myself",
            "killing myself",
            "end my life",
            "end myself",
            "do not want to be alive",
            "don't want to be alive",
            "suicide",
            "suicidal",
            "self harm",
            "self-harm",
            "hurt myself",
            "cut myself",
        )
    )
    if self_harm_hit:
        has_support_context = bool(_FZ_SELF_HARM_SUPPORT_RE.search(norm))
        explicit_ideation = bool(
            re.search(
                r"\b(i want to|i'm going to|i am going to|gonna|thoughts of|thinking of|"
                r"ways to|plan to|kill myself|hurt myself|end my life)\b",
                norm,
            )
        )
        if has_support_context and not explicit_ideation:
            support_self_harm_context = True
            triggers.append("support:self_harm_resource")
        else:
            triggers.append("self_harm")
            return 0.0, triggers

    # Drugs — procurement intent (buy/make/get)
    if _FZ_DRUG_INTENT_RE.search(norm):
        if "weedkiller" not in compact:
            triggers.append("drugs:intent")
            return 0.0, triggers

    # Drugs — first-person active use / addiction crisis
    # e.g. "I've been using heroin and I can't stop", "I'm addicted to meth"
    if _FZ_DRUG_PERSONAL_USE_RE.search(norm):
        triggers.append("drugs:personal_use_crisis")
        return 0.0, triggers

    # Additional explicit intent patterns not reliably captured by token checks
    if _FZ_VIOLENCE_INTENT_RE.search(norm):
        triggers.append("violence:intent")
        return 0.0, triggers
    if _FZ_WEAPON_INTENT_RE.search(norm):
        triggers.append("weapons:intent")
        return 0.0, triggers
    if _FZ_CYBERCRIME_INTENT_RE.search(norm):
        triggers.append("cybercrime:intent")
        return 0.0, triggers

    # Explosives (allow food slang)
    if (
        _FZ_EXPLOSIVES_RE.search(norm) or _FZ_MAKE_BOMB_RE.search(norm)
        or re.search(r"\b(improvised|ied)\s+explosive\b", norm)
        or "pipebomb" in compact
        or "molotov" in compact
        or "detonator" in compact
        or re.search(r"\bied\b", norm)
    ) and not _fz_is_bomb_food_slang(norm, tokens):
        triggers.append("explosives:intent")
        return 0.0, triggers

    # Violence threats (allow idioms and photo contexts)
    if _FZ_VIOLENCE_THREAT_RE.search(norm):
        if not any(idm in norm for idm in _FZ_VIOLENCE_IDIOMS) and not any(
            idm in norm for idm in _FZ_PHOTO_IDIOMS
        ):
            triggers.append("violence:threat")
            return 0.0, triggers

    # Extremism with intent
    if _FZ_EXTREMISM_INTENT_RE.search(norm) or (
        any(t in norm for t in _FZ_EXTREMISM_TERMS)
        and re.search(r"\b(join|support|fund|donate|recruit)\b", norm)
    ):
        triggers.append("extremism:intent")
        return 0.0, triggers

    # NSFW instructions / explicit body
    if _FZ_NSFW_INSTRUCTION_RE.search(norm):
        triggers.append("nsfw:instruction")
        return 0.0, triggers
    if any(t in _FZ_EXPLICIT_BODY for t in tokens):
        triggers.append("nsfw:explicit_body")
        return 0.0, triggers

    # --- Soft caps (return lower score but not 0.0) ---
    score = 0.95

    if support_self_harm_context:
        score = min(score, 0.15)

    if _FZ_LOW_SENSITIVITY_RE.search(norm):
        triggers.append("sensitivity:low_cap")
        score = min(score, 0.2)
    elif _FZ_MEDIUM_SENSITIVITY_RE.search(norm):
        triggers.append("sensitivity:medium_cap")
        score = min(score, 0.55)

    # Drug/addiction mentions without procurement/manufacture intent are allowed,
    # but they stay in medium-sensitivity range.
    if any(t in _FZ_DRUG_TERMS for t in tokens) and not ("weed" in tokens and "killer" in tokens):
        triggers.append("drugs:mention_soft_cap")
        if _FZ_DRUG_SUPPORT_RE.search(norm):
            score = min(score, 0.45)
        else:
            score = min(score, 0.55)

    # Profanity near food vs standalone
    profanity_tokens = set(tokens)
    if "bomb ass" in norm:
        profanity_tokens.discard("ass")
        profanity_tokens.discard("arse")
    if any(t in _FZ_PROFANITY for t in profanity_tokens):
        if any(t in _FZ_FOOD_NOUNS for t in tokens):
            triggers.append("profanity:near_food")
            score = min(score, 0.85)
        else:
            triggers.append("profanity:general")
            score = min(score, 0.2)

    # Spam / scam
    if len(_FZ_URL_RE.findall(query)) >= 2:
        triggers.append("spam:multi_url")
        score = min(score, 0.2)
    if _FZ_REPEAT_CHAR_RE.search(query):
        triggers.append("spam:repeat_chars")
        score = min(score, 0.2)
    if any(p in norm for p in _FZ_SCAM_PHRASES):
        triggers.append("spam:scam_phrase")
        score = min(score, 0.2)

    return max(0.0, min(1.0, score)), triggers


class FuzzyBlocklistStrategy:
    """Deterministic ad eligibility filter with obfuscation-resistant normalization.

    Improvements over RegexBlocklistStrategy:
    - Leetspeak normalisation before matching
    - Collapsed-spaced-letter detection (catches 's u i c i d e')
    - Compact-form matching for run-together evasions
    - Violence / bomb idiom guards (won't block "killing it" or "bomb tacos")
    - PII / doxxing detection
    - Spam / scam heuristics

    Returns a BlocklistMatch for hard blocks (score == 0.0) and soft caps
    (0 < score < 1.0) when named triggers fire. Pass-through queries (score == 0.95,
    no triggers) return None. Soft caps are applied as an eligibility ceiling
    inside EligibilityScorer._score_internal after semantic scoring.
    """

    def match(self, query: str) -> BlocklistMatch | None:
        score, triggers = _fz_score(query)
        # Only return a match when something actually fired (hard block or named soft cap).
        # Pass-through queries have score=0.95 with empty triggers — return None.
        if triggers:
            rule = "; ".join(triggers)
            return BlocklistMatch(strategy="fuzzy", rule=rule, score=score)
        return None


class NoopBlocklistStrategy:
    """Blocklist strategy that never blocks (useful for benchmarking)."""

    def match(self, query: str) -> BlocklistMatch | None:  # noqa: ARG002
        return None


class LRBlocklistStrategy:
    """Logistic Regression blocklist using a pre-trained sklearn TF-IDF + LR pipeline.

    Loads a joblib dict with keys:
        "model"     — sklearn Pipeline (TF-IDF + LogisticRegression), classes [0=safe, 1=harmful]
        "threshold" — float, minimum harmful probability to block

    Conservative fail-safe: any exception during inference returns a match (blocks).
    """

    def __init__(self, model_path: str | Path) -> None:
        import joblib

        artifact = joblib.load(model_path)
        self._model = artifact["model"]
        self._threshold: float = float(artifact.get("threshold", 0.8))

    def match(self, query: str) -> BlocklistMatch | None:
        try:
            harmful_prob = float(self._model.predict_proba([query])[0][1])
            if harmful_prob >= self._threshold:
                return BlocklistMatch(
                    strategy="lr",
                    rule=f"harmful_prob={harmful_prob:.4f} >= {self._threshold}",
                )
            return None
        except Exception as exc:  # noqa: BLE001
            return BlocklistMatch(strategy="lr", rule=f"error:{exc}")


class CompositeBlocklistStrategy:
    """Runs multiple blocklist strategies in parallel (OR logic).

    Returns a match if ANY strategy fires or raises an exception.
    All strategies are always consulted — the first non-None result is returned.
    Exceptions are treated as blocks (fail-safe / conservative).
    """

    def __init__(self, strategies: list[BlocklistStrategy]) -> None:
        self._strategies = strategies

    def match(self, query: str) -> BlocklistMatch | None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        soft_caps: list[BlocklistMatch] = []

        with ThreadPoolExecutor(max_workers=len(self._strategies)) as executor:
            futures = {executor.submit(s.match, query): s for s in self._strategies}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        if result.score == 0.0:
                            # Hard block — cancel queued futures and return immediately
                            for f in futures:
                                f.cancel()
                            return result
                        soft_caps.append(result)
                except Exception as exc:  # noqa: BLE001
                    strategy_name = type(futures[future]).__name__
                    for f in futures:
                        f.cancel()
                    return BlocklistMatch(strategy=strategy_name, rule=f"error:{exc}", score=0.0)

        # No hard blocks — return most restrictive soft cap if any
        if soft_caps:
            return min(soft_caps, key=lambda m: m.score)
        return None


class ToxicBertBlocklistStrategy:
    """ML-based blocklist using unitary/toxic-bert to detect harmful content.

    Loads the model once at initialization. Uses torch.inference_mode() for
    efficient inference. Blocks queries where toxic probability exceeds threshold.
    """

    def __init__(self, threshold: float = 0.5, model_name: str = "unitary/toxic-bert") -> None:
        self._threshold = threshold
        self._model_name = model_name

        # Detect device (MPS for Apple Silicon, else CPU)
        self._device = (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )

        # Load tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self._device.type == "mps" else None,
        ).to(self._device).eval()

        # Warmup (triggers JIT compilation, allocates buffers)
        with torch.inference_mode():
            _ = self._classify_toxic(["warmup"])

    @torch.inference_mode()
    def _classify_toxic(self, texts: list[str]) -> list[float]:
        """Return toxic probability for each text (0.0 = safe, 1.0 = toxic)."""
        enc = self._tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        enc = {k: v.to(self._device) for k, v in enc.items()}

        logits = self._model(**enc).logits
        probs = F.softmax(logits, dim=-1)  # shape (batch, 2)
        return probs[:, 1].float().cpu().tolist()  # toxic prob for each text

    def match(self, query: str) -> BlocklistMatch | None:
        """Check if query is toxic above threshold."""
        toxic_prob = self._classify_toxic([query])[0]
        if toxic_prob >= self._threshold:
            return BlocklistMatch(
                strategy="toxic_bert",
                rule=f"toxic_prob={toxic_prob:.4f} >= {self._threshold}",
            )
        return None



def create_blocklist_strategy(
    name: str,
    blocklist_path: str | Path | None = None,
    toxic_threshold: float = 0.5,
    lr_model_path: str | Path | None = None,
) -> BlocklistStrategy:
    """Factory for blocklist strategy, configured by name.

    Supported strategy names:
        - "all" or "composite": regex + fuzzy in parallel (recommended)
        - "regex": regex patterns loaded from file or built-ins
        - "fuzzy": deterministic filter with leetspeak + obfuscation-resistant normalization
        - "lr": sklearn LR pipeline (requires lr_model_path)
        - "toxic_bert" or "bert": ML-based toxic content detection using unitary/toxic-bert
        - "none": never block

    Returns:
        Concrete BlocklistStrategy selected from `name`.

    Raises:
        ValueError: if `name` is unknown, or if `name="lr"` without a model path.
    """
    key = (name or "all").strip().lower()
    if key in {"none", "noop", "off", "disabled"}:
        return NoopBlocklistStrategy()
    if key in {"regex", "re"}:
        return RegexBlocklistStrategy(_load_blocklist(blocklist_path))
    if key in {"fuzzy", "deterministic"}:
        return FuzzyBlocklistStrategy()
    if key in {"lr", "logistic", "logistic_regression"}:
        if lr_model_path is None:
            raise ValueError("lr_model_path required for LR blocklist strategy")
        return LRBlocklistStrategy(lr_model_path)
    if key in {"toxic_bert", "bert", "toxic", "ml"}:
        return ToxicBertBlocklistStrategy(threshold=toxic_threshold)
    if key in {"all", "composite"}:
        return CompositeBlocklistStrategy(
            [
                RegexBlocklistStrategy(_load_blocklist(blocklist_path)),
                FuzzyBlocklistStrategy(),
            ]
        )
    raise ValueError(f"Unknown blocklist strategy: {name!r}")


def _build_sub_centroids(
    vecs: NDArray[np.float32], k: int = 3, n_iter: int = 20
) -> list[NDArray[np.float32]]:
    """Build k sub-centroids via k-means (pure numpy).

    Produces better cluster coverage than a single mean centroid for large,
    semantically diverse phrase groups (e.g. grief_loss covers death, divorce,
    pregnancy loss — one centroid misses the fringes).

    Args:
        vecs:   Unit-normalised embeddings, shape (n, dim).
        k:      Number of sub-centroids (capped at n if n < k).
        n_iter: Maximum k-means iterations.

    Returns:
        List of k unit-normalised centroid vectors.
    """
    n = len(vecs)
    if n <= k:
        return [vecs[i].astype(np.float32) for i in range(n)]

    rng = np.random.default_rng(42)
    indices = rng.choice(n, size=k, replace=False)
    centers: NDArray[np.float32] = vecs[indices].copy()

    for _ in range(n_iter):
        # Cosine similarity assignment (vecs and centers are unit-normalised)
        sims = vecs @ centers.T  # (n, k)
        assignments = np.argmax(sims, axis=1)  # (n,)

        new_centers = np.zeros_like(centers)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                raw = vecs[mask].mean(axis=0)
                norm = float(np.linalg.norm(raw))
                new_centers[c] = (raw / norm).astype(np.float32) if norm > 0 else raw
            else:
                new_centers[c] = centers[c]

        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    return [centers[c].astype(np.float32) for c in range(k)]


class EligibilityScorer:
    """Scores how appropriate it is to show ads for a given query.

    Usage:
        scorer = EligibilityScorer(embedding_service)
        vec = embedding_service.embed("best running shoes for marathon")
        result = scorer.score("best running shoes for marathon", vec)
        # result["eligibility"] → 0.95
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        blocklist_path: str | Path | None = None,
        blocklist_strategy: BlocklistStrategy | None = None,
        insensitive_model_path: str | Path | None = None,
    ) -> None:
        # Blocklist behavior is configurable via strategy.
        self._blocklist_strategy: BlocklistStrategy = (
            blocklist_strategy
            if blocklist_strategy is not None
            else create_blocklist_strategy("regex", blocklist_path=blocklist_path)
        )

        # v5 insensitive-query model (Layer 2a).
        model_path = Path(insensitive_model_path) if insensitive_model_path is not None else _DEFAULT_INSENSITIVE_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Insensitive-query model not found: {model_path}")
        self._insensitive_model = InsensitiveQueryModel(model_path)

        # Sensitivity cluster centroids (Layer 2b) — consulted alongside the v5 model.
        # Each cluster's phrases are embedded and reduced to k=3 sub-centroids via k-means
        # so the centroid captures semantically diverse sub-groups within a cluster
        # (e.g. grief_loss covers death, divorce, pregnancy loss — a single mean misses the fringes).
        self._cluster_centroids: dict[str, list[NDArray[np.float32]]] = {}
        self._cluster_names: list[str] = []
        for cluster_name, phrases in _SENSITIVITY_CLUSTERS.items():
            if not phrases:
                continue
            vecs = embedding_service.embed_batch(phrases)
            self._cluster_centroids[cluster_name] = _build_sub_centroids(vecs, k=3)
            self._cluster_names.append(cluster_name)

        # Pre-compute commercial affinity centroid (Layer 3).
        comm_vecs = embedding_service.embed_batch(_COMMERCIAL_EXEMPLARS)  # (n, dim)
        centroid = comm_vecs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        self._commercial_centroid: NDArray[np.float32] = (
            (centroid / norm).astype(np.float32) if norm > 0 else centroid
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_blocked(self, query: str) -> bool:
        """Return True if the query is hard-blocked (score == 0.0).

        No embedding required. Runs in microseconds and can be called before the
        embedding step to short-circuit the entire pipeline.
        Soft-cap matches (0 < score < 1) return False — they are handled inside score().
        """
        match = self.blocklist_match(query)
        return match is not None and match.score == 0.0

    def blocklist_match(self, query: str) -> BlocklistMatch | None:
        """Return blocklist match info (or None if not blocked)."""
        return self._blocklist_strategy.match(query)

    def score(self, query: str, query_embedding: NDArray[np.float32]) -> float:
        """Return eligibility score in [0.0, 1.0].

        Args:
            query: Raw query text (used for blocklist regex).
            query_embedding: Pre-computed unit-normalised query vector.

        Returns:
            0.0  → do not show ads.
            1.0  → perfectly appropriate to show ads.
        """
        return self._score_internal(query, query_embedding)["eligibility"]

    def score_with_metadata(
        self, query: str, query_embedding: NDArray[np.float32]
    ) -> dict:
        """Score with full debug metadata for the API response's metadata field.

        Returns a dict with:
            eligibility                  (float)    final score in [0, 1]
            blocklist_triggered          (bool)     True if regex/ML blocklist fired
            hard_block_cluster           (str|None) which signal hard-blocked (model or cluster name)
            sensitivity_penalty          (float)    max(model_prob, cluster_penalty) — conservative OR
            commercial_boost             (float)    Layer 3 boost factor
            top_cluster                  (str)      highest-similarity signal source
            top_cluster_sim              (float)    its cosine similarity (or model probability)
            commercial_sim               (float)    raw cosine sim to commercial centroid
            co_activation_bonus          (float)    reserved (currently 0.0)
            distress_signal              (float)    reserved (currently 0.0)
            amplifier                    (float)    reserved (currently 1.0)
            insensitive_model_probability (float)   v5 model's insensitivity probability
            insensitive_model_threshold  (float)    v5 model's hard-block threshold
            cluster_penalty              (float)    max continuous penalty from cluster centroids
        """
        return self._score_internal(query, query_embedding)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_internal(self, query: str, query_embedding: NDArray[np.float32]) -> dict:
        # ----------------------------------------------------------------
        # Layer 1: Regex / ML blocklist
        # ----------------------------------------------------------------
        match = self._blocklist_strategy.match(query)
        fuzzy_cap: float = 1.0
        if match is not None:
            if match.score == 0.0:
                return {
                    "eligibility": 0.0,
                    "blocklist_triggered": True,
                    "blocklist_strategy": match.strategy,
                    "blocklist_rule": match.rule,
                    "sensitivity_penalty": 1.0,
                    "commercial_boost": 0.0,
                    "top_cluster": "blocklist",
                    "top_cluster_sim": 1.0,
                    "commercial_sim": 0.0,
                    "hard_block_cluster": None,
                    "distress_signal": 0.0,
                    "amplifier": 1.0,
                    "co_activation_bonus": 0.0,
                }
            # Soft cap — continue scoring but cap final eligibility at this value
            fuzzy_cap = match.score

        # Strip generic "I want / suggest / looking for ..." scaffolding only
        # after regex+fuzzy checks have already inspected the raw query text.
        query_for_semantic_scoring = _strip_request_intent_prefix(query)
        query_for_v5 = _filter_v5_false_positive_terms(query_for_semantic_scoring)

        # ----------------------------------------------------------------
        # Layer 2a: v5 insensitive-query model
        # ----------------------------------------------------------------
        try:
            insensitive_prob = self._insensitive_model.predict_insensitive_probability(
                query_for_v5
            )
        except Exception:
            # Conservative fail-safe: inference errors should never allow ads through.
            insensitive_prob = 1.0

        model_hard_block = insensitive_prob >= self._insensitive_model.threshold

        # ----------------------------------------------------------------
        # Layer 2b: Sensitivity cluster centroids (parallel signal)
        # Max cosine similarity across all sub-centroids determines each cluster's
        # contribution. Both a model hard-block AND a cluster hard-gate will suppress
        # ads (OR logic). The continuous penalty is the max of both signals (conservative).
        # ----------------------------------------------------------------
        top_cluster_name: str = "insensitive_model_v5"
        top_cluster_sim: float = float(insensitive_prob)
        hard_block_cluster: str | None = None
        cluster_penalty: float = 0.0

        for cluster_name in self._cluster_names:
            sub_centroids = self._cluster_centroids[cluster_name]
            # Max cosine similarity across all sub-centroids for this cluster
            cluster_sim = max(float(query_embedding @ c) for c in sub_centroids)

            # Track the highest-similarity cluster for metadata
            if cluster_sim > top_cluster_sim:
                top_cluster_sim = cluster_sim
                top_cluster_name = cluster_name

            # Per-cluster hard gate
            threshold = _CLUSTER_HARD_THRESHOLDS.get(cluster_name, 1.0)
            if cluster_sim >= threshold and hard_block_cluster is None:
                hard_block_cluster = cluster_name

            # Continuous penalty contribution (only above noise floor)
            if cluster_sim >= SENSITIVITY_MIN_SIM:
                scale = _CLUSTER_SENSITIVITY_SCALE.get(cluster_name, SENSITIVITY_SCALE)
                penalty = min(cluster_sim * scale, 1.0)
                cluster_penalty = max(cluster_penalty, penalty)

        # ----------------------------------------------------------------
        # Hard block if either signal fires
        # ----------------------------------------------------------------
        if model_hard_block or hard_block_cluster is not None:
            block_source = (
                hard_block_cluster if hard_block_cluster is not None else "insensitive_model_v5"
            )
            return {
                "eligibility": 0.0,
                "blocklist_triggered": False,
                "hard_block_cluster": block_source,
                "sensitivity_penalty": 1.0,
                "commercial_boost": 0.0,
                "top_cluster": top_cluster_name,
                "top_cluster_sim": round(top_cluster_sim, 4),
                "commercial_sim": 0.0,
                "distress_signal": 0.0,
                "amplifier": 1.0,
                "co_activation_bonus": 0.0,
                "insensitive_model_probability": round(float(insensitive_prob), 4),
                "insensitive_model_threshold": round(float(self._insensitive_model.threshold), 4),
                "cluster_penalty": round(cluster_penalty, 4),
            }

        # Conservative combination: take the max of both continuous signals.
        sensitivity_penalty = max(float(insensitive_prob), cluster_penalty)

        distress_signal = 0.0
        amplifier = 1.0
        co_activation_bonus = 0.0

        # ----------------------------------------------------------------
        # Layer 3: Commercial affinity boost
        # ----------------------------------------------------------------
        commercial_sim = float(self._commercial_centroid @ query_embedding)
        if sensitivity_penalty > 0.70:
            # High sensitivity: apply no commercial boost — let penalty dominate
            commercial_boost = 1.0
        else:
            commercial_boost = COMMERCIAL_MIN + COMMERCIAL_RANGE * min(
                max(commercial_sim * COMMERCIAL_SCALE, 0.0), 1.0
            )

        eligibility = float(
            np.clip(
                min((1.0 - sensitivity_penalty) * commercial_boost, fuzzy_cap),
                0.0, 1.0,
            )
        )

        return {
            "eligibility": round(eligibility, 4),
            "blocklist_triggered": False,
            "hard_block_cluster": None,
            "sensitivity_penalty": round(sensitivity_penalty, 4),
            "commercial_boost": round(commercial_boost, 4),
            "top_cluster": top_cluster_name,
            "top_cluster_sim": round(top_cluster_sim, 4),
            "commercial_sim": round(commercial_sim, 4),
            "distress_signal": round(distress_signal, 4),
            "amplifier": round(amplifier, 4),
            "co_activation_bonus": round(co_activation_bonus, 4),
            "insensitive_model_probability": round(float(insensitive_prob), 4),
            "insensitive_model_threshold": round(float(self._insensitive_model.threshold), 4),
            "cluster_penalty": round(cluster_penalty, 4),
        }

    @property
    def num_clusters(self) -> int:
        """Number of sensitivity clusters participating in semantic scoring."""
        return len(self._cluster_names)

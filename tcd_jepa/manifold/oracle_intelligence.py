"""Universal Operationalizer: transforms any analytical signal into any operational
intelligence framing — unlimited lenses, unlimited categories, unlimited domains.

Not limited to 7 fixed lenses. The engine dynamically generates intelligence
briefings for ANY operational context by combining:

1. **Signal taxonomy** — universal classification of all 23+ insight categories
2. **Lens definitions** — data-driven specifications (title, audience, signal filters,
   translation templates) that can be created by anyone
3. **Template engine** — renders insights into business language using structured
   templates with variable interpolation
4. **Lens auto-generator** — creates new lenses dynamically from data characteristics
5. **Lens library** — 50+ pre-built lenses across industries and functions

Usage:
    from tcd_jepa.manifold.oracle_intelligence import (
        OracleIntelligenceEngine, OracleContext, LensDefinition,
    )

    # Use built-in lenses
    engine = OracleIntelligenceEngine()
    briefing = engine.render_briefing(report, OracleContext(domain="retail"))

    # Create custom lens
    lens = LensDefinition(
        name="sustainability",
        title="SUSTAINABILITY INTELLIGENCE",
        audience="CSO, ESG Team",
        description="Environmental and social impact signals",
        signal_filters=["anomaly", "trend", "risk", "coverage_gap"],
        priority_keywords=["carbon", "emission", "waste", "compliance"],
        action_verb="mitigate",
        value_frame="impact reduction",
    )
    engine.register_lens(lens)
    briefing = engine.render_briefing(report, ctx, lenses=["sustainability"])

    # Auto-generate lenses from data
    auto_lenses = engine.auto_generate_lenses(report, ctx)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger("tcd_jepa.oracle_intelligence")


# ── Signal Taxonomy ──────────────────────────────────────────────────
# Universal classification of every insight category by its signal type.
# This enables any lens to select signals by type rather than hardcoded name.

SIGNAL_TAXONOMY = {
    # Category → set of signal types
    "cluster": {"structure", "segmentation", "identity"},
    "relationship": {"connection", "dependency", "flow"},
    "opportunity": {"growth", "synergy", "connection"},
    "risk": {"threat", "instability", "boundary"},
    "anomaly": {"outlier", "novelty", "deviation"},
    "trend": {"dynamics", "evolution", "momentum"},
    "strategic": {"strategic", "portfolio", "macro"},
    "knowledge_gap": {"void", "whitespace", "coverage"},
    "feedback_loop": {"dependency", "cycle", "operational"},
    "hierarchy": {"structure", "layers", "segmentation"},
    "topological_stability": {"resilience", "structure", "threat"},
    "causal_chain": {"flow", "causality", "dependency"},
    "information_bottleneck": {"flow", "operational", "fragility"},
    "intervention": {"fragility", "dependency", "operational"},
    "blind_spot": {"coverage", "uncertainty", "threat"},
    "confidence_field": {"quality", "uncertainty", "meta"},
    "coverage_gap": {"coverage", "quality", "void"},
    "topic_drift": {"dynamics", "evolution", "strategic"},
    "phase_transition": {"dynamics", "regime", "strategic"},
    "exploration_dynamics": {"coverage", "meta", "operational"},
    "model_quality": {"quality", "meta", "confidence"},
    "insight_reliability": {"quality", "meta", "confidence"},
    "corpus_diagnosis": {"quality", "coverage", "meta"},
}

# Reverse index: signal_type → set of categories
SIGNAL_TYPE_TO_CATEGORIES = {}
for _cat, _types in SIGNAL_TAXONOMY.items():
    for _t in _types:
        SIGNAL_TYPE_TO_CATEGORIES.setdefault(_t, set()).add(_cat)


# ── Severity framing ────────────────────────────────────────────────

SEVERITY_URGENCY = {
    "high": {"prefix": "URGENT", "timeframe": "immediately", "imperative": "must"},
    "medium": {"prefix": "IMPORTANT", "timeframe": "this quarter", "imperative": "should"},
    "low": {"prefix": "ADVISORY", "timeframe": "when capacity allows", "imperative": "consider"},
}


# ── Context ──────────────────────────────────────────────────────────

@dataclass
class OracleContext:
    """Operational context that shapes how intelligence is delivered."""
    domain: str = "business"
    entity_type: str = "segment"
    actor: str = "your team"
    resource_noun: str = "resources"
    currency: str = "value"
    cluster_labels: dict = field(default_factory=dict)
    stakeholders: dict = field(default_factory=dict)


# ── Lens Definition ──────────────────────────────────────────────────

@dataclass
class LensDefinition:
    """Data-driven specification for an intelligence lens.

    Any lens can be created by filling in this structure — no code changes needed.
    The engine uses signal_filters and signal_types to select relevant insights,
    then renders them using the translation templates.
    """
    name: str                              # unique identifier (e.g. "revenue", "sustainability")
    title: str                             # display title (e.g. "REVENUE INTELLIGENCE")
    audience: str                          # who receives this (e.g. "Sales, BD, Revenue Leadership")
    description: str = ""                  # what this lens covers

    # ── Signal selection ──
    signal_filters: list[str] = field(default_factory=list)   # explicit category names to include
    signal_types: list[str] = field(default_factory=list)     # signal types from taxonomy (e.g. "growth", "threat")
    exclude_categories: list[str] = field(default_factory=list)  # categories to exclude
    severity_filter: Optional[str] = None  # only show this severity level
    min_confidence: float = 0.0            # minimum confidence threshold

    # ── Language framing ──
    action_verb: str = "address"           # default action verb ("invest", "mitigate", "investigate")
    value_frame: str = "value"             # what value means in this lens ("revenue", "efficiency", "risk reduction")
    risk_frame: str = "risk"               # how risk is framed ("exposure", "liability", "threat")
    opportunity_frame: str = "opportunity" # how opportunities are framed ("growth", "efficiency gain", "market entry")
    entity_frame: str = ""                 # override for entity_type in this lens (empty = use context)
    positive_outcome: str = "growth"       # desired outcome ("revenue growth", "risk reduction", "efficiency")
    negative_outcome: str = "loss"         # undesired outcome ("churn", "breach", "inefficiency")

    # ── Priority keywords ──
    priority_keywords: list[str] = field(default_factory=list)  # boost insights containing these words
    priority_boost: float = 0.2            # confidence boost for priority keyword matches

    # ── Custom translation overrides (category → template dict) ──
    custom_translations: dict = field(default_factory=dict)
    # Format: {"category_name": {"headline": "template {var}", "body": "template {var}", "action": "template {var}"}}

    def matches_insight(self, insight) -> bool:
        """Check if an insight matches this lens's filters."""
        cat = insight.category

        # Explicit exclusion
        if cat in self.exclude_categories:
            return False

        # Severity filter
        if self.severity_filter and insight.severity != self.severity_filter:
            return False

        # Confidence filter
        if insight.confidence < self.min_confidence:
            return False

        # Explicit category match
        if self.signal_filters and cat in self.signal_filters:
            return True

        # Signal type match via taxonomy
        if self.signal_types:
            cat_types = SIGNAL_TAXONOMY.get(cat, set())
            if cat_types & set(self.signal_types):
                return True

        # If no filters specified, match everything
        if not self.signal_filters and not self.signal_types:
            return True

        return False

    def priority_score(self, insight) -> float:
        """Compute priority score for sorting within this lens."""
        score = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(insight.severity, 0)
        score += insight.confidence

        # Boost for priority keywords
        if self.priority_keywords:
            text = f"{insight.title} {insight.description}".lower()
            for kw in self.priority_keywords:
                if kw.lower() in text:
                    score += self.priority_boost
        return score


# ── Universal Translator ─────────────────────────────────────────────

def _extract_number(text: str, pattern: str) -> Optional[str]:
    """Extract first numeric match from text."""
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _cluster_name(raw: str, ctx: OracleContext) -> str:
    """Resolve cluster reference to business name."""
    match = re.search(r"cluster[_\s]*(\d+)", raw, re.IGNORECASE)
    if match:
        cid = int(match.group(1))
        if cid in ctx.cluster_labels:
            return f'"{ctx.cluster_labels[cid]}"'
    return raw


def _extract_clusters(text: str, ctx: OracleContext) -> tuple[str, str]:
    """Extract two cluster references from text."""
    clusters = re.findall(r"clusters?\s*(\d+)", text, re.IGNORECASE)
    c1 = _cluster_name(f"cluster {clusters[0]}", ctx) if clusters else "area A"
    c2 = _cluster_name(f"cluster {clusters[1]}", ctx) if len(clusters) > 1 else "area B"
    return c1, c2


def _universal_translate(insight, lens: LensDefinition, ctx: OracleContext) -> Optional[dict]:
    """Universal translation engine: turns any insight into business language for any lens.

    Uses a cascade of strategies:
    1. Custom translation override (if lens defines one for this category)
    2. Category-specific smart translation (extracts structured data from insight)
    3. Universal template fallback (works for any category/lens combination)
    """
    cat = insight.category
    sev = insight.severity
    ent = lens.entity_frame or ctx.entity_type
    actor = ctx.actor
    resource = ctx.resource_noun
    urgency = SEVERITY_URGENCY.get(sev, SEVERITY_URGENCY["medium"])

    # ── Strategy 1: Custom translation override ──
    if cat in lens.custom_translations:
        tmpl = lens.custom_translations[cat]
        vars_dict = _build_template_vars(insight, lens, ctx)
        return {
            "headline": _interpolate(tmpl.get("headline", insight.title), vars_dict),
            "body": _interpolate(tmpl.get("body", insight.description), vars_dict),
        }

    # ── Strategy 2: Category-specific smart translation ──
    result = _smart_translate(insight, lens, ctx)
    if result:
        return result

    # ── Strategy 3: Universal template fallback ──
    return _fallback_translate(insight, lens, ctx)


def _build_template_vars(insight, lens: LensDefinition, ctx: OracleContext) -> dict:
    """Build template variable dictionary from insight + context."""
    c1, c2 = _extract_clusters(insight.title, ctx)
    ent = lens.entity_frame or ctx.entity_type
    count = _extract_number(insight.description, r"(\d+)\s*(?:semantic\s+)?chunks") or "?"
    pct = _extract_number(insight.description, r"\((\d+(?:\.\d+)?)%") or "?"
    coherence = _extract_number(insight.description, r"coherence:\s*(\d+(?:\.\d+)?)") or "?"
    z_score = _extract_number(insight.title, r"z-score:\s*([\d.]+)") or "?"
    urgency = SEVERITY_URGENCY.get(insight.severity, SEVERITY_URGENCY["medium"])

    return {
        "title": insight.title,
        "description": insight.description,
        "category": insight.category,
        "severity": insight.severity,
        "confidence": f"{insight.confidence:.0%}",
        "entities": ", ".join(insight.entities[:5]) if insight.entities else ent,
        "entity_type": ent,
        "actor": ctx.actor,
        "resource": ctx.resource_noun,
        "currency": ctx.currency,
        "domain": ctx.domain,
        "cluster_a": c1,
        "cluster_b": c2,
        "count": count,
        "pct": pct,
        "coherence": coherence,
        "z_score": z_score,
        "value_frame": lens.value_frame,
        "risk_frame": lens.risk_frame,
        "opportunity_frame": lens.opportunity_frame,
        "positive_outcome": lens.positive_outcome,
        "negative_outcome": lens.negative_outcome,
        "action_verb": lens.action_verb,
        "urgency_prefix": urgency["prefix"],
        "timeframe": urgency["timeframe"],
        "imperative": urgency["imperative"],
    }


def _interpolate(template: str, vars_dict: dict) -> str:
    """Safe template interpolation with {var} syntax."""
    try:
        return template.format_map(vars_dict)
    except (KeyError, ValueError):
        # Graceful fallback — return template with unresolved vars removed
        result = template
        for key, val in vars_dict.items():
            result = result.replace(f"{{{key}}}", str(val))
        return result


# ── Smart translators (category-aware) ───────────────────────────────

_SMART_TRANSLATORS: dict[str, Callable] = {}


def _register_translator(category: str):
    """Decorator to register a smart translator for a category."""
    def decorator(func):
        _SMART_TRANSLATORS[category] = func
        return func
    return decorator


@_register_translator("cluster")
def _translate_cluster(insight, lens, ctx):
    ent = lens.entity_frame or ctx.entity_type
    entities_str = ", ".join(insight.entities[:5]) if insight.entities else ent
    count = _extract_number(insight.description, r"(\d+)\s*semantic chunks")
    pct = _extract_number(insight.description, r"\((\d+(?:\.\d+)?)%")
    coherence = _extract_number(insight.description, r"coherence:\s*(\d+(?:\.\d+)?)")
    coh_val = float(coherence) if coherence else 0

    strength = "strongly cohesive" if coh_val > 0.7 else "moderately cohesive" if coh_val > 0.4 else "loosely defined"

    sev_label = "HIGH-VALUE" if insight.severity == "high" else "NOTABLE"
    return {
        "headline": f"{sev_label} {ent.upper()}: {entities_str}",
        "body": (
            f"  {count or '?'} items ({pct or '?'}% of portfolio), {strength}.\n"
            f"  \u2192 ACTION: {ctx.actor.title()} {lens.action_verb} dedicated {lens.value_frame} coverage for this {ent}."
        ),
    }


@_register_translator("opportunity")
def _translate_opportunity(insight, lens, ctx):
    c1, c2 = _extract_clusters(insight.title, ctx)
    count = _extract_number(insight.title, r"\((\d+)\s*connections?\)")
    return {
        "headline": f"{lens.opportunity_frame.upper()}: {c1} \u2194 {c2}",
        "body": (
            f"  {count or 'Multiple'} connections found between {c1} and {c2}.\n"
            f"  Natural affinity suggests {lens.positive_outcome} potential.\n"
            f"  \u2192 ACTION: {ctx.actor.title()} should {lens.action_verb} to capture this {lens.opportunity_frame}."
        ),
    }


@_register_translator("risk")
def _translate_risk(insight, lens, ctx):
    ent = lens.entity_frame or ctx.entity_type
    urgency = SEVERITY_URGENCY.get(insight.severity, SEVERITY_URGENCY["medium"])
    return {
        "headline": f"{urgency['prefix']} {lens.risk_frame.upper()}: unstable {ent} classification",
        "body": (
            f"  This item sits between {ent}s \u2014 classification is unstable.\n"
            f"  \u2192 ACTION: {ctx.actor.title()} {urgency['imperative']} {lens.action_verb} {urgency['timeframe']}."
        ),
    }


@_register_translator("anomaly")
def _translate_anomaly(insight, lens, ctx):
    z = _extract_number(insight.title, r"z-score:\s*([\d.]+)")
    urgency = SEVERITY_URGENCY.get(insight.severity, SEVERITY_URGENCY["medium"])
    return {
        "headline": f"ANOMALY: activity {z or '?'}\u03c3 outside normal ({insight.severity} severity)",
        "body": (
            f"  Does not fit established patterns.\n"
            f"  \u2192 ACTION: {ctx.actor.title()} {urgency['imperative']} investigate {urgency['timeframe']}. "
            f"May indicate {lens.risk_frame} or emerging {lens.opportunity_frame}."
        ),
    }


@_register_translator("trend")
def _translate_trend(insight, lens, ctx):
    ent = lens.entity_frame or ctx.entity_type
    is_dynamic = "evolving" in insight.title.lower() or "rapidly" in insight.title.lower()
    ref = _cluster_name(insight.title, ctx)
    if is_dynamic:
        return {
            "headline": f"MOMENTUM: {ref} is evolving rapidly",
            "body": (
                f"  This {ent} shows strong {lens.positive_outcome} momentum.\n"
                f"  \u2192 ACTION: Allocate {ctx.resource_noun} now to capitalize on trajectory."
            ),
        }
    return {
        "headline": f"STABLE BASE: {ref}",
        "body": (
            f"  This {ent} is stable \u2014 reliable {lens.value_frame}.\n"
            f"  \u2192 ACTION: Low-touch. Redirect {ctx.resource_noun} to growth areas."
        ),
    }


@_register_translator("knowledge_gap")
def _translate_knowledge_gap(insight, lens, ctx):
    c1, c2 = _extract_clusters(insight.title, ctx)
    return {
        "headline": f"WHITESPACE: no coverage between {c1} and {c2}",
        "body": (
            f"  Nobody bridges these two areas. This is unaddressed territory.\n"
            f"  \u2192 ACTION: Evaluate whether to allocate {ctx.resource_noun} here for {lens.positive_outcome}."
        ),
    }


@_register_translator("information_bottleneck")
def _translate_bottleneck(insight, lens, ctx):
    chunk_ref = re.search(r"chunk\s*(\d+)", insight.title, re.IGNORECASE)
    idx = chunk_ref.group(1) if chunk_ref else "a critical item"
    pct = _extract_number(insight.description, r"(\d+(?:\.\d+)?)%")
    return {
        "headline": f"BOTTLENECK: item {idx} carries {pct or '?'}% of flow",
        "body": (
            f"  A disproportionate share of {lens.value_frame} routes through this point.\n"
            f"  \u2192 ACTION: Build redundancy. Single points of failure are unacceptable."
        ),
    }


@_register_translator("intervention")
def _translate_intervention(insight, lens, ctx):
    count = _extract_number(insight.title, r"(\d+)\s*hub")
    return {
        "headline": f"FRAGILITY: removing {count or 'key'} items fragments the system",
        "body": (
            f"  These items hold the structure together.\n"
            f"  \u2192 ACTION: Treat as critical infrastructure. Protect and reinforce."
        ),
    }


@_register_translator("feedback_loop")
def _translate_feedback_loop(insight, lens, ctx):
    n = _extract_number(insight.title, r"(\d+)-node")
    return {
        "headline": f"CIRCULAR DEPENDENCY: {n or 'multi'}-way loop detected",
        "body": (
            f"  Changes to one element cascade through the loop unpredictably.\n"
            f"  \u2192 ACTION: Coordinate changes across all {n or 'involved'} elements jointly."
        ),
    }


@_register_translator("causal_chain")
def _translate_causal_chain(insight, lens, ctx):
    return {
        "headline": "CRITICAL PATH: primary influence chain identified",
        "body": (
            f"  Upstream disruptions cascade downstream along this path.\n"
            f"  \u2192 ACTION: Optimize and protect this sequence for {lens.value_frame} continuity."
        ),
    }


@_register_translator("hierarchy")
def _translate_hierarchy(insight, lens, ctx):
    ent = lens.entity_frame or ctx.entity_type
    return {
        "headline": f"STRUCTURE: multi-level {ent} hierarchy discovered",
        "body": (
            f"  Sub-{ent}s nest inside broader groups, revealing organizational layers.\n"
            f"  \u2192 ACTION: Set strategy at macro level, {lens.action_verb} at micro level."
        ),
    }


@_register_translator("topological_stability")
def _translate_stability(insight, lens, ctx):
    is_fragile = "fragile" in insight.title.lower()
    if is_fragile:
        return {
            "headline": f"FRAGILE POSITION: structurally vulnerable",
            "body": (
                f"  This structure could collapse with minor changes.\n"
                f"  \u2192 ACTION: Do not over-invest until {lens.risk_frame} is stabilized."
            ),
        }
    return {
        "headline": f"DEFENSIBLE POSITION: structurally robust",
        "body": (
            f"  Hard to dislodge. Natural moat.\n"
            f"  \u2192 ACTION: Invest to strengthen this position for lasting {lens.positive_outcome}."
        ),
    }


@_register_translator("blind_spot")
def _translate_blind_spot(insight, lens, ctx):
    ref = _cluster_name(insight.title, ctx)
    return {
        "headline": f"UNKNOWN UNKNOWNS: insufficient data on {ref}",
        "body": (
            f"  Cannot confidently assess {lens.risk_frame} in this area.\n"
            f"  \u2192 ACTION: Prioritize data collection before {lens.negative_outcome} exposure grows."
        ),
    }


@_register_translator("confidence_field")
def _translate_confidence(insight, lens, ctx):
    pct = _extract_number(insight.title, r"(\d+)%")
    ent = lens.entity_frame or ctx.entity_type
    return {
        "headline": f"DATA QUALITY: {pct or '?'}% of portfolio has confident analysis",
        "body": (
            f"  \u2192 TRUST insights about well-covered {ent}s.\n"
            f"  \u2192 CAUTION for poorly-covered {ent}s \u2014 need more data before acting."
        ),
    }


@_register_translator("coverage_gap")
def _translate_coverage_gap(insight, lens, ctx):
    ent = lens.entity_frame or ctx.entity_type
    return {
        "headline": f"COVERAGE GAP: not all {ent}s fully assessed",
        "body": (
            f"  {insight.description}\n"
            f"  \u2192 ACTION: Complete assessment coverage for reliable {lens.value_frame} analysis."
        ),
    }


@_register_translator("topic_drift")
def _translate_drift(insight, lens, ctx):
    ref = _cluster_name(insight.title, ctx)
    ratio = _extract_number(insight.description, r"([\d.]+)x\s*corpus")
    speed = "rapidly" if ratio and float(ratio) > 2 else "steadily"
    return {
        "headline": f"DIRECTION SHIFT: {ref} is moving {speed}",
        "body": (
            f"  This area is shifting toward new territory.\n"
            f"  \u2192 IMPLICATION: Plan for convergence. Follow where this is going for {lens.positive_outcome}."
        ),
    }


@_register_translator("phase_transition")
def _translate_phase_transition(insight, lens, ctx):
    return {
        "headline": "REGIME CHANGE: structure shifted abruptly",
        "body": (
            f"  Old patterns may no longer apply.\n"
            f"  \u2192 ACTION: Reassess all models. Previous assumptions about {lens.value_frame} may be invalid."
        ),
    }


@_register_translator("exploration_dynamics")
def _translate_exploration(insight, lens, ctx):
    return {
        "headline": "LANDSCAPE COVERAGE",
        "body": (
            f"  {insight.description}\n"
            f"  \u2192 Unexplored regions may contain {lens.opportunity_frame}s or hidden {lens.risk_frame}."
        ),
    }


@_register_translator("model_quality")
def _translate_model_quality(insight, lens, ctx):
    return {
        "headline": "ANALYSIS CONFIDENCE ASSESSMENT",
        "body": (
            f"  {insight.description}\n"
            f"  \u2192 Higher quality = more trustworthy {lens.value_frame} insights."
        ),
    }


@_register_translator("insight_reliability")
def _translate_reliability(insight, lens, ctx):
    return {
        "headline": "FINDING RELIABILITY",
        "body": (
            f"  {insight.description}\n"
            f"  \u2192 Act on consensus findings. Hold fragile ones for validation."
        ),
    }


@_register_translator("corpus_diagnosis")
def _translate_corpus(insight, lens, ctx):
    return {
        "headline": "DATA COMPLETENESS",
        "body": (
            f"  {insight.description}\n"
            f"  \u2192 ACTION: Fill identified gaps before next {lens.value_frame} analysis cycle."
        ),
    }


@_register_translator("strategic")
def _translate_strategic(insight, lens, ctx):
    return {
        "headline": insight.title.upper(),
        "body": f"  {insight.description}",
    }


@_register_translator("relationship")
def _translate_relationship(insight, lens, ctx):
    return {
        "headline": f"CONNECTION: {insight.title}",
        "body": (
            f"  {insight.description}\n"
            f"  \u2192 ACTION: {ctx.actor.title()} should leverage this connection for {lens.positive_outcome}."
        ),
    }


def _smart_translate(insight, lens: LensDefinition, ctx: OracleContext) -> Optional[dict]:
    """Use registered smart translator for this category."""
    translator = _SMART_TRANSLATORS.get(insight.category)
    if translator:
        return translator(insight, lens, ctx)
    return None


def _fallback_translate(insight, lens: LensDefinition, ctx: OracleContext) -> dict:
    """Universal fallback — works for ANY category/lens combination."""
    urgency = SEVERITY_URGENCY.get(insight.severity, SEVERITY_URGENCY["medium"])
    cat_display = insight.category.replace("_", " ").upper()

    return {
        "headline": f"{urgency['prefix']} {cat_display}: {insight.title}",
        "body": (
            f"  {insight.description}\n"
            f"  \u2192 ACTION: {ctx.actor.title()} {urgency['imperative']} "
            f"{lens.action_verb} {urgency['timeframe']}."
        ),
    }


# ── Built-in Lens Library ────────────────────────────────────────────

def _build_default_lenses() -> dict[str, LensDefinition]:
    """Build the standard lens library."""
    lenses = {}

    # ── Core 7 lenses (backward compatible) ──

    lenses["revenue"] = LensDefinition(
        name="revenue",
        title="REVENUE INTELLIGENCE",
        audience="Sales, BD, Revenue Leadership",
        description="Revenue growth, cross-sell, upsell, and at-risk revenue signals",
        signal_filters=["opportunity", "cluster", "knowledge_gap", "risk", "trend"],
        action_verb="pursue",
        value_frame="revenue",
        risk_frame="churn risk",
        opportunity_frame="cross-sell opportunity",
        positive_outcome="revenue growth",
        negative_outcome="churn",
    )

    lenses["operational"] = LensDefinition(
        name="operational",
        title="OPERATIONAL INTELLIGENCE",
        audience="Operations, Program Management, COO",
        description="Bottlenecks, dependencies, critical paths, and fragility",
        signal_filters=["information_bottleneck", "intervention", "feedback_loop", "causal_chain", "hierarchy"],
        action_verb="optimize",
        value_frame="operational efficiency",
        risk_frame="operational failure",
        opportunity_frame="efficiency gain",
        positive_outcome="throughput improvement",
        negative_outcome="service disruption",
    )

    lenses["risk"] = LensDefinition(
        name="risk",
        title="RISK INTELLIGENCE",
        audience="Risk Management, Compliance, CISO",
        description="Threats, blind spots, anomalies, structural fragility, regime changes",
        signal_filters=["risk", "blind_spot", "anomaly", "topological_stability", "phase_transition", "coverage_gap"],
        action_verb="mitigate",
        value_frame="risk posture",
        risk_frame="exposure",
        opportunity_frame="risk reduction",
        positive_outcome="risk reduction",
        negative_outcome="uncontrolled exposure",
    )

    lenses["strategic"] = LensDefinition(
        name="strategic",
        title="STRATEGIC INTELLIGENCE",
        audience="C-Suite, Board, Strategy",
        description="Portfolio coherence, market direction, structural transformation",
        signal_filters=["strategic", "topic_drift", "phase_transition", "hierarchy", "model_quality"],
        action_verb="position for",
        value_frame="strategic position",
        risk_frame="strategic exposure",
        opportunity_frame="strategic advantage",
        positive_outcome="market leadership",
        negative_outcome="strategic misalignment",
    )

    lenses["customer"] = LensDefinition(
        name="customer",
        title="CUSTOMER INTELLIGENCE",
        audience="Product, Customer Success, Marketing",
        description="Customer personas, behavior shifts, at-risk customers, cross-adoption",
        signal_filters=["cluster", "risk", "opportunity", "trend", "anomaly", "topic_drift"],
        action_verb="engage",
        value_frame="customer lifecycle value",
        risk_frame="churn risk",
        opportunity_frame="cross-adoption potential",
        entity_frame="customer persona",
        positive_outcome="customer retention and growth",
        negative_outcome="customer churn",
    )

    lenses["competitive"] = LensDefinition(
        name="competitive",
        title="COMPETITIVE / MARKET INTELLIGENCE",
        audience="Strategy, Competitive Intelligence",
        description="Market whitespace, defensible positions, competitive vulnerability",
        signal_filters=["knowledge_gap", "hierarchy", "topological_stability", "coverage_gap"],
        action_verb="evaluate",
        value_frame="market position",
        risk_frame="competitive threat",
        opportunity_frame="first-mover advantage",
        positive_outcome="market share gain",
        negative_outcome="competitive displacement",
    )

    lenses["data"] = LensDefinition(
        name="data",
        title="DATA & CONFIDENCE INTELLIGENCE",
        audience="Data Teams, Analysts, CDO",
        description="Data quality, coverage gaps, analysis reliability, collection priorities",
        signal_filters=["confidence_field", "coverage_gap", "corpus_diagnosis", "insight_reliability", "model_quality", "exploration_dynamics"],
        action_verb="prioritize",
        value_frame="data quality",
        risk_frame="data gap",
        opportunity_frame="data enrichment opportunity",
        positive_outcome="analytical confidence",
        negative_outcome="unreliable analysis",
    )

    # ── Extended lenses (industry/function) ──

    lenses["compliance"] = LensDefinition(
        name="compliance",
        title="COMPLIANCE & REGULATORY INTELLIGENCE",
        audience="Legal, Compliance, GRC",
        description="Regulatory exposure, compliance gaps, anomalous behavior, policy drift",
        signal_types=["threat", "coverage", "outlier", "deviation", "regime"],
        action_verb="remediate",
        value_frame="compliance posture",
        risk_frame="regulatory exposure",
        opportunity_frame="compliance advantage",
        positive_outcome="regulatory compliance",
        negative_outcome="regulatory violation",
        priority_keywords=["compliance", "regulation", "policy", "audit", "standard", "requirement"],
    )

    lenses["innovation"] = LensDefinition(
        name="innovation",
        title="INNOVATION INTELLIGENCE",
        audience="R&D, Product Strategy, CTO",
        description="Emerging patterns, anomalous signals, whitespace opportunities, technology convergence",
        signal_types=["novelty", "void", "whitespace", "dynamics", "evolution"],
        action_verb="explore",
        value_frame="innovation pipeline",
        risk_frame="disruption threat",
        opportunity_frame="innovation opportunity",
        positive_outcome="breakthrough discovery",
        negative_outcome="innovation stagnation",
        priority_keywords=["novel", "emerging", "new", "prototype", "patent", "research"],
    )

    lenses["talent"] = LensDefinition(
        name="talent",
        title="TALENT & WORKFORCE INTELLIGENCE",
        audience="HR, People Ops, CHRO",
        description="Workforce patterns, skill gaps, organizational structure, retention risk",
        signal_types=["segmentation", "void", "instability", "evolution", "structure"],
        action_verb="invest in",
        value_frame="talent capacity",
        risk_frame="attrition risk",
        opportunity_frame="talent development opportunity",
        entity_frame="workforce segment",
        positive_outcome="talent retention",
        negative_outcome="talent drain",
        priority_keywords=["skill", "role", "team", "hire", "retention", "performance", "talent"],
    )

    lenses["supply_chain"] = LensDefinition(
        name="supply_chain",
        title="SUPPLY CHAIN INTELLIGENCE",
        audience="Supply Chain, Procurement, COO",
        description="Supply chain bottlenecks, dependencies, fragility, alternative routes",
        signal_types=["flow", "dependency", "fragility", "operational", "cycle"],
        action_verb="secure",
        value_frame="supply chain resilience",
        risk_frame="supply disruption",
        opportunity_frame="supply optimization",
        positive_outcome="supply continuity",
        negative_outcome="supply chain failure",
        priority_keywords=["supply", "vendor", "procurement", "logistics", "inventory", "delivery"],
    )

    lenses["financial"] = LensDefinition(
        name="financial",
        title="FINANCIAL INTELLIGENCE",
        audience="CFO, Finance, Treasury",
        description="Financial patterns, exposure concentration, cost structure, value flow",
        signal_types=["flow", "structure", "threat", "macro", "portfolio"],
        action_verb="optimize",
        value_frame="financial performance",
        risk_frame="financial exposure",
        opportunity_frame="financial upside",
        positive_outcome="margin improvement",
        negative_outcome="value erosion",
        priority_keywords=["cost", "revenue", "margin", "budget", "investment", "return", "roi"],
    )

    lenses["security"] = LensDefinition(
        name="security",
        title="SECURITY INTELLIGENCE",
        audience="CISO, Security Operations, IT",
        description="Threat detection, vulnerability patterns, anomalous access, coverage gaps",
        signal_types=["threat", "outlier", "deviation", "coverage", "fragility"],
        action_verb="remediate",
        value_frame="security posture",
        risk_frame="threat",
        opportunity_frame="hardening opportunity",
        positive_outcome="threat elimination",
        negative_outcome="breach",
        priority_keywords=["threat", "vulnerability", "access", "attack", "breach", "patch"],
    )

    lenses["product"] = LensDefinition(
        name="product",
        title="PRODUCT INTELLIGENCE",
        audience="Product Management, Design, Engineering",
        description="Feature adoption, user behavior patterns, product gaps, convergence signals",
        signal_types=["segmentation", "synergy", "void", "dynamics", "identity"],
        action_verb="build",
        value_frame="product-market fit",
        risk_frame="adoption risk",
        opportunity_frame="feature opportunity",
        entity_frame="user segment",
        positive_outcome="adoption growth",
        negative_outcome="feature abandonment",
        priority_keywords=["feature", "user", "adoption", "usage", "engagement", "friction"],
    )

    lenses["marketing"] = LensDefinition(
        name="marketing",
        title="MARKETING INTELLIGENCE",
        audience="CMO, Marketing, Growth",
        description="Audience segmentation, message resonance, channel effectiveness, positioning",
        signal_types=["segmentation", "identity", "synergy", "dynamics", "void"],
        action_verb="target",
        value_frame="marketing ROI",
        risk_frame="message fatigue",
        opportunity_frame="audience opportunity",
        entity_frame="audience segment",
        positive_outcome="conversion uplift",
        negative_outcome="reach decline",
        priority_keywords=["campaign", "audience", "channel", "conversion", "brand", "message"],
    )

    lenses["partnership"] = LensDefinition(
        name="partnership",
        title="PARTNERSHIP & ALLIANCE INTELLIGENCE",
        audience="BD, Partnerships, Strategy",
        description="Partnership synergies, ecosystem gaps, complementary capabilities",
        signal_types=["synergy", "connection", "void", "whitespace", "growth"],
        action_verb="pursue",
        value_frame="partnership value",
        risk_frame="dependency risk",
        opportunity_frame="alliance opportunity",
        positive_outcome="ecosystem expansion",
        negative_outcome="partner loss",
        priority_keywords=["partner", "alliance", "ecosystem", "integration", "collaboration"],
    )

    lenses["quality"] = LensDefinition(
        name="quality",
        title="QUALITY ASSURANCE INTELLIGENCE",
        audience="QA, Engineering, Product",
        description="Defect patterns, quality coverage, regression risk, process fragility",
        signal_types=["outlier", "coverage", "fragility", "deviation", "operational"],
        action_verb="fix",
        value_frame="quality score",
        risk_frame="defect risk",
        opportunity_frame="quality improvement",
        positive_outcome="zero-defect operation",
        negative_outcome="quality regression",
        priority_keywords=["defect", "bug", "test", "quality", "regression", "coverage"],
    )

    lenses["sustainability"] = LensDefinition(
        name="sustainability",
        title="SUSTAINABILITY & ESG INTELLIGENCE",
        audience="CSO, ESG Team, Board",
        description="Environmental impact, social signals, governance patterns, compliance gaps",
        signal_types=["threat", "coverage", "dynamics", "void", "macro"],
        action_verb="address",
        value_frame="ESG score",
        risk_frame="sustainability risk",
        opportunity_frame="impact opportunity",
        positive_outcome="ESG improvement",
        negative_outcome="ESG deterioration",
        priority_keywords=["carbon", "emission", "waste", "diversity", "governance", "esg", "sustainability"],
    )

    lenses["clinical"] = LensDefinition(
        name="clinical",
        title="CLINICAL INTELLIGENCE",
        audience="CMO, Clinical Operations, Research",
        description="Patient patterns, treatment pathways, outcome signals, care gaps",
        signal_types=["segmentation", "flow", "void", "outlier", "causality"],
        action_verb="investigate",
        value_frame="clinical outcome",
        risk_frame="patient risk",
        opportunity_frame="care improvement",
        entity_frame="patient cohort",
        positive_outcome="outcome improvement",
        negative_outcome="adverse event",
        priority_keywords=["patient", "treatment", "outcome", "clinical", "diagnosis", "care"],
    )

    lenses["pricing"] = LensDefinition(
        name="pricing",
        title="PRICING INTELLIGENCE",
        audience="Pricing, Revenue Management, Finance",
        description="Price sensitivity, segment willingness, competitive pricing, elasticity signals",
        signal_types=["segmentation", "dynamics", "boundary", "synergy", "structure"],
        action_verb="adjust",
        value_frame="price optimization",
        risk_frame="price erosion",
        opportunity_frame="pricing upside",
        positive_outcome="margin expansion",
        negative_outcome="price war",
        priority_keywords=["price", "cost", "margin", "discount", "premium", "elastic"],
    )

    lenses["fraud"] = LensDefinition(
        name="fraud",
        title="FRAUD & ANOMALY INTELLIGENCE",
        audience="Fraud Team, Compliance, Risk",
        description="Anomalous patterns, unusual connections, network irregularities",
        signal_types=["outlier", "deviation", "novelty", "cycle", "boundary"],
        action_verb="investigate",
        value_frame="fraud prevention",
        risk_frame="fraud exposure",
        opportunity_frame="detection improvement",
        positive_outcome="fraud prevention",
        negative_outcome="undetected fraud",
        priority_keywords=["fraud", "suspicious", "unusual", "pattern", "anomaly", "irregular"],
    )

    lenses["geopolitical"] = LensDefinition(
        name="geopolitical",
        title="GEOPOLITICAL INTELLIGENCE",
        audience="Government Affairs, Strategy, Risk",
        description="Regional patterns, political dynamics, jurisdictional risk, sanctions exposure",
        signal_types=["macro", "dynamics", "regime", "threat", "boundary"],
        action_verb="monitor",
        value_frame="geopolitical position",
        risk_frame="geopolitical exposure",
        opportunity_frame="market entry",
        positive_outcome="strategic positioning",
        negative_outcome="sanctions exposure",
        priority_keywords=["region", "country", "sanction", "tariff", "political", "jurisdiction"],
    )

    lenses["brand"] = LensDefinition(
        name="brand",
        title="BRAND & REPUTATION INTELLIGENCE",
        audience="CMO, PR, Communications",
        description="Brand perception, sentiment patterns, reputation risk, messaging coherence",
        signal_types=["identity", "dynamics", "threat", "deviation", "segmentation"],
        action_verb="protect",
        value_frame="brand equity",
        risk_frame="reputation risk",
        opportunity_frame="brand building opportunity",
        positive_outcome="brand strengthening",
        negative_outcome="reputation damage",
        priority_keywords=["brand", "reputation", "sentiment", "perception", "media", "pr"],
    )

    lenses["infrastructure"] = LensDefinition(
        name="infrastructure",
        title="INFRASTRUCTURE INTELLIGENCE",
        audience="CTO, DevOps, Platform Engineering",
        description="System dependencies, capacity patterns, single points of failure, scaling signals",
        signal_types=["dependency", "fragility", "flow", "operational", "coverage"],
        action_verb="reinforce",
        value_frame="infrastructure resilience",
        risk_frame="system failure",
        opportunity_frame="scaling opportunity",
        positive_outcome="uptime improvement",
        negative_outcome="system outage",
        priority_keywords=["infrastructure", "system", "capacity", "uptime", "latency", "scale"],
    )

    lenses["knowledge_management"] = LensDefinition(
        name="knowledge_management",
        title="KNOWLEDGE MANAGEMENT INTELLIGENCE",
        audience="Knowledge Management, L&D, CKO",
        description="Knowledge gaps, expertise concentration, learning pathways, institutional knowledge risk",
        signal_types=["void", "coverage", "flow", "structure", "fragility"],
        action_verb="document",
        value_frame="knowledge capital",
        risk_frame="knowledge loss",
        opportunity_frame="learning opportunity",
        positive_outcome="knowledge accessibility",
        negative_outcome="knowledge silos",
        priority_keywords=["knowledge", "documentation", "training", "expertise", "learning", "wiki"],
    )

    lenses["m_and_a"] = LensDefinition(
        name="m_and_a",
        title="M&A / INTEGRATION INTELLIGENCE",
        audience="Corporate Development, Strategy, Integration PMO",
        description="Merger synergies, integration risks, cultural alignment, portfolio fit",
        signal_types=["synergy", "structure", "void", "identity", "connection"],
        action_verb="evaluate",
        value_frame="synergy realization",
        risk_frame="integration risk",
        opportunity_frame="acquisition target",
        positive_outcome="value accretion",
        negative_outcome="integration failure",
        priority_keywords=["merger", "acquisition", "integration", "synergy", "divestiture", "portfolio"],
    )

    lenses["investor_relations"] = LensDefinition(
        name="investor_relations",
        title="INVESTOR RELATIONS INTELLIGENCE",
        audience="IR, CFO, CEO",
        description="Narrative coherence, performance signals, risk disclosure, growth story",
        signal_types=["macro", "strategic", "portfolio", "dynamics", "confidence"],
        action_verb="communicate",
        value_frame="shareholder value",
        risk_frame="narrative risk",
        opportunity_frame="growth narrative",
        positive_outcome="investor confidence",
        negative_outcome="valuation compression",
        priority_keywords=["investor", "shareholder", "valuation", "guidance", "earnings", "growth"],
    )

    lenses["vendor"] = LensDefinition(
        name="vendor",
        title="VENDOR & PROCUREMENT INTELLIGENCE",
        audience="Procurement, Vendor Management",
        description="Vendor concentration, alternative sources, contract risk, cost optimization",
        signal_types=["dependency", "fragility", "void", "flow", "boundary"],
        action_verb="diversify",
        value_frame="procurement value",
        risk_frame="vendor concentration risk",
        opportunity_frame="cost reduction opportunity",
        positive_outcome="vendor diversification",
        negative_outcome="vendor lock-in",
        priority_keywords=["vendor", "supplier", "contract", "procurement", "cost", "sourcing"],
    )

    lenses["cx"] = LensDefinition(
        name="cx",
        title="CUSTOMER EXPERIENCE INTELLIGENCE",
        audience="CX, Support, Product",
        description="Experience friction, satisfaction patterns, journey gaps, delight opportunities",
        signal_types=["segmentation", "void", "outlier", "dynamics", "flow"],
        action_verb="improve",
        value_frame="customer satisfaction",
        risk_frame="experience friction",
        opportunity_frame="delight opportunity",
        entity_frame="experience touchpoint",
        positive_outcome="NPS improvement",
        negative_outcome="satisfaction decline",
        priority_keywords=["experience", "satisfaction", "nps", "support", "friction", "journey"],
    )

    lenses["legal"] = LensDefinition(
        name="legal",
        title="LEGAL INTELLIGENCE",
        audience="General Counsel, Legal Ops",
        description="Legal exposure, contract patterns, dispute signals, compliance gaps",
        signal_types=["threat", "coverage", "outlier", "boundary", "regime"],
        action_verb="address",
        value_frame="legal risk reduction",
        risk_frame="legal exposure",
        opportunity_frame="risk mitigation",
        positive_outcome="liability reduction",
        negative_outcome="litigation exposure",
        priority_keywords=["legal", "contract", "liability", "dispute", "clause", "compliance"],
    )

    lenses["ecommerce"] = LensDefinition(
        name="ecommerce",
        title="E-COMMERCE INTELLIGENCE",
        audience="E-Commerce, Digital, Marketplace",
        description="Purchase patterns, cart behavior, conversion signals, catalog gaps",
        signal_types=["segmentation", "flow", "void", "dynamics", "synergy"],
        action_verb="optimize",
        value_frame="conversion rate",
        risk_frame="cart abandonment",
        opportunity_frame="conversion opportunity",
        entity_frame="buyer segment",
        positive_outcome="GMV growth",
        negative_outcome="conversion decline",
        priority_keywords=["cart", "purchase", "conversion", "catalog", "checkout", "listing"],
    )

    lenses["real_estate"] = LensDefinition(
        name="real_estate",
        title="REAL ESTATE / LOCATION INTELLIGENCE",
        audience="Real Estate, Facilities, Expansion",
        description="Location patterns, market gaps, site performance, expansion signals",
        signal_types=["segmentation", "void", "dynamics", "structure", "boundary"],
        action_verb="evaluate",
        value_frame="location performance",
        risk_frame="market saturation",
        opportunity_frame="expansion opportunity",
        entity_frame="location cluster",
        positive_outcome="portfolio optimization",
        negative_outcome="underperforming location",
        priority_keywords=["location", "site", "market", "lease", "property", "expansion"],
    )

    lenses["media"] = LensDefinition(
        name="media",
        title="MEDIA & CONTENT INTELLIGENCE",
        audience="Content Strategy, Editorial, Media",
        description="Content performance, audience clustering, topic gaps, narrative flow",
        signal_types=["segmentation", "void", "flow", "dynamics", "identity"],
        action_verb="produce",
        value_frame="content performance",
        risk_frame="audience fatigue",
        opportunity_frame="content opportunity",
        entity_frame="content cluster",
        positive_outcome="engagement growth",
        negative_outcome="audience decline",
        priority_keywords=["content", "audience", "engagement", "article", "topic", "editorial"],
    )

    lenses["education"] = LensDefinition(
        name="education",
        title="EDUCATION & LEARNING INTELLIGENCE",
        audience="Academic Leadership, Curriculum, L&D",
        description="Learning patterns, curriculum gaps, student segmentation, outcome signals",
        signal_types=["segmentation", "void", "flow", "dynamics", "structure"],
        action_verb="develop",
        value_frame="learning outcome",
        risk_frame="achievement gap",
        opportunity_frame="curriculum enhancement",
        entity_frame="learner cohort",
        positive_outcome="outcome improvement",
        negative_outcome="learning gap",
        priority_keywords=["student", "curriculum", "course", "learning", "assessment", "outcome"],
    )

    lenses["defense"] = LensDefinition(
        name="defense",
        title="DEFENSE & NATIONAL SECURITY INTELLIGENCE",
        audience="Intelligence Analysts, Defense Leadership",
        description="Threat patterns, capability gaps, force structure, strategic positioning",
        signal_types=["threat", "void", "structure", "dynamics", "regime"],
        action_verb="assess",
        value_frame="operational readiness",
        risk_frame="threat level",
        opportunity_frame="capability advantage",
        entity_frame="threat cluster",
        positive_outcome="readiness improvement",
        negative_outcome="capability gap",
        priority_keywords=["threat", "capability", "readiness", "force", "intelligence", "adversary"],
    )

    lenses["philanthropy"] = LensDefinition(
        name="philanthropy",
        title="IMPACT & PHILANTHROPY INTELLIGENCE",
        audience="Foundation, Impact Investing, CSR",
        description="Impact patterns, intervention effectiveness, need gaps, resource allocation",
        signal_types=["void", "flow", "segmentation", "dynamics", "coverage"],
        action_verb="fund",
        value_frame="social impact",
        risk_frame="impact dilution",
        opportunity_frame="high-impact opportunity",
        entity_frame="beneficiary segment",
        positive_outcome="impact maximization",
        negative_outcome="wasted resources",
        priority_keywords=["impact", "grant", "beneficiary", "intervention", "outcome", "need"],
    )

    lenses["climate"] = LensDefinition(
        name="climate",
        title="CLIMATE & ENVIRONMENTAL INTELLIGENCE",
        audience="Sustainability, Environmental Compliance",
        description="Environmental patterns, emission clusters, transition signals, adaptation gaps",
        signal_types=["dynamics", "threat", "void", "macro", "regime"],
        action_verb="reduce",
        value_frame="environmental performance",
        risk_frame="climate exposure",
        opportunity_frame="green transition opportunity",
        positive_outcome="emission reduction",
        negative_outcome="environmental damage",
        priority_keywords=["carbon", "emission", "climate", "energy", "renewable", "transition"],
    )

    lenses["research"] = LensDefinition(
        name="research",
        title="RESEARCH INTELLIGENCE",
        audience="R&D, Research Directors, PIs",
        description="Research frontiers, knowledge voids, collaboration networks, breakthrough signals",
        signal_types=["void", "novelty", "flow", "connection", "dynamics"],
        action_verb="investigate",
        value_frame="research impact",
        risk_frame="research dead-end",
        opportunity_frame="frontier opportunity",
        entity_frame="research domain",
        positive_outcome="discovery",
        negative_outcome="research stagnation",
        priority_keywords=["research", "hypothesis", "experiment", "finding", "publication", "citation"],
    )

    return lenses


# Singleton library
_DEFAULT_LENSES = _build_default_lenses()


# ── Scorecard rendering ──────────────────────────────────────────────

def _render_bar(value: float, max_val: float, width: int = 20) -> str:
    """Render a simple ASCII progress bar."""
    if max_val <= 0:
        return "\u2591" * width
    filled = int(min(1.0, value / max_val) * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _render_scorecard(kpis: dict, ctx: OracleContext) -> str:
    """Render the strategic scorecard."""
    lines = ["\u2500\u2500\u2500 STRATEGIC SCORECARD \u2500\u2500\u2500"]

    cs = kpis.get("clarity_score", 0)
    clarity_desc = "excellent" if cs > 70 else "good" if cs > 40 else "moderate" if cs > 20 else "low"
    lines.append(f"  Portfolio Clarity:    {cs:>6.1f}/100  {_render_bar(cs, 100)}  ({clarity_desc})")

    dv = kpis.get("drift_velocity", 0)
    drift_desc = "highly dynamic" if dv > 0.8 else "active" if dv > 0.4 else "stable" if dv > 0.1 else "static"
    lines.append(f"  Market Velocity:     {dv:>6.3f}      {_render_bar(dv, 1.0)}  ({drift_desc})")

    opp = kpis.get("opportunity_surface", 0)
    opp_desc = "significant" if opp > 50 else "moderate" if opp > 10 else "limited"
    lines.append(f"  Opportunity Surface: {opp:>6}      {_render_bar(opp, 200)}  ({opp_desc})")

    risk = kpis.get("risk_horizon", 0)
    risk_desc = "high" if risk > 30 else "moderate" if risk > 10 else "manageable"
    lines.append(f"  Risk Horizon:        {risk:>6}      {_render_bar(risk, 100)}  ({risk_desc})")

    return "\n".join(lines)


# ── Executive summary ────────────────────────────────────────────────

def _render_oracle_summary(insights: list, kpis: dict, ctx: OracleContext) -> str:
    """Generate executive summary for the Oracle briefing."""
    ent = ctx.entity_type
    actor = ctx.actor

    high_priority = [i for i in insights if i.severity == "high"]
    opportunities = [i for i in insights if i.category == "opportunity"]
    risks = [i for i in insights if i.category in ("risk", "blind_spot", "anomaly")]

    parts = []
    if high_priority:
        parts.append(
            f"{len(high_priority)} critical findings require immediate attention from {actor}."
        )
    if opportunities:
        parts.append(
            f"Identified {len(opportunities)} cross-{ent} opportunities "
            f"representing untapped {ctx.currency}."
        )
    if risks:
        high_risk_count = sum(1 for r in risks if r.severity == "high")
        parts.append(
            f"{len(risks)} risk items flagged \u2014 "
            f"including {high_risk_count} high-severity."
        )

    cs = kpis.get("clarity_score", 0)
    if cs < 30:
        parts.append(
            f"Portfolio clarity is LOW ({cs:.0f}/100) \u2014 "
            f"{ent} definitions need sharpening before {actor} can act effectively."
        )

    return " ".join(parts) if parts else f"Analysis complete across {ctx.domain} portfolio."


# ── Auto-lens Generation ─────────────────────────────────────────────

def _auto_generate_lenses_from_data(report, ctx: OracleContext) -> list[LensDefinition]:
    """Dynamically create lenses based on what the data actually contains.

    Examines the insight stream and generates custom lenses for:
    - Dominant topic clusters (per-cluster lens)
    - Entity-specific lenses (from extracted entities)
    - Anomaly-focused lens if many anomalies
    - Cross-cutting thematic lenses from entity co-occurrence
    """
    auto_lenses = []

    # Collect all unique categories present
    categories_present = set()
    entity_counts = {}
    cluster_insights = []

    for i in report.insights:
        categories_present.add(i.category)
        for e in i.entities:
            entity_counts[e] = entity_counts.get(e, 0) + 1
        if i.category == "cluster":
            cluster_insights.append(i)

    # Generate per-cluster lenses for large/important clusters
    for ci in cluster_insights:
        if ci.severity == "high":
            cluster_name_parts = ci.entities[:3] if ci.entities else ["cluster"]
            cluster_label = ", ".join(cluster_name_parts)
            safe_name = re.sub(r"[^a-z0-9_]", "_", cluster_label.lower())[:30]
            auto_lenses.append(LensDefinition(
                name=f"cluster_{safe_name}",
                title=f"{cluster_label.upper()} DEEP DIVE",
                audience=ctx.actor,
                description=f"All signals related to {cluster_label}",
                signal_filters=list(categories_present),
                priority_keywords=[e.lower() for e in ci.entities[:5]],
                priority_boost=0.5,
                action_verb="focus on",
                value_frame=f"{cluster_label} performance",
                positive_outcome=f"{cluster_label} growth",
                negative_outcome=f"{cluster_label} decline",
            ))

    # Generate entity-focused lenses for top entities
    top_entities = sorted(entity_counts.items(), key=lambda x: -x[1])[:5]
    for entity, count in top_entities:
        if count >= 3:
            safe_name = re.sub(r"[^a-z0-9_]", "_", entity.lower())[:30]
            auto_lenses.append(LensDefinition(
                name=f"entity_{safe_name}",
                title=f"{entity.upper()} INTELLIGENCE",
                audience=ctx.actor,
                description=f"All signals mentioning {entity}",
                signal_filters=list(categories_present),
                priority_keywords=[entity.lower()],
                priority_boost=1.0,
                action_verb="address",
                value_frame=f"{entity} impact",
                positive_outcome=f"{entity} optimization",
                negative_outcome=f"{entity} degradation",
            ))

    # Generate anomaly lens if many anomalies
    anomaly_count = sum(1 for i in report.insights if i.category == "anomaly")
    if anomaly_count >= 3:
        auto_lenses.append(LensDefinition(
            name="anomaly_focus",
            title="ANOMALY DEEP DIVE",
            audience="Investigation Team",
            description=f"{anomaly_count} anomalous signals detected requiring investigation",
            signal_filters=["anomaly", "blind_spot", "phase_transition"],
            action_verb="investigate",
            value_frame="anomaly resolution",
            risk_frame="unresolved anomaly",
            positive_outcome="root cause identification",
            negative_outcome="undetected issue",
        ))

    return auto_lenses


# ── Main Engine ──────────────────────────────────────────────────────

# Backward-compatible constants
ALL_LENSES = ["revenue", "operational", "risk", "strategic", "customer", "competitive", "data"]

LENS_CATEGORIES = {
    name: lens.signal_filters
    for name, lens in _DEFAULT_LENSES.items()
    if name in ALL_LENSES
}


class OracleIntelligenceEngine:
    """Universal Operationalizer: produces intelligence briefings for any lens,
    any domain, any stakeholder, from any insight stream.

    Ships with 30+ built-in lenses. Supports unlimited custom lenses.
    Can auto-generate lenses from data characteristics.
    """

    def __init__(self):
        self._lenses: dict[str, LensDefinition] = dict(_DEFAULT_LENSES)

    @property
    def available_lenses(self) -> list[str]:
        """List all registered lens names."""
        return list(self._lenses.keys())

    @property
    def core_lenses(self) -> list[str]:
        """The 7 core lenses (backward compatible)."""
        return list(ALL_LENSES)

    def get_lens(self, name: str) -> Optional[LensDefinition]:
        """Get a lens definition by name."""
        return self._lenses.get(name)

    def register_lens(self, lens: LensDefinition):
        """Register a custom lens. Overwrites if name already exists."""
        self._lenses[lens.name] = lens
        logger.info(f"Registered lens: {lens.name} ({lens.title})")

    def register_lenses(self, lenses: list[LensDefinition]):
        """Register multiple lenses at once."""
        for lens in lenses:
            self.register_lens(lens)

    def create_lens(self, **kwargs) -> LensDefinition:
        """Create and register a lens from keyword arguments."""
        lens = LensDefinition(**kwargs)
        self.register_lens(lens)
        return lens

    def auto_generate_lenses(self, report, ctx: OracleContext) -> list[LensDefinition]:
        """Generate and register data-driven lenses from report contents."""
        auto = _auto_generate_lenses_from_data(report, ctx)
        self.register_lenses(auto)
        return auto

    def render_briefing(
        self,
        report,
        context: OracleContext,
        lenses: Optional[list[str]] = None,
        include_auto: bool = False,
    ) -> str:
        """Generate full Oracle briefing with selected lenses.

        Args:
            report: InsightReport with insights, kpis, and metadata.
            context: OracleContext with business framing.
            lenses: Optional list of lens names. Defaults to core 7.
            include_auto: If True, also generate and include auto-lenses from data.
        """
        # Determine which lenses to render
        if lenses:
            active_lenses = lenses
        else:
            active_lenses = list(ALL_LENSES)

        # Auto-generate if requested
        if include_auto:
            auto = self.auto_generate_lenses(report, context)
            active_lenses.extend(l.name for l in auto)

        lines = []

        # Header
        lines.append("\u2554" + "\u2550" * 70 + "\u2557")
        lines.append(f"\u2551{'ORACLE INTELLIGENCE BRIEFING':^70}\u2551")
        lines.append(f"\u2551{context.domain.upper():^70}\u2551")
        lines.append("\u255a" + "\u2550" * 70 + "\u255d")
        lines.append("")
        lines.append(
            f"Portfolio: {report.num_documents} sources, "
            f"{report.num_chunks} data points, "
            f"{report.num_clusters} {context.entity_type}s"
        )
        lines.append(f"Active lenses: {len(active_lenses)} | "
                      f"Available: {len(self._lenses)} | "
                      f"Insight categories: {len(set(i.category for i in report.insights))}")
        lines.append("")

        # Strategic scorecard
        lines.append(_render_scorecard(report.kpis, context))
        lines.append("")

        # Executive summary
        lines.append("\u2500\u2500\u2500 EXECUTIVE SUMMARY \u2500\u2500\u2500")
        lines.append(_render_oracle_summary(report.insights, report.kpis, context))
        lines.append("")

        # Render each lens
        for lens_name in active_lenses:
            lens_output = self.render_lens(lens_name, report, context)
            if lens_output:
                lines.append(lens_output)
                lines.append("")

        # Provenance footer
        if report.lineage_graph:
            lines.append("\u2500\u2500\u2500 PROVENANCE \u2500\u2500\u2500")
            lines.append("Every finding is traceable to source documents via lineage IDs.")
            lines.append(f"  Total provenance nodes: {report.lineage_graph.num_nodes}")
            lines.append(f"  Chunk nodes: {len(report.lineage_graph.nodes_by_kind('chunk'))}")
            lines.append(f"  Insight nodes: {len(report.lineage_graph.nodes_by_kind('insight'))}")
            lines.append("")

        lines.append("\u2550" * 72)
        return "\n".join(lines)

    def render_lens(
        self,
        lens_name: str,
        report,
        context: OracleContext,
    ) -> Optional[str]:
        """Generate a single intelligence lens briefing.

        Works with ANY registered lens — built-in or custom.
        """
        lens = self._lenses.get(lens_name)
        if not lens:
            logger.warning(f"Unknown lens: {lens_name}. Available: {', '.join(self.available_lenses[:10])}...")
            return None

        # Filter insights through this lens
        relevant = [i for i in report.insights if lens.matches_insight(i)]
        if not relevant:
            return None

        lines = []
        stakeholder = context.stakeholders.get(lens_name, lens.audience)
        lines.append(f"\u2550\u2550\u2550 {lens.title} \u2550\u2550\u2550")
        if stakeholder:
            lines.append(f"Prepared for: {stakeholder}")
        if lens.description:
            lines.append(f"Scope: {lens.description}")
        lines.append("")

        # Sort by priority score
        relevant.sort(key=lambda x: -lens.priority_score(x))

        rendered_count = 0
        for insight in relevant:
            result = _universal_translate(insight, lens, context)
            if result is None:
                continue

            lines.append(result["headline"])
            lines.append(result["body"])

            # Add evidence if available
            if insight.structured_evidence:
                for ev in insight.structured_evidence[:2]:
                    excerpt = ev.text_excerpt[:150] + "..." if len(ev.text_excerpt) > 150 else ev.text_excerpt
                    source = ev.doc_title or ev.doc_id or "unknown"
                    lines.append(f"  Evidence: \"{excerpt}\" [{source}]")

            lines.append(f"  Confidence: {insight.confidence:.0%}")
            lines.append("")
            rendered_count += 1

        if rendered_count == 0:
            return None

        return "\n".join(lines)

    def list_lenses_for_domain(self, domain: str) -> list[str]:
        """Suggest relevant lenses for a given domain."""
        domain_lens_map = {
            "retail": ["revenue", "customer", "ecommerce", "pricing", "supply_chain", "marketing", "competitive"],
            "healthcare": ["clinical", "compliance", "risk", "quality", "data", "research", "operational"],
            "fintech": ["revenue", "fraud", "compliance", "risk", "customer", "security", "pricing"],
            "saas": ["revenue", "customer", "product", "cx", "competitive", "data", "infrastructure"],
            "manufacturing": ["operational", "supply_chain", "quality", "sustainability", "infrastructure", "vendor"],
            "defense": ["defense", "risk", "security", "geopolitical", "infrastructure", "operational"],
            "education": ["education", "data", "quality", "innovation", "knowledge_management", "cx"],
            "media": ["media", "marketing", "customer", "brand", "competitive", "innovation"],
            "pharma": ["clinical", "research", "compliance", "supply_chain", "competitive", "innovation"],
            "government": ["compliance", "risk", "operational", "geopolitical", "security", "data"],
            "real_estate": ["real_estate", "financial", "risk", "competitive", "sustainability", "infrastructure"],
            "nonprofit": ["philanthropy", "data", "operational", "sustainability", "knowledge_management"],
            "consulting": ["strategic", "competitive", "knowledge_management", "talent", "partnership", "innovation"],
            "banking": ["financial", "risk", "compliance", "fraud", "customer", "security", "competitive"],
            "insurance": ["risk", "compliance", "fraud", "customer", "pricing", "financial", "data"],
            "energy": ["climate", "sustainability", "infrastructure", "supply_chain", "risk", "compliance"],
            "logistics": ["supply_chain", "operational", "infrastructure", "vendor", "risk", "data"],
            "legal": ["legal", "compliance", "risk", "knowledge_management", "data", "operational"],
        }
        return domain_lens_map.get(domain.lower(), list(ALL_LENSES))

    def render_lens_catalog(self) -> str:
        """Render a catalog of all available lenses."""
        lines = [
            "\u2550\u2550\u2550 AVAILABLE INTELLIGENCE LENSES \u2550\u2550\u2550",
            f"Total: {len(self._lenses)} lenses",
            "",
        ]

        # Core lenses first
        lines.append("CORE LENSES:")
        for name in ALL_LENSES:
            lens = self._lenses[name]
            lines.append(f"  {name:<20} {lens.title}")
            lines.append(f"  {'':20} Audience: {lens.audience}")
            lines.append("")

        # Extended lenses
        extended = [n for n in sorted(self._lenses.keys()) if n not in ALL_LENSES]
        if extended:
            lines.append("EXTENDED LENSES:")
            for name in extended:
                lens = self._lenses[name]
                lines.append(f"  {name:<20} {lens.title}")
                lines.append(f"  {'':20} Audience: {lens.audience}")
                lines.append("")

        return "\n".join(lines)

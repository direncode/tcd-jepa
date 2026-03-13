"""Oracle Operational Intelligence: multi-lens intelligence briefings from TCD-JEPA insights.

Transforms the unified stream of 23 technical insight categories into 7 operational
intelligence lenses, each producing a dedicated briefing section with concrete actions
for a specific stakeholder audience.

The 7 Intelligence Lenses:
1. REVENUE       — Sales, BD, Revenue leadership
2. OPERATIONAL   — Operations, Program Management, COO
3. RISK          — Risk Management, Compliance, CISO
4. STRATEGIC     — C-suite, Board, Strategy
5. CUSTOMER      — Product, Customer Success, Marketing
6. COMPETITIVE   — Strategy, Competitive Intelligence
7. DATA          — Data teams, Analysts, CDO

Usage:
    from tcd_jepa.manifold.oracle_intelligence import OracleIntelligenceEngine, OracleContext

    ctx = OracleContext(domain="retail", entity_type="customer segment",
                        actor="sales team", resource_noun="budget")
    engine = OracleIntelligenceEngine()
    briefing = engine.render_briefing(report, ctx)
    print(briefing)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("tcd_jepa.oracle_intelligence")

# ── Lens definitions ─────────────────────────────────────────────────

ALL_LENSES = [
    "revenue", "operational", "risk", "strategic",
    "customer", "competitive", "data",
]

# Which insight categories feed each lens
LENS_CATEGORIES = {
    "revenue": [
        "opportunity", "cluster", "knowledge_gap", "risk", "trend",
    ],
    "operational": [
        "information_bottleneck", "intervention", "feedback_loop",
        "causal_chain", "hierarchy",
    ],
    "risk": [
        "risk", "blind_spot", "anomaly", "topological_stability",
        "phase_transition", "coverage_gap",
    ],
    "strategic": [
        "strategic", "topic_drift", "phase_transition", "hierarchy",
        "model_quality",
    ],
    "customer": [
        "cluster", "risk", "opportunity", "trend", "anomaly", "topic_drift",
    ],
    "competitive": [
        "knowledge_gap", "hierarchy", "topological_stability", "coverage_gap",
    ],
    "data": [
        "confidence_field", "coverage_gap", "corpus_diagnosis",
        "insight_reliability", "model_quality", "exploration_dynamics",
    ],
}

LENS_TITLES = {
    "revenue": "REVENUE INTELLIGENCE",
    "operational": "OPERATIONAL INTELLIGENCE",
    "risk": "RISK INTELLIGENCE",
    "strategic": "STRATEGIC INTELLIGENCE",
    "customer": "CUSTOMER INTELLIGENCE",
    "competitive": "COMPETITIVE / MARKET INTELLIGENCE",
    "data": "DATA & CONFIDENCE INTELLIGENCE",
}

LENS_AUDIENCES = {
    "revenue": "Sales, BD, Revenue Leadership",
    "operational": "Operations, Program Management, COO",
    "risk": "Risk Management, Compliance, CISO",
    "strategic": "C-Suite, Board, Strategy",
    "customer": "Product, Customer Success, Marketing",
    "competitive": "Strategy, Competitive Intelligence",
    "data": "Data Teams, Analysts, CDO",
}


# ── Context ──────────────────────────────────────────────────────────

@dataclass
class OracleContext:
    """Operational context that shapes how intelligence is delivered.

    Without this, the Oracle uses generic business language.
    With it, insights reference actual business terms.
    """
    domain: str = "business"
    entity_type: str = "segment"
    actor: str = "your team"
    resource_noun: str = "resources"
    currency: str = "value"
    cluster_labels: dict = field(default_factory=dict)
    stakeholders: dict = field(default_factory=dict)  # {lens_name: "VP Sales"}


# ── Translators (one per insight category) ───────────────────────────

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


def _translate_for_lens(insight, lens: str, ctx: OracleContext) -> dict:
    """Translate a single insight for a specific lens.

    Returns a dict with 'headline' and 'body' keys containing the
    business-language version of the insight. Returns None if the
    insight doesn't meaningfully contribute to this lens.
    """
    cat = insight.category
    sev = insight.severity
    ent = ctx.entity_type
    actor = ctx.actor
    resource = ctx.resource_noun

    # ── REVENUE LENS ──────────────────────────────────────────────
    if lens == "revenue":
        if cat == "opportunity":
            clusters = re.findall(r"clusters?\s*(\d+)", insight.title, re.IGNORECASE)
            c1 = _cluster_name(f"cluster {clusters[0]}", ctx) if clusters else "area A"
            c2 = _cluster_name(f"cluster {clusters[1]}", ctx) if len(clusters) > 1 else "area B"
            count = _extract_number(insight.title, r"\((\d+)\s*connections?\)")
            return {
                "headline": f"CROSS-SELL OPPORTUNITY: {c1} \u2194 {c2}",
                "body": (
                    f"  {count or 'Multiple'} connections found between {c1} and {c2}.\n"
                    f"  These {ent}s have natural affinity but are treated separately.\n"
                    f"  \u2192 ACTION: {actor.title()} should target {c1} with {c2} offerings."
                ),
            }
        elif cat == "cluster":
            entities_str = ", ".join(insight.entities[:5]) if insight.entities else ent
            count = _extract_number(insight.description, r"(\d+)\s*semantic chunks")
            pct = _extract_number(insight.description, r"\((\d+(?:\.\d+)?)%")
            coherence = _extract_number(insight.description, r"coherence:\s*(\d+(?:\.\d+)?)")
            strength = ""
            if coherence:
                c = float(coherence)
                strength = "strongly cohesive" if c > 0.7 else "moderately cohesive" if c > 0.4 else "loosely defined"
            return {
                "headline": f"{'HIGH-VALUE' if sev == 'high' else 'NOTABLE'} {ent.upper()}: {entities_str}",
                "body": (
                    f"  {count or '?'} items ({pct or '?'}% of portfolio)"
                    f"{f', {strength}' if strength else ''}.\n"
                    f"  \u2192 ACTION: Ensure {actor} has dedicated coverage for this {ent}."
                ),
            }
        elif cat == "knowledge_gap":
            clusters = re.findall(r"cluster\s*(\d+)", insight.title, re.IGNORECASE)
            c1 = _cluster_name(f"cluster {clusters[0]}", ctx) if clusters else "area A"
            c2 = _cluster_name(f"cluster {clusters[1]}", ctx) if len(clusters) > 1 else "area B"
            return {
                "headline": f"WHITESPACE: No coverage between {c1} and {c2}",
                "body": (
                    f"  Nobody in your portfolio bridges these two areas.\n"
                    f"  \u2192 ACTION: This is unaddressed demand. "
                    f"Evaluate whether to allocate {resource} here."
                ),
            }
        elif cat == "risk":
            return {
                "headline": f"AT-RISK {currency_label(ctx)}: unstable {ent} classification",
                "body": (
                    f"  This item sits between {ent}s \u2014 it could go either way.\n"
                    f"  \u2192 ACTION: Proactive retention engagement within 30 days."
                ),
            }
        elif cat == "trend":
            is_dynamic = "evolving" in insight.title.lower() or "rapidly" in insight.title.lower()
            ref = _cluster_name(insight.title, ctx)
            if is_dynamic:
                return {
                    "headline": f"GROWING DEMAND: {ref}",
                    "body": (
                        f"  This {ent} is evolving rapidly.\n"
                        f"  \u2192 ACTION: Allocate {resource} now to capitalize on momentum."
                    ),
                }
            else:
                return {
                    "headline": f"RELIABLE BASE: {ref}",
                    "body": (
                        f"  This {ent} is stable \u2014 dependable recurring {ctx.currency}.\n"
                        f"  \u2192 ACTION: Low-touch maintenance. Redirect {resource} to growth areas."
                    ),
                }

    # ── OPERATIONAL LENS ──────────────────────────────────────────
    elif lens == "operational":
        if cat == "information_bottleneck":
            chunk_ref = re.search(r"chunk\s*(\d+)", insight.title, re.IGNORECASE)
            idx = chunk_ref.group(1) if chunk_ref else "a critical item"
            pct = _extract_number(insight.description, r"(\d+(?:\.\d+)?)%")
            return {
                "headline": f"BOTTLENECK ALERT: item {idx} carries {pct or '?'}% of flow",
                "body": (
                    f"  A disproportionate share of information routes through this point.\n"
                    f"  \u2192 ACTION: Build redundancy. Single points of failure are unacceptable."
                ),
            }
        elif cat == "intervention":
            count = _extract_number(insight.title, r"(\d+)\s*hub")
            return {
                "headline": f"FRAGILITY: removing {count or 'key'} items fragments operations",
                "body": (
                    f"  These items hold the structure together.\n"
                    f"  \u2192 ACTION: Treat as critical infrastructure. Protect and reinforce."
                ),
            }
        elif cat == "feedback_loop":
            n = _extract_number(insight.title, r"(\d+)-node")
            return {
                "headline": f"CIRCULAR DEPENDENCY: {n or 'multi'}-way loop detected",
                "body": (
                    f"  Changes to one element cascade through the loop unpredictably.\n"
                    f"  \u2192 ACTION: Coordinate changes across all {n or 'involved'} elements jointly."
                ),
            }
        elif cat == "causal_chain":
            return {
                "headline": "CRITICAL PATH: primary value chain identified",
                "body": (
                    f"  Influence flows along this path \u2014 upstream disruptions cascade downstream.\n"
                    f"  \u2192 ACTION: Optimize and protect this sequence."
                ),
            }
        elif cat == "hierarchy":
            return {
                "headline": f"STRUCTURE: multi-level {ent} hierarchy discovered",
                "body": (
                    f"  Sub-{ent}s nest inside broader groups.\n"
                    f"  \u2192 ACTION: Set strategy at macro level, tactics at micro level."
                ),
            }

    # ── RISK LENS ─────────────────────────────────────────────────
    elif lens == "risk":
        if cat == "risk":
            return {
                "headline": f"EXPOSURE: unstable {ent} classification",
                "body": (
                    f"  Items in contested territory \u2014 classification is unstable.\n"
                    f"  \u2192 ACTION: Assess exposure and assign to clear {ent} or flag for review."
                ),
            }
        elif cat == "blind_spot":
            ref = _cluster_name(insight.title, ctx)
            return {
                "headline": f"UNKNOWN UNKNOWNS: insufficient data on {ref}",
                "body": (
                    f"  Cannot confidently assess risk in this area.\n"
                    f"  \u2192 ACTION: Prioritize data collection before exposure grows."
                ),
            }
        elif cat == "anomaly":
            z = _extract_number(insight.title, r"z-score:\s*([\d.]+)")
            return {
                "headline": f"ANOMALY: activity {z or '?'}\u03c3 outside normal ({sev} severity)",
                "body": (
                    f"  Doesn't fit established patterns.\n"
                    f"  \u2192 ACTION: Flag for compliance review. May indicate threat or emerging opportunity."
                ),
            }
        elif cat == "topological_stability":
            is_fragile = "fragile" in insight.title.lower()
            if is_fragile:
                return {
                    "headline": "STRUCTURAL FRAGILITY: position could collapse",
                    "body": (
                        f"  This structure is not robust \u2014 minor changes could eliminate it.\n"
                        f"  \u2192 ACTION: Do not over-invest until stabilized."
                    ),
                }
            else:
                return None  # Robust structures aren't risk-relevant
        elif cat == "phase_transition":
            return {
                "headline": "REGIME CHANGE DETECTED: structure shifted abruptly",
                "body": (
                    f"  Old patterns may no longer apply.\n"
                    f"  \u2192 ACTION: Reassess ALL risk models. Previous assumptions may be invalid."
                ),
            }
        elif cat == "coverage_gap":
            return {
                "headline": f"INTELLIGENCE GAP: not all {ent}s assessed for risk",
                "body": (
                    f"  Unassessed areas = unknown risk.\n"
                    f"  \u2192 ACTION: Complete risk assessment coverage before next review cycle."
                ),
            }

    # ── STRATEGIC LENS ────────────────────────────────────────────
    elif lens == "strategic":
        if cat == "strategic":
            # Pass through with minor cleanup
            return {
                "headline": insight.title.upper(),
                "body": f"  {insight.description}",
            }
        elif cat == "topic_drift":
            ref = _cluster_name(insight.title, ctx)
            ratio = _extract_number(insight.description, r"([\d.]+)x\s*corpus")
            return {
                "headline": f"MARKET DIRECTION: {ref} is shifting",
                "body": (
                    f"  Moving {'rapidly' if ratio and float(ratio) > 2 else 'steadily'} "
                    f"toward new territory.\n"
                    f"  \u2192 STRATEGIC IMPLICATION: Plan for convergence. "
                    f"Follow where this {ent} is going."
                ),
            }
        elif cat == "phase_transition":
            return {
                "headline": "STRUCTURAL TRANSFORMATION: market underwent phase transition",
                "body": (
                    f"  New structure is emerging.\n"
                    f"  \u2192 STRATEGIC IMPLICATION: Old playbooks may need rewriting."
                ),
            }
        elif cat == "hierarchy":
            return {
                "headline": f"STRATEGIC LAYERS: multi-level {ent} hierarchy",
                "body": (
                    f"  Fine-grained sub-{ent}s consolidate into macro-themes.\n"
                    f"  \u2192 STRATEGIC IMPLICATION: Organize portfolio at both levels."
                ),
            }
        elif cat == "model_quality":
            return {
                "headline": "CONFIDENCE ASSESSMENT",
                "body": (
                    f"  {insight.description}\n"
                    f"  \u2192 Consensus findings should drive decisions."
                ),
            }

    # ── CUSTOMER LENS ─────────────────────────────────────────────
    elif lens == "customer":
        if cat == "cluster":
            entities_str = ", ".join(insight.entities[:5]) if insight.entities else "key behaviors"
            count = _extract_number(insight.description, r"(\d+)\s*semantic chunks")
            coherence = _extract_number(insight.description, r"coherence:\s*(\d+(?:\.\d+)?)")
            coh_val = float(coherence) if coherence else 0
            label = "PERSONA" if coh_val > 0.5 else "LOOSE GROUP"
            behavior_desc = "Strongly defined behavior pattern." if coh_val > 0.7 else "Loosely defined — customers here behave variably."
            action_desc = "Build dedicated playbook for this persona." if coh_val > 0.5 else "Investigate what sub-segments exist within this group."
            return {
                "headline": f"{label}: {entities_str} ({count or '?'} items, {int(coh_val*100)}% coherence)",
                "body": f"  {behavior_desc}\n  \u2192 ACTION: {action_desc}",
            }
        elif cat == "risk":
            return {
                "headline": f"AT-RISK CUSTOMERS: between personas",
                "body": (
                    f"  These don't clearly belong to any persona.\n"
                    f"  \u2192 ACTION: Customer success should engage \u2014 these are churn candidates."
                ),
            }
        elif cat == "opportunity":
            clusters = re.findall(r"clusters?\s*(\d+)", insight.title, re.IGNORECASE)
            c1 = _cluster_name(f"cluster {clusters[0]}", ctx) if clusters else "group A"
            c2 = _cluster_name(f"cluster {clusters[1]}", ctx) if len(clusters) > 1 else "group B"
            return {
                "headline": f"CROSS-ADOPTION: {c1} and {c2} have similar needs",
                "body": (
                    f"  Natural affinity suggests bundled offering potential.\n"
                    f"  \u2192 ACTION: Product team should explore cross-{ent} packaging."
                ),
            }
        elif cat == "trend":
            is_dynamic = "evolving" in insight.title.lower() or "rapidly" in insight.title.lower()
            ref = _cluster_name(insight.title, ctx)
            if is_dynamic:
                return {
                    "headline": f"EVOLVING NEEDS: {ref}",
                    "body": (
                        f"  Customer needs in this {ent} are actively changing.\n"
                        f"  \u2192 ACTION: Update messaging and product positioning."
                    ),
                }
            return None  # Stable trends less relevant for customer lens
        elif cat == "anomaly":
            return {
                "headline": "OUTLIER CUSTOMER: doesn't match any persona",
                "body": (
                    f"  May be early adopter of a new use case.\n"
                    f"  \u2192 ACTION: Investigate \u2014 could signal emerging customer segment."
                ),
            }
        elif cat == "topic_drift":
            ref = _cluster_name(insight.title, ctx)
            return {
                "headline": f"BEHAVIOR SHIFT: {ref} is migrating",
                "body": (
                    f"  Customer behavior in this {ent} is shifting.\n"
                    f"  \u2192 ACTION: Update positioning to match where they're going."
                ),
            }

    # ── COMPETITIVE LENS ──────────────────────────────────────────
    elif lens == "competitive":
        if cat == "knowledge_gap":
            clusters = re.findall(r"cluster\s*(\d+)", insight.title, re.IGNORECASE)
            c1 = _cluster_name(f"cluster {clusters[0]}", ctx) if clusters else "area A"
            c2 = _cluster_name(f"cluster {clusters[1]}", ctx) if len(clusters) > 1 else "area B"
            return {
                "headline": f"WHITESPACE: unoccupied territory between {c1} and {c2}",
                "body": (
                    f"  No player covers this intersection.\n"
                    f"  \u2192 ACTION: First-mover opportunity. Evaluate entry cost vs. return."
                ),
            }
        elif cat == "hierarchy":
            return {
                "headline": f"MARKET STRUCTURE: multi-tier {ent} landscape",
                "body": (
                    f"  {insight.description}\n"
                    f"  \u2192 ACTION: Identify which tier offers best competitive positioning."
                ),
            }
        elif cat == "topological_stability":
            is_fragile = "fragile" in insight.title.lower()
            if is_fragile:
                return {
                    "headline": "VULNERABLE POSITION: structurally fragile",
                    "body": (
                        f"  Competitor entry could dissolve this position.\n"
                        f"  \u2192 ACTION: Shore up or prepare to exit."
                    ),
                }
            else:
                return {
                    "headline": "DEFENSIBLE POSITION: structurally robust",
                    "body": (
                        f"  Hard for competitors to dislodge.\n"
                        f"  \u2192 ACTION: Invest to strengthen this natural moat."
                    ),
                }
        elif cat == "coverage_gap":
            return {
                "headline": f"BLIND SPOT: competitors may already occupy uncovered areas",
                "body": (
                    f"  {insight.description}\n"
                    f"  \u2192 ACTION: Scout these areas before making strategic bets."
                ),
            }

    # ── DATA LENS ─────────────────────────────────────────────────
    elif lens == "data":
        if cat == "confidence_field":
            pct = _extract_number(insight.title, r"(\d+)%")
            return {
                "headline": f"DATA QUALITY: {pct or '?'}% of portfolio has confident analysis",
                "body": (
                    f"  \u2192 TRUST insights about well-covered {ent}s.\n"
                    f"  \u2192 CAUTION for poorly-covered {ent}s \u2014 need more data before acting."
                ),
            }
        elif cat == "coverage_gap":
            return {
                "headline": f"COLLECTION PRIORITY: data gaps in {ent}s",
                "body": (
                    f"  {insight.description}\n"
                    f"  \u2192 ACTION: Prioritize data collection for under-covered areas."
                ),
            }
        elif cat == "corpus_diagnosis":
            return {
                "headline": "DATA COMPLETENESS ASSESSMENT",
                "body": (
                    f"  {insight.description}\n"
                    f"  \u2192 ACTION: Fill identified gaps before next analysis cycle."
                ),
            }
        elif cat == "insight_reliability":
            return {
                "headline": "WHICH FINDINGS TO TRUST",
                "body": (
                    f"  {insight.description}\n"
                    f"  \u2192 Act on consensus findings. Hold fragile ones for validation."
                ),
            }
        elif cat == "model_quality":
            return {
                "headline": "ANALYSIS ENGINE QUALITY",
                "body": (
                    f"  {insight.description}\n"
                    f"  \u2192 Higher quality = more trustworthy insights."
                ),
            }
        elif cat == "exploration_dynamics":
            return {
                "headline": "LANDSCAPE COVERAGE",
                "body": (
                    f"  {insight.description}\n"
                    f"  \u2192 Unexplored regions may contain surprises."
                ),
            }

    return None


def currency_label(ctx: OracleContext) -> str:
    """Human-friendly currency label."""
    return ctx.currency.upper() if ctx.currency != "value" else "REVENUE"


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
    lines.append(f"  Opportunity Surface: {opp:>6}      {_render_bar(opp, 200)}  {'significant' if opp > 50 else 'moderate' if opp > 10 else 'limited'}")

    risk = kpis.get("risk_horizon", 0)
    lines.append(f"  Risk Horizon:        {risk:>6}      {_render_bar(risk, 100)}  {'high' if risk > 30 else 'moderate' if risk > 10 else 'manageable'}")

    return "\n".join(lines)


# ── Executive summary ────────────────────────────────────────────────

def _render_oracle_summary(insights: list, kpis: dict, ctx: OracleContext) -> str:
    """Generate a 2-3 sentence executive summary for the Oracle briefing."""
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
        parts.append(
            f"{len(risks)} risk items flagged \u2014 "
            f"including {sum(1 for r in risks if r.severity == 'high')} high-severity."
        )

    cs = kpis.get("clarity_score", 0)
    if cs < 30:
        parts.append(
            f"Portfolio clarity is LOW ({cs:.0f}/100) \u2014 "
            f"{ent} definitions need sharpening before {actor} can act effectively."
        )

    return " ".join(parts) if parts else f"Analysis complete across {ctx.domain} portfolio."


# ── Main engine ──────────────────────────────────────────────────────

class OracleIntelligenceEngine:
    """Produces multi-lens Oracle intelligence briefings from unified insight streams."""

    def render_briefing(
        self,
        report,
        context: OracleContext,
        lenses: Optional[list[str]] = None,
    ) -> str:
        """Generate full Oracle briefing with all active lenses.

        Args:
            report: InsightReport with insights, kpis, and metadata.
            context: OracleContext with business framing.
            lenses: Optional list of lens names. Defaults to all 7.
        """
        active_lenses = lenses or ALL_LENSES
        lines = []

        # Header
        lines.append("\u2554" + "\u2550" * 70 + "\u2557")
        lines.append(
            f"\u2551{'ORACLE INTELLIGENCE BRIEFING':^70}\u2551"
        )
        lines.append(
            f"\u2551{f'{context.domain.upper()}':^70}\u2551"
        )
        lines.append("\u255a" + "\u2550" * 70 + "\u255d")
        lines.append("")
        lines.append(
            f"Portfolio: {report.num_documents} sources, "
            f"{report.num_chunks} data points, "
            f"{report.num_clusters} {context.entity_type}s"
        )
        lines.append("")

        # Strategic scorecard
        lines.append(_render_scorecard(report.kpis, context))
        lines.append("")

        # Executive summary
        lines.append("\u2500\u2500\u2500 EXECUTIVE SUMMARY \u2500\u2500\u2500")
        lines.append(_render_oracle_summary(report.insights, report.kpis, context))
        lines.append("")

        # Render each lens
        for lens in active_lenses:
            lens_output = self.render_lens(lens, report, context)
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
        lens: str,
        report,
        context: OracleContext,
    ) -> Optional[str]:
        """Generate a single intelligence lens briefing.

        Args:
            lens: One of the 7 lens names.
            report: InsightReport.
            context: OracleContext.

        Returns:
            Formatted lens section string, or None if no insights for this lens.
        """
        if lens not in LENS_CATEGORIES:
            logger.warning(f"Unknown lens: {lens}")
            return None

        categories = LENS_CATEGORIES[lens]
        relevant = [i for i in report.insights if i.category in categories]

        if not relevant:
            return None

        lines = []
        stakeholder = context.stakeholders.get(lens, LENS_AUDIENCES.get(lens, ""))
        lines.append(f"\u2550\u2550\u2550 {LENS_TITLES[lens]} \u2550\u2550\u2550")
        if stakeholder:
            lines.append(f"Prepared for: {stakeholder}")
        lines.append("")

        # Sort: high severity first, then by confidence
        relevant.sort(
            key=lambda x: (-{"high": 3, "medium": 2, "low": 1}[x.severity], -x.confidence)
        )

        rendered_count = 0
        for insight in relevant:
            result = _translate_for_lens(insight, lens, context)
            if result is None:
                continue

            lines.append(result["headline"])
            lines.append(result["body"])

            # Add evidence if available (max 2 per insight)
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

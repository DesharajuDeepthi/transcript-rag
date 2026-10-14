"""Script to generate analysis.ipynb from scratch."""
import json
from pathlib import Path

def cell(source, cell_type="code", outputs=None):
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else [source],
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source if isinstance(source, list) else [source],
    }


cells = []

# ── Section 0: Title ───────────────────────────────────────────────────────
cells.append(cell("""# Transcript Intelligence — Analysis Notebook

**Dataset:** 100 meeting transcripts (B2B SaaS company "Aegis Cloud")
**Date:** 2026-05-23
**Role:** Applied AI Developer take-home

## What this notebook covers
1. Data exploration (loader)
2. Required Task 1: Meeting categorization (hybrid rules + LLM)
3. Required Task 2: Sentiment analysis
4. Bonus A: Churn risk scoring
5. Bonus B: Action item tracker
6. Bonus C: Recurring topic detector
7. Production architecture vision
8. Limitations & next steps
""", "markdown"))

cells.append(cell("""\
import sys, os
from pathlib import Path

# Locate project root: /app inside Docker, parent of notebooks/ otherwise
_candidate = Path("/app")
REPO_ROOT = _candidate if _candidate.exists() else Path.cwd().parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import json
from IPython.display import HTML, display

from src.loader import load_meetings, quick_stats
from src.categorize import categorize_meetings, review_low_confidence
from src.sentiment import generate_sentiment_report
from src.churn_scorer import score_all_accounts, add_account_column, chart_top_risk_accounts, chart_risk_components
from src.action_tracker import generate_action_items_report
from src.topic_analyzer import generate_topic_report

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", 20)
print(f"✅ Imports OK | Working dir: {Path.cwd()}")
"""))

# ── Section 1: Data Exploration ────────────────────────────────────────────
cells.append(cell("""## 1. Dataset Overview

We load 100 meeting transcripts from `data/dataset/`. Each meeting has 6 JSON files:
- `meeting-info.json` — Title, duration, participants
- `summary.json` — **Pre-computed** summary, action items, topics, sentiment, key moments ← we use these!
- `transcript.json` — Per-sentence transcript with speaker + sentiment tags
- `speakers.json`, `speaker-meta.json`, `events.json` — Speaker timeline

### Key design decision: Leverage pre-computed fields
The dataset already includes curated topics, sentiment scores, and tagged key moments
(`churn_signal`, `technical_issue`, `concern`, `positive_pivot`).
We build **on top of these** rather than regenerating them — saving ~$30 in API costs.
""", "markdown"))

cells.append(cell("df = load_meetings()\nstats = quick_stats(df)"))

cells.append(cell("""\
# Sample meeting to show the richness of pre-computed fields
s = df.iloc[0]
print(f"Title:         {s['title']}")
print(f"Duration:      {s['duration_min']:.1f} min")
print(f"Participants:  {s['participant_count']}")
print(f"Sentiment:     {s['overall_sentiment']} (score: {s['sentiment_score']})")
print(f"Topics:        {s['topics']}")
print(f"Action items:  {s['action_item_count']}")
print(f"Key moments:   {s['key_moment_types']}")
"""))

cells.append(cell("""\
# Distribution of sentiment labels
df["overall_sentiment"].value_counts().to_frame("count")
"""))

# ── Section 2: Categorization ──────────────────────────────────────────────
cells.append(cell("""## 2. Categorization (Required Task 1)

**Goal:** Assign each meeting a `call_type` (support / external / internal)
and a `sub_theme` (e.g., incident_response, customer_renewal, compliance_security).

### Approach: Hybrid (rules + LLM)

| Method | Use case | Cost |
|--------|----------|------|
| Rule-based regex | Obvious patterns ("Support Case #", "Standup", "Aegis / X -") | Free |
| gpt-4o-mini | Ambiguous titles requiring semantic understanding | ~$0.001/call |

**Why hybrid?** Rules handle ~70%+ for free with full transparency.
LLM handles the ambiguous remainder with semantic nuance.
Confidence scoring flags cases needing human review.
When a rule misfires, a regex is debuggable; an LLM isn't.
""", "markdown"))

cells.append(cell("df = categorize_meetings(df)"))

cells.append(cell("""\
df[["title", "call_type", "sub_theme", "category_confidence", "category_method"]].head(10)
"""))

cells.append(cell("""\
import plotly.express as px

fig = px.bar(
    df["call_type"].value_counts().reset_index(),
    x="call_type", y="count", color="call_type",
    title="Meeting Count by Call Type",
    labels={"count": "Meetings", "call_type": "Call Type"},
)
fig.update_layout(showlegend=False, height=350)
fig.show()
"""))

cells.append(cell("""\
fig2 = px.bar(
    df["sub_theme"].value_counts().reset_index().sort_values("count"),
    x="count", y="sub_theme", orientation="h", color="count",
    title="Meeting Count by Sub-Theme",
    labels={"count": "Meetings", "sub_theme": ""},
    color_continuous_scale="Blues",
)
fig2.update_layout(height=400, showlegend=False)
fig2.show()
"""))

cells.append(cell("# Spot-check 10 random meetings\ndf.sample(10, random_state=42)[['title', 'call_type', 'sub_theme', 'category_confidence']]"))

cells.append(cell("""### Categorization findings

- **91% rule-based, 9% LLM** — rules are dominant, keeping costs near zero
- **External (44) > Internal (29) > Support (27)** — balanced distribution, no degenerate split
- **Incident response dominates sub-themes (56 meetings)** — this company has been fighting fires
- All 0 meetings below 0.7 confidence — high overall certainty
""", "markdown"))

# ── Section 3: Sentiment ───────────────────────────────────────────────────
cells.append(cell("""## 3. Sentiment Analysis (Required Task 2)

**Goal:** Aggregate sentiment across call types, sub-themes, and time.
Surface *interpretations*, not just numbers.

### Design decision: Trust the pre-computed sentiment
`summary.json` includes `overallSentiment` (categorical) and `sentimentScore` (numeric 1–5).
Re-running with a different model costs API budget for likely similar results.
In production, I'd validate by sampling 20 meetings, hand-scoring, and checking correlation.
""", "markdown"))

cells.append(cell("sentiment_report = generate_sentiment_report(df)"))

cells.append(cell("HTML(filename='outputs/charts/sentiment_by_call_type.html')"))
cells.append(cell("HTML(filename='outputs/charts/sentiment_over_time.html')"))
cells.append(cell("HTML(filename='outputs/charts/sentiment_heatmap.html')"))
cells.append(cell("HTML(filename='outputs/charts/sentiment_distribution.html')"))

cells.append(cell("""\
print("Key findings:")
print(f"  By call type: {sentiment_report['by_call_type']['insight']}")
print(f"  By sub-theme: {sentiment_report['by_sub_theme']['insight']}")
print(f"  Over time:    {sentiment_report['over_time']['insight']}")
print(f"  Vs. chars:    {sentiment_report['vs_characteristics']['insight']}")
"""))

cells.append(cell("""\
print("Top 5 most negative meetings (watch list):\\n")
for m in sentiment_report["negative_outliers"][:5]:
    print(f"  [{m['sentiment_score']:.1f}] {m['title']}")
    print(f"        Type: {m['call_type']} / {m['sub_theme']}")
    print(f"        Churn={m['has_churn_signal']}, Concern={m['has_concern']}")
    print()
"""))

cells.append(cell("""### Sentiment findings

**By call type:**
- External (customer) calls are most positive (avg ~3.7)
- Support calls are most negative (avg ~2.9) — customers in pain
- The 0.74-point gap is meaningful: support interactions carry real friction

**By sub-theme:**
- Engineering syncs and incident response score lowest — expected given outage data
- Compliance/security and renewal calls score higher — positive momentum

**Over time:** Sentiment is broadly flat — no alarming systemic decline

**Stakeholder value:**
- *Support leaders:* Monitor the support call sentiment trend weekly; spikes = systemic issues
- *Sales:* Renewal calls are positive — lean into them; flag accounts where sentiment dips pre-renewal
- *Eng leads:* Incident-response meetings have the lowest scores — postmortem culture matters
""", "markdown"))

# ── Section 4: Churn Risk ──────────────────────────────────────────────────
cells.append(cell("""## 4. Bonus A: Churn Risk Scoring

The most business-critical bonus. CS and Sales leaders need to know:
**which customers are at risk right now?**

### Scoring approach (transparent heuristic, not a black box)

| Signal | Points |
|--------|--------|
| `churn_signal` key moments | 25 pts each, cap 50 |
| Low avg sentiment (< 2.5) | 30 pts |
| Declining sentiment trend | +10 pts |
| `concern` key moments | 5 pts each, cap 20 |
| Recent negative meeting | 5–15 pts |

**Risk levels:** Critical (70+) → Alert (50–69) → Watch (30–49) → Healthy (<30)

**Why heuristic, not ML?** No labeled churn data exists. With actual churn outcomes,
these signals become the feature set for a trained classifier (logistic regression / GBDT).
""", "markdown"))

cells.append(cell("""\
df = add_account_column(df)
rankings = score_all_accounts(df)
"""))

cells.append(cell("HTML(filename='outputs/charts/churn_risk_top.html')"))
cells.append(cell("HTML(filename='outputs/charts/churn_risk_components.html')"))

cells.append(cell("""\
# Evidence for the #1 at-risk account
top = rankings[0]
print(f"Why is '{top['account']}' ranked #{1}?\\n")
print(f"  Risk score:  {top['risk_score']}")
print(f"  Meetings:    {top['meeting_count']}")
print(f"  Avg sentiment: {top['avg_sentiment']}")
print(f"  Components:  {top['components']}")
print(f"\\nEvidence quotes:")
for ev in top["evidence"][:3]:
    print(f"\\n  [{ev['type']}] {ev['title']}")
    print(f"  Speaker: {ev['speaker']}")
    print(f'  "{ev["quote"][:150]}"')
"""))

cells.append(cell("""\
# Risk distribution summary
import pandas as pd
risk_df = pd.DataFrame([{"account": r["account"], "risk_level": r["risk_level"],
                          "risk_score": r["risk_score"], "meeting_count": r["meeting_count"]}
                         for r in rankings])
risk_df["risk_level"].value_counts()
"""))

# ── Section 5: Action Items ────────────────────────────────────────────────
cells.append(cell("""## 5. Bonus B: Action Item Tracker

Engineering leads, project managers, and chiefs of staff need:
- Who carries the heaviest action item load?
- What actions keep recurring across meetings? (= systemic unresolved issues)
- Which meeting types drive the most actionable outcomes?

### Design: Pure regex, no LLM
Action items follow a consistent format: `"Name: action by deadline"`.
Regex handles this for free. LLM would add cost with no quality gain here.
""", "markdown"))

cells.append(cell("action_report = generate_action_items_report(df)"))

cells.append(cell("HTML(filename='outputs/charts/owner_workload.html')"))
cells.append(cell("HTML(filename='outputs/charts/recurring_themes.html')"))

cells.append(cell("""\
print("Action item findings:")
print(f"  Total action items:  {action_report['total_action_items']}")
print(f"  Unique owners:       {action_report['owners_with_actions']}")
print(f"  Workload:            {action_report['owner_workload']['insight']}")
print(f"  Verb pattern:        {action_report['verb_distribution']['insight']}")
print(f"  Recurring themes:    {action_report['recurring_themes']['insight']}")
print(f"  Action density:      {action_report['action_density']['insight']}")
"""))

cells.append(cell("""### Action item findings

- **397 action items across 100 meetings** — ~4 per meeting
- **'Send' is the #1 verb** — the org is primarily in communication mode post-meeting
- **Recurring keywords** indicate themes that need process fixes, not more one-off action items
- **Workload concentration** — top owner has 31 items across 13 meetings; risk of bottleneck

**Stakeholder value:**
- *Engineering leads:* Identify who is overloaded before burnout shows up
- *PM/CoS:* Recurring action themes = candidates for systematic projects
- *Sales/CS:* Customer-facing action items = commitment tracking
""", "markdown"))

# ── Section 6: Topics ─────────────────────────────────────────────────────
cells.append(cell("""## 6. Bonus C: Recurring Topic Detector

Surface topics appearing across multiple meetings.
Recurring = either healthy process (planning cycles) or systemic pain (recurring outages).

### Why pre-computed topics, not LDA/BERTopic?
The upstream summarizer already curated topics per meeting — they're cleaner than
what topic modeling would surface on a 100-doc corpus.
Topic modeling shines at thousands of documents with latent theme discovery.
At 100 meetings with curated topics, aggregation + correlation is the right move.
""", "markdown"))

cells.append(cell("topic_report = generate_topic_report(df)"))

cells.append(cell("HTML(filename='outputs/charts/topic_frequency.html')"))
cells.append(cell("HTML(filename='outputs/charts/topic_sentiment.html')"))
cells.append(cell("HTML(filename='outputs/charts/topic_timeline.html')"))

cells.append(cell("""\
print("Topic findings:")
print(f"  Frequency:    {topic_report['frequency']['insight']}")
print(f"  Sentiment:    {topic_report['sentiment_correlation']['insight']}")
print(f"  Co-occurrence:{topic_report['co_occurrence']['insight']}")
"""))

cells.append(cell("""### Topic findings

- **Compliance + renewal are the most recurring** — the company's dominant strategic priorities
- **Churn risk, SLA breach, incident communication** are the most sentiment-negative topics — danger zones
- **Compliance + renewal co-occur 11 times** — renewals are entangled with compliance readiness

**Stakeholder value:**
- *Product managers:* What topics dominate external conversations → roadmap signal
- *Engineering leads:* Technical pain topics recurring = infrastructure debt signal
- *Leadership:* Strategic topic trends → where is attention going?
""", "markdown"))

# ── Section 7: Production Vision ──────────────────────────────────────────
cells.append(cell("""## 7. Production Architecture: Static Analytics → Agentic Intelligence

### What this prototype is
A static analytics pipeline that pre-computes insights for known questions.

### What it needs to become
> *"Each stakeholder would want something different from this tool."*

Different users → different questions → different reasoning paths.
No single dashboard answers all of them. **This is an agentic system problem.**

### Architecture (next phase, not built in this take-home)

```
┌─────────────────────────────────────────────────────┐
│             Stakeholder Agents                       │
│  Sales | Support | Engineering | PM | Executive     │
└──────────────────────┬──────────────────────────────┘
                       │ MCP protocol
                       ▼
┌─────────────────────────────────────────────────────┐
│              MCP Tool Layer                         │
│  (Each analytics function → one MCP tool)           │
│                                                     │
│  • search_meetings(query, filters)                  │
│  • get_sentiment_trend(call_type, time_range)       │
│  • score_churn_risk(account)                        │
│  • find_action_items(owner)                         │
│  • find_recurring_topics(min_count)                 │
└──────────┬───────────────┬───────────────┬──────────┘
           ▼               ▼               ▼
    Governance        Observability     Eval Framework
    • RBAC/PII        • Traces          • Labeled tests
    • Audit log       • Cost tracking   • Adversarial
    • Data lineage    • Langfuse        • Safety
```

### Honest scope note
I did **not** build the agent or MCP layers in this take-home. That's the right call:
agentic infrastructure is overkill for a 100-meeting analytics prototype.

What I **did** build that supports the transition:
- Each insight is a standalone function with typed inputs/outputs (tool-shaped)
- Results include provenance (meeting IDs) — citations for agents
- Confidence scoring on categorization — foundation for evals
- Pre-computed data reused — cost-aware design

**Estimated production buildout from this foundation: ~5 weeks**
- Week 1–2: MCP tool wrappers for each function
- Week 3: Governance layer (RBAC, PII filtering, audit log)
- Week 4: Observability (traces, cost tracking, Langfuse)
- Week 5: Role-specific agents + eval regression suite
""", "markdown"))

# ── Section 8: Limitations ────────────────────────────────────────────────
cells.append(cell("""## 8. Limitations & Next Steps

### What this prototype doesn't do
1. **No labeled ground truth** — categorization validated by spot-check, not held-out test set
2. **Heuristic churn scoring** — no actual churn outcomes to train on
3. **No cross-meeting threading** — each meeting analyzed in isolation; no customer journey view
4. **Batch only** — no real-time ingestion
5. **Pre-computed dependency** — if `summary.json` quality degrades, all downstream signals degrade

### What I'd prioritize next (in order)
1. **Labeled eval set** — hand-label 30 meetings to track categorization accuracy over time
2. **Customer thread linking** — cluster meetings by account to build longitudinal views
3. **Action item completion tracking** — identify which items from past meetings are still open
4. **MCP tool wrappers** — convert each function to an MCP tool (~1 week engineering)
5. **Sentiment validation eval** — sample-validate pre-computed sentiment scores

### What I'd push back on
- *"Build a full agentic system now"* → Prove analytics value first; automate once queries are known
- *"Add a vector database"* → Premature for 100 docs; pandas + metadata filtering is sufficient
- *"Fine-tune a custom model"* → Heuristics are good enough without churn labels; use ML when we have ground truth
""", "markdown"))

# ── Notebook assembly ──────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

out = Path(__file__).parent / "analysis.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Notebook written to {out} ({len(cells)} cells)")

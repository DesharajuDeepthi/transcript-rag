// Transcript Intelligence — 12-slide deck
// Theme: dark navy #0D1B2A, white text, teal #00B4D8, amber #FFB703

const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10" x 5.625"
pres.title = "Transcript Intelligence";
pres.author = "Applied AI Developer";

// ── Palette ──────────────────────────────────────────────────────────────────
const BG      = "0D1B2A";   // dark navy
const WHITE   = "FFFFFF";
const TEAL    = "00B4D8";
const AMBER   = "FFB703";
const GRAY    = "8EA8C3";   // muted blue-grey
const CARD_BG = "1A2F45";   // slightly lighter navy for cards
const RED     = "E63946";
const GREEN   = "2EC4B6";

// ── Helpers ───────────────────────────────────────────────────────────────────

function bg(slide) {
  slide.background = { color: BG };
}

function title(slide, text, y = 0.35, color = TEAL, size = 34) {
  slide.addText(text, {
    x: 0.5, y, w: 9, h: 0.65,
    fontSize: size, bold: true, color, fontFace: "Calibri",
    align: "left", margin: 0,
  });
}

function subtitle(slide, text, y = 1.0, color = GRAY, size = 14) {
  slide.addText(text, {
    x: 0.5, y, w: 9, h: 0.4,
    fontSize: size, color, fontFace: "Calibri",
    align: "left", margin: 0, italic: true,
  });
}

function card(slide, x, y, w, h, fillColor = CARD_BG) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: fillColor },
    line: { color: TEAL, width: 0.75 },
  });
}

function statBox(slide, x, y, w, number, label, numColor = AMBER) {
  card(slide, x, y, w, 1.25, CARD_BG);
  slide.addText(number, {
    x: x + 0.1, y: y + 0.1, w: w - 0.2, h: 0.65,
    fontSize: 36, bold: true, color: numColor, fontFace: "Calibri",
    align: "center", margin: 0,
  });
  slide.addText(label, {
    x: x + 0.1, y: y + 0.75, w: w - 0.2, h: 0.4,
    fontSize: 11, color: GRAY, fontFace: "Calibri",
    align: "center", margin: 0,
  });
}

function bullets(slide, items, x, y, w, h, size = 13) {
  const runs = items.map((t, i) => ({
    text: t,
    options: { bullet: true, color: WHITE, fontSize: size, breakLine: i < items.length - 1 },
  }));
  slide.addText(runs, { x, y, w, h, fontFace: "Calibri", valign: "top" });
}

function sectionTag(slide, text, x = 0.5, y = 5.2) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 1.8, h: 0.28,
    fill: { color: TEAL },
    line: { color: TEAL },
  });
  slide.addText(text, {
    x, y, w: 1.8, h: 0.28,
    fontSize: 9, bold: true, color: "0D1B2A",
    fontFace: "Calibri", align: "center", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — TITLE
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s);

  // Large accent bar on the left
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625,
    fill: { color: TEAL }, line: { color: TEAL },
  });

  // Faint grid texture (horizontal lines)
  for (let i = 1; i < 6; i++) {
    s.addShape(pres.shapes.LINE, {
      x: 0.2, y: i * 0.9, w: 9.6, h: 0,
      line: { color: "1C3352", width: 0.5 },
    });
  }

  s.addText("TRANSCRIPT", {
    x: 0.6, y: 1.0, w: 9, h: 1.1,
    fontSize: 66, bold: true, color: WHITE,
    fontFace: "Calibri", align: "left", charSpacing: 4, margin: 0,
  });
  s.addText("INTELLIGENCE", {
    x: 0.6, y: 2.0, w: 9, h: 1.1,
    fontSize: 66, bold: true, color: TEAL,
    fontFace: "Calibri", align: "left", charSpacing: 4, margin: 0,
  });
  s.addText("From Meeting Data to Stakeholder Decisions", {
    x: 0.6, y: 3.15, w: 9, h: 0.45,
    fontSize: 18, color: GRAY, fontFace: "Calibri",
    italic: true, align: "left", margin: 0,
  });
  s.addShape(pres.shapes.LINE, {
    x: 0.6, y: 3.65, w: 5, h: 0,
    line: { color: AMBER, width: 1.5 },
  });
  s.addText("Applied AI Developer Take-Home  |  100 Meeting Transcripts  |  May 2026", {
    x: 0.6, y: 3.9, w: 9, h: 0.35,
    fontSize: 12, color: GRAY, fontFace: "Calibri",
    align: "left", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — THE OPPORTUNITY
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s);
  title(s, "100 Meetings. Buried Insights.");
  sectionTag(s, "THE OPPORTUNITY");

  // LEFT column — Today
  card(s, 0.4, 1.15, 4.2, 3.85, "1A2F45");
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.15, w: 4.2, h: 0.4,
    fill: { color: "263545" }, line: { color: TEAL, width: 0.75 },
  });
  s.addText("TODAY", {
    x: 0.5, y: 1.17, w: 4.0, h: 0.36,
    fontSize: 13, bold: true, color: GRAY,
    fontFace: "Calibri", align: "left", margin: 0,
  });
  bullets(s, [
    "Insights are manual, slow, reactive",
    "No cross-meeting visibility",
    "Each stakeholder re-reads transcripts",
    "Churn signals missed until too late",
  ], 0.55, 1.65, 3.9, 3.1, 13);

  // RIGHT column — With TI
  card(s, 5.2, 1.15, 4.35, 3.85, "132838");
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.15, w: 4.35, h: 0.4,
    fill: { color: TEAL }, line: { color: TEAL, width: 0.75 },
  });
  s.addText("WITH TRANSCRIPT INTELLIGENCE", {
    x: 5.3, y: 1.17, w: 4.15, h: 0.36,
    fontSize: 11, bold: true, color: "0D1B2A",
    fontFace: "Calibri", align: "left", margin: 0,
  });
  bullets(s, [
    "Categorized + scored automatically",
    "Sentiment trends surfaced in seconds",
    "At-risk accounts ranked with evidence",
    "Action item owners identified instantly",
  ], 5.35, 1.65, 4.05, 3.1, 13);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — THE DATASET
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s);
  title(s, "What We're Working With");
  subtitle(s, "Feb 3 – Apr 28, 2026  |  Aegis Cloud — B2B SaaS");
  sectionTag(s, "THE DATASET");

  // Stat boxes
  statBox(s, 0.4,  1.45, 2.1, "100",  "meetings");
  statBox(s, 2.65, 1.45, 2.1, "50.5h","of recordings");
  statBox(s, 4.9,  1.45, 2.1, "397",  "action items");
  statBox(s, 7.15, 1.45, 2.1, "76",   "unique speakers");

  // Per-meeting fields
  card(s, 0.4, 2.95, 5.6, 1.75, CARD_BG);
  s.addText("Per-meeting pre-computed fields", {
    x: 0.55, y: 3.0, w: 5.3, h: 0.3,
    fontSize: 11, bold: true, color: TEAL, fontFace: "Calibri", margin: 0,
  });
  bullets(s, [
    "Summary text",
    "Topics list (~6/meeting)",
    "Sentiment score (1–5) + label",
    "Key moments: churn_signal, concern, technical_issue, positive_pivot",
  ], 0.55, 3.3, 5.3, 1.3, 11);

  // Design decision callout
  card(s, 6.2, 2.95, 3.35, 1.75, "0A2238");
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.2, y: 2.95, w: 0.08, h: 1.75,
    fill: { color: AMBER }, line: { color: AMBER },
  });
  s.addText("Design Decision", {
    x: 6.38, y: 3.0, w: 3.1, h: 0.3,
    fontSize: 11, bold: true, color: AMBER, fontFace: "Calibri", margin: 0,
  });
  s.addText("Pre-computed fields already exist — build on top, don't regenerate.\nSaves ~$30 in API costs.", {
    x: 6.38, y: 3.32, w: 3.1, h: 1.25,
    fontSize: 11, color: WHITE, fontFace: "Calibri", margin: 0,
  });

  // Call type mini-bar
  const types = [{ label: "External", pct: 44, color: TEAL }, { label: "Internal", pct: 29, color: AMBER }, { label: "Support", pct: 27, color: GREEN }];
  let bx = 0.4;
  types.forEach(t => {
    const bw = (9.2 * t.pct / 100);
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: 4.9, w: bw, h: 0.38, fill: { color: t.color }, line: { color: t.color } });
    s.addText(`${t.label} ${t.pct}%`, {
      x: bx + 0.05, y: 4.9, w: bw - 0.1, h: 0.38,
      fontSize: 10, bold: true, color: "0D1B2A",
      fontFace: "Calibri", align: "left", valign: "middle", margin: 0,
    });
    bx += bw;
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — APPROACH
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s);
  title(s, "How It's Built");
  sectionTag(s, "APPROACH");

  const pillars = [
    { x: 0.4,  y: 1.1, label: "Hybrid Categorization", color: TEAL,
      body: "Rules handle 91% free.\ngpt-4o-mini for the 9%\nambiguous cases.\nConfidence score on every result." },
    { x: 5.1,  y: 1.1, label: "Trust Pre-Computed Data", color: AMBER,
      body: "summary.json already has\nsentiment, topics, key moments.\nAggregate — don't re-run.\nSaves ~$30 in API cost." },
    { x: 0.4,  y: 3.1, label: "Tool-Shaped Functions", color: GREEN,
      body: "Each insight: typed inputs →\nstructured dict + meeting IDs.\nNatural MCP tool foundation\nfor future agentic system." },
    { x: 5.1,  y: 3.1, label: "Cost-Aware Design", color: RED,
      body: "Total LLM spend: under $1.\nRules first, LLM only when\nneeded. Haiku for cost;\nSonnet for nuance." },
  ];

  pillars.forEach(p => {
    card(s, p.x, p.y, 4.5, 1.9, CARD_BG);
    s.addShape(pres.shapes.RECTANGLE, {
      x: p.x, y: p.y, w: 4.5, h: 0.38,
      fill: { color: p.color }, line: { color: p.color },
    });
    s.addText(p.label, {
      x: p.x + 0.12, y: p.y + 0.04, w: 4.26, h: 0.3,
      fontSize: 13, bold: true, color: "0D1B2A",
      fontFace: "Calibri", align: "left", margin: 0,
    });
    s.addText(p.body, {
      x: p.x + 0.12, y: p.y + 0.48, w: 4.26, h: 1.35,
      fontSize: 12, color: WHITE, fontFace: "Calibri", margin: 0,
    });
  });

  // Pipeline arrow bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 5.18, w: 9.2, h: 0.3,
    fill: { color: "102030" }, line: { color: GRAY, width: 0.5 },
  });
  s.addText("Raw Meetings  →  Loader  →  Categorize  →  Sentiment  →  Churn + Actions + Topics  →  Charts", {
    x: 0.5, y: 5.18, w: 9.0, h: 0.3,
    fontSize: 10, color: GRAY, fontFace: "Calibri",
    align: "center", valign: "middle", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — CATEGORIZATION
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s);
  title(s, "Meeting Categorization — Required Task 1");
  subtitle(s, "Hybrid: Rules (91%) + gpt-4o-mini (9%)  |  Zero meetings below 0.7 confidence");
  sectionTag(s, "REQUIRED TASK 1");

  // Chart
  s.addChart(pres.charts.BAR, [{
    name: "Meetings",
    labels: ["External", "Internal", "Support"],
    values: [44, 29, 27],
  }], {
    x: 0.4, y: 1.4, w: 4.5, h: 3.7,
    barDir: "col",
    chartColors: [TEAL, AMBER, GREEN],
    chartArea: { fill: { color: CARD_BG } },
    catAxisLabelColor: GRAY,
    valAxisLabelColor: GRAY,
    valGridLine: { color: "1C3352", size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelColor: WHITE,
    dataLabelFontSize: 13,
    dataLabelFontBold: true,
    showLegend: false,
  });

  // Sub-themes table
  const themes = [
    ["incident_response",    "56", "dominant"],
    ["customer_support_issue","12",""],
    ["compliance_security",  "11", ""],
    ["customer_onboarding",  " 6", ""],
    ["product_planning",     " 6", ""],
    ["customer_renewal",     " 5", ""],
    ["other",                " 3", ""],
  ];

  const tableData = [
    [
      { text: "Sub-Theme", options: { bold: true, color: "0D1B2A", fill: { color: TEAL }, fontSize: 11 } },
      { text: "Meetings",  options: { bold: true, color: "0D1B2A", fill: { color: TEAL }, fontSize: 11 } },
    ],
    ...themes.map(([theme, count, note]) => [
      { text: theme + (note ? `  ★` : ""), options: { color: note ? AMBER : WHITE, fontSize: 10, bold: !!note } },
      { text: count, options: { color: WHITE, fontSize: 10, align: "center" } },
    ]),
  ];
  s.addTable(tableData, {
    x: 5.2, y: 1.4, w: 4.35,
    colW: [3.3, 1.05],
    rowH: 0.37,
    border: { pt: 0.5, color: "1C3352" },
    fill: { color: CARD_BG },
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — SENTIMENT ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s);
  title(s, "Sentiment Analysis — Required Task 2");
  subtitle(s, "Aggregating pre-computed sentiment scores (1–5 scale) across 100 meetings");
  sectionTag(s, "REQUIRED TASK 2");

  // Sentiment bar chart — by call type
  s.addChart(pres.charts.BAR, [{
    name: "Avg Sentiment",
    labels: ["Support", "Internal", "External"],
    values: [2.94, 3.1, 3.68],
  }], {
    x: 0.4, y: 1.35, w: 4.4, h: 2.6,
    barDir: "bar",
    chartColors: [RED, AMBER, TEAL],
    chartArea: { fill: { color: CARD_BG } },
    catAxisLabelColor: GRAY,
    valAxisLabelColor: GRAY,
    valGridLine: { color: "1C3352", size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelColor: WHITE,
    dataLabelFontSize: 12,
    dataLabelFontBold: true,
    showLegend: false,
    valAxisMaxVal: 5,
    valAxisMinVal: 0,
  });

  // Findings — 4 cards, each tall enough to show 2-line text at 10.5pt
  const findings = [
    { color: RED,   text: "Support calls most negative: avg 2.94 vs External 3.68 (gap: 0.74 pts)" },
    { color: AMBER, text: "Incident response and engineering sync are lowest-scoring sub-themes" },
    { color: TEAL,  text: "Sentiment is FLAT Feb-Apr (slope +0.012/week, no alarming decline)" },
    { color: GREEN, text: "Longer meetings correlate with higher sentiment (r = +0.36)" },
  ];

  findings.forEach((f, i) => {
    const cy = 1.35 + i * 0.74;
    card(s, 5.05, cy, 4.5, 0.65, CARD_BG);
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.05, y: cy, w: 0.08, h: 0.65,
      fill: { color: f.color }, line: { color: f.color },
    });
    s.addText(f.text, {
      x: 5.25, y: cy + 0.04, w: 4.2, h: 0.57,
      fontSize: 10.5, color: WHITE, fontFace: "Calibri", margin: 0, valign: "top",
    });
  });

  // Stakeholder row — safely below last card (ends at 1.35 + 3*0.74 + 0.65 = 4.22)
  card(s, 0.4, 4.38, 9.2, 0.9, "102030");
  s.addText("Support leaders: monitor weekly  |  Sales: external calls are positive — lean in at renewal  |  Eng leads: invest in postmortem culture", {
    x: 0.55, y: 4.40, w: 9.0, h: 0.86,
    fontSize: 10.5, color: GRAY, fontFace: "Calibri",
    align: "center", valign: "middle", italic: true, margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — CHURN RISK SCORING
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s);
  title(s, "Bonus A: Churn Risk Scoring");
  subtitle(s, "Which customers are at risk right now?");
  sectionTag(s, "BONUS A");

  // Scoring method
  card(s, 0.4, 1.1, 4.3, 2.35, CARD_BG);
  s.addText("Scoring Method", {
    x: 0.55, y: 1.15, w: 4.0, h: 0.3,
    fontSize: 12, bold: true, color: TEAL, fontFace: "Calibri", margin: 0,
  });
  bullets(s, [
    "churn_signal moments: 25 pts each (cap 50)",
    "Low / declining sentiment: up to 40 pts",
    "concern moments: 5 pts each (cap 20)",
    "Recent negative meeting: 5–15 pts",
  ], 0.55, 1.5, 4.0, 1.85, 11.5);

  // Risk level table
  const riskData = [
    [
      { text: "Level",   options: { bold: true, color: "0D1B2A", fill: { color: TEAL }, fontSize: 11 } },
      { text: "Score",   options: { bold: true, color: "0D1B2A", fill: { color: TEAL }, fontSize: 11 } },
      { text: "Accounts",options: { bold: true, color: "0D1B2A", fill: { color: TEAL }, fontSize: 11 } },
      { text: "Action",  options: { bold: true, color: "0D1B2A", fill: { color: TEAL }, fontSize: 11 } },
    ],
    [
      { text: "Critical", options: { color: RED,   bold: true, fontSize: 11 } },
      { text: "70+",      options: { color: WHITE, fontSize: 11 } },
      { text: "13",       options: { color: WHITE, bold: true, fontSize: 13 } },
      { text: "Exec outreach this week", options: { color: WHITE, fontSize: 10 } },
    ],
    [
      { text: "Alert",   options: { color: AMBER, bold: true, fontSize: 11 } },
      { text: "50–69",   options: { color: WHITE, fontSize: 11 } },
      { text: "3",       options: { color: WHITE, bold: true, fontSize: 13 } },
      { text: "CS check-in within 7 days", options: { color: WHITE, fontSize: 10 } },
    ],
    [
      { text: "Watch",   options: { color: "FFD166", bold: true, fontSize: 11 } },
      { text: "30–49",   options: { color: WHITE, fontSize: 11 } },
      { text: "6",       options: { color: WHITE, bold: true, fontSize: 13 } },
      { text: "Monitor closely", options: { color: WHITE, fontSize: 10 } },
    ],
    [
      { text: "Healthy", options: { color: GREEN, bold: true, fontSize: 11 } },
      { text: "<30",     options: { color: WHITE, fontSize: 11 } },
      { text: "18",      options: { color: WHITE, bold: true, fontSize: 13 } },
      { text: "No action needed", options: { color: WHITE, fontSize: 10 } },
    ],
  ];
  s.addTable(riskData, {
    x: 4.9, y: 1.1, w: 4.65,
    colW: [1.15, 0.85, 1.0, 1.65],
    rowH: 0.44,
    border: { pt: 0.5, color: "1C3352" },
    fill: { color: CARD_BG },
  });

  // Top 3 callout
  card(s, 0.4, 3.65, 9.2, 1.3, "0A2238");
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 3.65, w: 9.2, h: 0.32,
    fill: { color: RED }, line: { color: RED },
  });
  s.addText("TOP 3 AT-RISK ACCOUNTS", {
    x: 0.55, y: 3.65, w: 9.0, h: 0.32,
    fontSize: 11, bold: true, color: WHITE, fontFace: "Calibri",
    align: "center", valign: "middle", margin: 0,
  });
  s.addText("Northstar Pharma  [100 pts]          Summit Trust  [75 pts]          Vanta Health Systems  [75 pts]", {
    x: 0.55, y: 4.0, w: 9.0, h: 0.8,
    fontSize: 15, bold: true, color: AMBER, fontFace: "Calibri",
    align: "center", valign: "middle", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — CHURN EVIDENCE
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s);
  // Smaller font so long title stays on one line and doesn't crowd the subtitle
  title(s, "#1 At-Risk: Northstar Pharma  (Score: 100/100)", 0.3, TEAL, 26);
  s.addText("Every score is explainable. Every claim is evidence-backed.", {
    x: 0.5, y: 0.82, w: 9, h: 0.32,
    fontSize: 13, color: GRAY, fontFace: "Calibri",
    align: "left", italic: true, margin: 0,
  });
  sectionTag(s, "BONUS A — EVIDENCE");

  // Component bars — well below subtitle (starts at y=1.3)
  const components = [
    { label: "Churn Signals",    score: 50, max: 50, color: RED   },
    { label: "Sentiment",        score: 30, max: 40, color: AMBER },
    { label: "Concerns",         score: 20, max: 20, color: "FF9F1C" },
    { label: "Recent Negativity",score: 15, max: 15, color: TEAL  },
  ];
  components.forEach((c, i) => {
    const y = 1.3 + i * 0.72;
    s.addText(c.label, {
      x: 0.4, y, w: 2.2, h: 0.45,
      fontSize: 12, color: GRAY, fontFace: "Calibri", align: "right", valign: "middle", margin: 0,
    });
    const barW = 3.5 * (c.score / c.max);
    s.addShape(pres.shapes.RECTANGLE, {
      x: 2.7, y: y + 0.06, w: 3.5, h: 0.35,
      fill: { color: "1A2F45" }, line: { color: "1C3352", width: 0.5 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 2.7, y: y + 0.06, w: barW, h: 0.35,
      fill: { color: c.color }, line: { color: c.color },
    });
    s.addText(`${c.score} pts`, {
      x: 6.3, y, w: 0.9, h: 0.45,
      fontSize: 12, bold: true, color: c.color, fontFace: "Calibri", align: "left", valign: "middle", margin: 0,
    });
  });

  // Total — bars end at 1.3 + 3*0.72 + 0.45 = 4.31
  s.addShape(pres.shapes.LINE, { x: 2.7, y: 4.2, w: 4.5, h: 0, line: { color: GRAY, width: 0.75 } });
  s.addText("TOTAL", { x: 0.4, y: 4.25, w: 2.2, h: 0.38, fontSize: 13, bold: true, color: WHITE, fontFace: "Calibri", align: "right", margin: 0 });
  s.addText("100 pts (capped)  ->  CRITICAL", { x: 2.7, y: 4.25, w: 4.5, h: 0.38, fontSize: 13, bold: true, color: RED, fontFace: "Calibri", margin: 0 });

  // Quote callouts — right side, aligned with bar rows
  const quotes = [
    "churn_signal: Explicit customer dissatisfaction flagged in key moments",
    "concern: Customer raised critical unresolved issue — support case still open",
  ];
  quotes.forEach((q, i) => {
    card(s, 7.05, 1.3 + i * 1.45, 2.6, 1.28, "0A2238");
    s.addShape(pres.shapes.RECTANGLE, {
      x: 7.05, y: 1.3 + i * 1.45, w: 0.08, h: 1.28,
      fill: { color: i === 0 ? RED : AMBER }, line: { color: i === 0 ? RED : AMBER },
    });
    s.addText(`"${q}"`, {
      x: 7.2, y: 1.35 + i * 1.45, w: 2.35, h: 1.12,
      fontSize: 10, color: WHITE, italic: true, fontFace: "Calibri", margin: 0,
    });
  });

  // Recommendation
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 4.82, w: 9.2, h: 0.52,
    fill: { color: "200A0A" }, line: { color: RED, width: 1.5 },
  });
  s.addText("Recommendation: Immediate executive outreach. High likelihood of churn.  |  Tool ranks, human decides.", {
    x: 0.55, y: 4.82, w: 9.0, h: 0.52,
    fontSize: 11, color: WHITE, fontFace: "Calibri",
    align: "center", valign: "middle", bold: true, margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — ACTION ITEM TRACKER
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s);
  title(s, "Bonus B: Action Item Tracker");
  // Stat chips inline — avoids special-char subtitle artifacts
  const chipData = [["397", "action items"], ["100", "meetings"], ["4", "avg per meeting"]];
  chipData.forEach(([num, lbl], i) => {
    statBox(s, 0.4 + i * 3.07, 1.02, 2.8, num, lbl, TEAL);
  });
  sectionTag(s, "BONUS B");

  // 3 columns — start below stat chips (chips end at y=1.02+1.25=2.27)
  const cols = [
    {
      x: 0.4, color: RED,
      header: "WHO IS OVERLOADED?",
      items: ["Maria Santos: 31 items, 13 meetings", "Top 5 owners hold 35% of all actions", "Risk of single-point-of-failure"],
    },
    {
      x: 3.55, color: AMBER,
      header: "WHAT ARE WE DOING?",
      items: ['Top verb: "send" (71 occurrences)', "Org is in communication mode post-meeting", '"deliver", "prepare", "review" follow'],
    },
    {
      x: 6.7, color: TEAL,
      header: "WHAT IS RECURRING?",
      items: ['"team", "week", "update" in 40+ meetings', "Recurring = unresolved systemic issue", "These need a project, not more action items"],
    },
  ];

  cols.forEach(c => {
    card(s, c.x, 2.38, 3.0, 2.5, CARD_BG);
    s.addShape(pres.shapes.RECTANGLE, {
      x: c.x, y: 2.38, w: 3.0, h: 0.42,
      fill: { color: c.color }, line: { color: c.color },
    });
    s.addText(c.header, {
      x: c.x + 0.1, y: 2.4, w: 2.8, h: 0.38,
      fontSize: 11, bold: true, color: "0D1B2A",
      fontFace: "Calibri", align: "center", valign: "middle", margin: 0,
    });
    bullets(s, c.items, c.x + 0.12, 2.9, 2.76, 1.9, 11.5);
  });

  // Footer
  card(s, 0.4, 5.06, 9.2, 0.38, "102030");
  s.addText("Engineering leads: IC workload visibility before burnout.  |  PM/CoS: recurring themes = candidates for systematic fixes, not more action items.", {
    x: 0.55, y: 5.06, w: 9.0, h: 0.38,
    fontSize: 10, color: GRAY, fontFace: "Calibri",
    align: "center", valign: "middle", italic: true, margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — RECURRING TOPICS
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s);
  title(s, "Bonus C: Recurring Topic Detector");
  subtitle(s, "47 topics appear in 3+ meetings across the corpus");
  sectionTag(s, "BONUS C");

  // Frequency chart
  s.addChart(pres.charts.BAR, [{
    name: "Meetings",
    labels: ["compliance", "compliance reporting", "renewal", "outage", "customer communication"],
    values: [23, 19, 17, 14, 13],
  }], {
    x: 0.4, y: 1.3, w: 4.5, h: 3.1,
    barDir: "bar",
    chartColors: [TEAL],
    chartArea: { fill: { color: CARD_BG } },
    catAxisLabelColor: GRAY,
    valAxisLabelColor: GRAY,
    valGridLine: { color: "1C3352", size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelColor: WHITE,
    dataLabelFontSize: 11,
    showLegend: false,
  });

  // Sentiment danger zone
  card(s, 5.1, 1.3, 4.45, 2.0, CARD_BG);
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.3, w: 4.45, h: 0.38,
    fill: { color: RED }, line: { color: RED },
  });
  s.addText("MOST SENTIMENT-NEGATIVE TOPICS", {
    x: 5.2, y: 1.3, w: 4.25, h: 0.38,
    fontSize: 10, bold: true, color: WHITE, fontFace: "Calibri",
    align: "center", valign: "middle", margin: 0,
  });
  const negTopics = [
    ["churn risk",            "2.12"],
    ["SLA breach",            "2.13"],
    ["incident communication","2.16"],
  ];
  negTopics.forEach(([topic, score], i) => {
    s.addText(topic, {
      x: 5.2, y: 1.78 + i * 0.44, w: 3.2, h: 0.38,
      fontSize: 12, color: WHITE, fontFace: "Calibri", margin: 0,
    });
    s.addText(`avg ${score}`, {
      x: 8.4, y: 1.78 + i * 0.44, w: 1.0, h: 0.38,
      fontSize: 12, bold: true, color: RED, fontFace: "Calibri", align: "right", margin: 0,
    });
  });

  // Co-occurrence
  card(s, 5.1, 3.5, 4.45, 1.35, CARD_BG);
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 3.5, w: 0.08, h: 1.35,
    fill: { color: AMBER }, line: { color: AMBER },
  });
  s.addText("Top Co-Occurrence", {
    x: 5.25, y: 3.55, w: 4.2, h: 0.3,
    fontSize: 11, bold: true, color: AMBER, fontFace: "Calibri", margin: 0,
  });
  s.addText('"compliance" + "renewal" appear together in\n11 meetings — renewals are entangled\nwith compliance readiness.', {
    x: 5.25, y: 3.87, w: 4.2, h: 0.9,
    fontSize: 11, color: WHITE, fontFace: "Calibri", margin: 0,
  });

  // PM insight footer
  card(s, 0.4, 5.0, 9.2, 0.42, "102030");
  s.addText("Product managers: what topics dominate external conversations → roadmap signal  |  Eng leads: recurring technical pain topics = infrastructure debt signal", {
    x: 0.55, y: 5.0, w: 9.0, h: 0.42,
    fontSize: 10, color: GRAY, fontFace: "Calibri",
    align: "center", valign: "middle", italic: true, margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 11 — PRODUCTION VISION
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s);
  title(s, "From Static Analytics → Agentic Intelligence");
  sectionTag(s, "PRODUCTION VISION");

  // Insight hook
  card(s, 0.4, 1.05, 9.2, 0.65, "0A2238");
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.05, w: 0.1, h: 0.65,
    fill: { color: AMBER }, line: { color: AMBER },
  });
  s.addText('"...each stakeholder would want something different from this tool."  →  Not a dashboard problem. An agentic system problem.', {
    x: 0.6, y: 1.08, w: 8.9, h: 0.58,
    fontSize: 12, italic: true, color: AMBER, fontFace: "Calibri", valign: "middle", margin: 0,
  });

  // Architecture layers
  const layers = [
    { label: "Stakeholder Agents", sub: "Sales  |  Support  |  Engineering  |  PM  |  Executive", color: TEAL, y: 1.9 },
    { label: "MCP Tool Layer", sub: "score_churn_risk()  |  get_sentiment_trend()  |  find_recurring_topics()  |  find_action_items()", color: AMBER, y: 2.75 },
    { label: "Governance + Observability + Evals", sub: "RBAC / PII Filtering  |  Traces + Cost Tracking  |  Regression + Safety Tests", color: GREEN, y: 3.6 },
    { label: "Data Layer (this prototype)", sub: "100 meetings  →  flat DataFrame  →  categorized + scored + charted", color: GRAY, y: 4.45 },
  ];

  layers.forEach((l, i) => {
    card(s, 0.4, l.y, 6.6, 0.72, CARD_BG);
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.4, y: l.y, w: 0.1, h: 0.72,
      fill: { color: l.color }, line: { color: l.color },
    });
    s.addText(l.label, {
      x: 0.62, y: l.y + 0.04, w: 6.2, h: 0.3,
      fontSize: 12, bold: true, color: l.color, fontFace: "Calibri", margin: 0,
    });
    s.addText(l.sub, {
      x: 0.62, y: l.y + 0.36, w: 6.2, h: 0.28,
      fontSize: 10, color: GRAY, fontFace: "Calibri", margin: 0,
    });
    if (i < 3) {
      s.addText("↓", {
        x: 3.2, y: l.y + 0.72, w: 0.5, h: 0.03,
        fontSize: 14, color: GRAY, fontFace: "Calibri", align: "center", margin: 0,
      });
    }
  });

  // Honest note
  card(s, 7.2, 1.9, 2.4, 3.27, "0A2238");
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.2, y: 1.9, w: 2.4, h: 0.38,
    fill: { color: "1C3352" }, line: { color: TEAL, width: 0.75 },
  });
  s.addText("What I built", {
    x: 7.25, y: 1.92, w: 2.3, h: 0.34,
    fontSize: 11, bold: true, color: TEAL, fontFace: "Calibri", align: "center", margin: 0,
  });
  bullets(s, [
    "Tool-shaped functions",
    "Typed inputs → structured dict",
    "Provenance (meeting IDs)",
    "Confidence scoring",
    "Cost-aware design",
  ], 7.25, 2.32, 2.3, 1.85, 10);

  s.addText("~5 weeks to full production", {
    x: 7.25, y: 4.62, w: 2.3, h: 0.46,
    fontSize: 11, bold: true, color: AMBER, fontFace: "Calibri",
    align: "center", valign: "middle", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 12 — LIMITATIONS + NEXT STEPS
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s);
  title(s, "Honest Limitations & Next Steps");
  sectionTag(s, "CLOSING");

  // LEFT — Limitations
  card(s, 0.4, 1.1, 4.3, 3.75, CARD_BG);
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.1, w: 4.3, h: 0.4,
    fill: { color: RED }, line: { color: RED },
  });
  s.addText("KNOWN LIMITATIONS", {
    x: 0.5, y: 1.12, w: 4.1, h: 0.36,
    fontSize: 12, bold: true, color: WHITE,
    fontFace: "Calibri", align: "center", valign: "middle", margin: 0,
  });
  bullets(s, [
    "No labeled ground truth for categorization accuracy",
    "Heuristic churn scoring — no actual churn outcomes to train on",
    "Single-document analysis — no customer journey threading",
    "Pre-computed dependency — quality follows upstream summarizer",
  ], 0.55, 1.62, 4.0, 3.1, 12);

  // RIGHT — Next steps
  card(s, 5.0, 1.1, 4.6, 3.75, CARD_BG);
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.0, y: 1.1, w: 4.6, h: 0.4,
    fill: { color: TEAL }, line: { color: TEAL },
  });
  s.addText("WHAT I'D BUILD NEXT", {
    x: 5.1, y: 1.12, w: 4.4, h: 0.36,
    fontSize: 12, bold: true, color: "0D1B2A",
    fontFace: "Calibri", align: "center", valign: "middle", margin: 0,
  });

  const steps = [
    ["1", "Labeled eval set", "Hand-label 30 meetings; track categorization accuracy"],
    ["2", "Customer threading", "Link meetings by account across time"],
    ["3", "MCP tool wrappers", "1 week to convert each function"],
    ["4", "Sales agent prototype", 'Demo "show me churn risk" end-to-end'],
    ["5", "Governance + evals", "RBAC + audit log + regression tests (2 weeks each)"],
  ];
  steps.forEach(([num, label, detail], i) => {
    s.addShape(pres.shapes.OVAL, {
      x: 5.1, y: 1.65 + i * 0.62, w: 0.3, h: 0.3,
      fill: { color: TEAL }, line: { color: TEAL },
    });
    s.addText(num, {
      x: 5.1, y: 1.65 + i * 0.62, w: 0.3, h: 0.3,
      fontSize: 10, bold: true, color: "0D1B2A",
      fontFace: "Calibri", align: "center", valign: "middle", margin: 0,
    });
    s.addText(label, {
      x: 5.5, y: 1.65 + i * 0.62, w: 3.95, h: 0.22,
      fontSize: 12, bold: true, color: WHITE, fontFace: "Calibri", margin: 0,
    });
    s.addText(detail, {
      x: 5.5, y: 1.87 + i * 0.62, w: 3.95, h: 0.2,
      fontSize: 10, color: GRAY, fontFace: "Calibri", margin: 0,
    });
  });

  // Banner
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 5.06, w: 9.2, h: 0.42,
    fill: { color: TEAL }, line: { color: TEAL },
  });
  s.addText("Total path to production agentic system: ~5 weeks from this analytics foundation", {
    x: 0.5, y: 5.06, w: 9.0, h: 0.42,
    fontSize: 13, bold: true, color: "0D1B2A",
    fontFace: "Calibri", align: "center", valign: "middle", margin: 0,
  });
}

// ── Write ────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: "/Users/deepthidesharaju/Documents/TakeHOme/slides/transcript_intelligence.pptx" })
  .then(() => console.log("✅ Saved: slides/transcript_intelligence.pptx"))
  .catch(err => { console.error("Error:", err); process.exit(1); });

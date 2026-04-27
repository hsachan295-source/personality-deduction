
import streamlit as st
import pandas as pd
import pickle
import numpy as np


from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroType · Personality AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (dark-glass premium theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
}
[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 13px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* ── Sliders ── */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #fff !important;
    border: 3px solid #818cf8 !important;
    box-shadow: 0 0 12px rgba(99,102,241,0.6) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 2.5rem;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.04em;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(99,102,241,0.45);
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.65);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.25) !important;
    color: #c7d2fe !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── DataFrames ── */
.dataframe { background: rgba(255,255,255,0.04) !important; color: #e2e8f0 !important; }
thead th { background: rgba(99,102,241,0.2) !important; color: #c7d2fe !important; }

/* ── Result banner ── */
.result-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    margin: 1rem 0;
}
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.result-sub { color: #94a3b8; font-size: 15px; margin-top: 0.5rem; }

/* ── Section headers ── */
.section-hdr {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #c7d2fe;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Info / success boxes ── */
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL + SCALER
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("personality_model.pkl", "rb") as f:
        mdl = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scl = pickle.load(f)
    return mdl, scl

model, scaler = load_artifacts()

# ─────────────────────────────────────────────
#  FEATURE DEFINITIONS  (name → category → icon)
# ─────────────────────────────────────────────
FEATURE_META = {
    # Social
    "social_energy":            ("Social & Communication", "⚡"),
    "talkativeness":            ("Social & Communication", "💬"),
    "group_comfort":            ("Social & Communication", "👥"),
    "party_liking":             ("Social & Communication", "🎉"),
    "friendliness":             ("Social & Communication", "🤝"),
    "online_social_usage":      ("Social & Communication", "📱"),
    "public_speaking_comfort":  ("Social & Communication", "🎤"),
    # Inner world
    "alone_time_preference":    ("Inner World", "🌙"),
    "deep_reflection":          ("Inner World", "🔮"),
    "reading_habit":            ("Inner World", "📚"),
    "listening_skill":          ("Inner World", "👂"),
    "empathy":                  ("Inner World", "💜"),
    # Action & Lifestyle
    "risk_taking":              ("Action & Lifestyle", "🎲"),
    "excitement_seeking":       ("Action & Lifestyle", "🔥"),
    "spontaneity":              ("Action & Lifestyle", "💥"),
    "adventurousness":          ("Action & Lifestyle", "🧗"),
    "sports_interest":          ("Action & Lifestyle", "⚽"),
    "travel_desire":            ("Action & Lifestyle", "✈️"),
    # Work & Thinking
    "organization":             ("Work & Thinking", "📋"),
    "leadership":               ("Work & Thinking", "🏆"),
    "curiosity":                ("Work & Thinking", "🔍"),
    "routine_preference":       ("Work & Thinking", "🔄"),
    "planning":                 ("Work & Thinking", "📅"),
    "decision_speed":           ("Work & Thinking", "⚡"),
    "work_style_collaborative": ("Work & Thinking", "🤜"),
    "gadget_usage":             ("Work & Thinking", "💻"),
}

FEATURES = list(FEATURE_META.keys())
CATEGORIES = list(dict.fromkeys(v[0] for v in FEATURE_META.values()))

CATEGORY_COLORS = {
    "Social & Communication": "#6366f1",
    "Inner World":            "#8b5cf6",
    "Action & Lifestyle":     "#ec4899",
    "Work & Thinking":        "#06b6d4",
}

PERSONALITY_INFO = {
    "Extrovert": {
        "emoji": "🌟",
        "color": "#f59e0b",
        "desc": "You thrive in social settings, draw energy from others, and love being the center of action.",
        "strengths": ["Natural leader", "High social energy", "Expressive communicator", "Action-oriented"],
        "growth":    ["Practice deep listening", "Build solitude tolerance", "Slow down before deciding"],
    },
    "Introvert": {
        "emoji": "🌙",
        "color": "#8b5cf6",
        "desc": "You recharge alone, think deeply, and prefer meaningful one-on-one connections over crowds.",
        "strengths": ["Deep thinker", "Excellent listener", "Creative & reflective", "Self-aware"],
        "growth":    ["Push comfort zone socially", "Share ideas more openly", "Seek collaborative wins"],
    },
    "Ambivert": {
        "emoji": "⚖️",
        "color": "#06b6d4",
        "desc": "You sit at the sweet spot — adaptable, versatile, and comfortable in both social and solitary settings.",
        "strengths": ["Highly adaptable", "Emotionally balanced", "Versatile communicator", "Empathetic"],
        "growth":    ["Identify energy drains early", "Set clear boundaries", "Embrace your fluid nature"],
    },
}

LABEL_MAP = {0: "Ambivert", 1: "Extrovert", 2: "Introvert"}


# ─────────────────────────────────────────────
#  SIDEBAR  — category score summaries
# ─────────────────────────────────────────────
def render_sidebar(user_input: dict):
    with st.sidebar:
        st.markdown("## 🧭 NeuroType")
        st.caption("AI Personality Intelligence")
        st.divider()

        st.markdown("### Category Averages")
        for cat in CATEGORIES:
            feats = [f for f, (c, _) in FEATURE_META.items() if c == cat]
            avg = np.mean([user_input[f] for f in feats])
            col = CATEGORY_COLORS[cat]
            pct = int(avg * 10)
            st.markdown(f"""
            <div style="margin-bottom:14px">
              <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px">
                <span style="color:#e2e8f0">{cat}</span>
                <span style="color:{col};font-weight:600">{avg:.1f}</span>
              </div>
              <div style="background:rgba(255,255,255,0.08);border-radius:6px;height:6px">
                <div style="background:{col};width:{pct}%;height:6px;border-radius:6px;transition:width .4s ease"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### Quick Stats")
        overall = np.mean(list(user_input.values()))
        top3 = sorted(user_input, key=user_input.get, reverse=True)[:3]
        low3 = sorted(user_input, key=user_input.get)[:3]
        st.metric("Overall Average", f"{overall:.1f} / 10")
        st.markdown("**Top traits**")
        for t in top3:
            icon = FEATURE_META[t][1]
            st.markdown(f"&nbsp;&nbsp;{icon} `{t.replace('_',' ').title()}` — **{user_input[t]:.1f}**")
        st.markdown("**Growth areas**")
        for t in low3:
            icon = FEATURE_META[t][1]
            st.markdown(f"&nbsp;&nbsp;{icon} `{t.replace('_',' ').title()}` — **{user_input[t]:.1f}**")


# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────
def radar_chart(user_input: dict):
    cats = CATEGORIES
    vals = []
    for cat in cats:
        feats = [f for f, (c, _) in FEATURE_META.items() if c == cat]
        vals.append(round(np.mean([user_input[f] for f in feats]), 2))
    vals_closed = vals + [vals[0]]
    cats_closed = cats + [cats[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill="toself",
        fillcolor="rgba(99,102,241,0.2)", line=dict(color="#818cf8", width=2),
        marker=dict(size=7, color="#c084fc"),
        name="Your Profile",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 10], color="#475569",
                            gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=10, color="#64748b")),
            angularaxis=dict(color="#94a3b8", gridcolor="rgba(255,255,255,0.08)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=320,
    )
    return fig


def bar_chart(user_input: dict):
    sorted_items = sorted(user_input.items(), key=lambda x: x[1], reverse=True)
    labels = [k.replace("_", " ").title() for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors_list = [CATEGORY_COLORS[FEATURE_META[k][0]] for k, _ in sorted_items]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors_list, line=dict(width=0)),
        text=[f"{v:.1f}" for v in values], textposition="outside",
        textfont=dict(size=11, color="#94a3b8"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 11], color="#475569", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10)),
        yaxis=dict(color="#94a3b8", tickfont=dict(size=11)),
        margin=dict(l=10, r=50, t=10, b=10),
        height=560,
        font=dict(family="Inter"),
    )
    return fig


def confidence_gauge(prob: float, label: str):
    color = PERSONALITY_INFO[label]["color"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob, 1),
        number={"suffix": "%", "font": {"size": 36, "color": "#f1f5f9", "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 50],  "color": "rgba(255,255,255,0.04)"},
                {"range": [50, 75], "color": "rgba(255,255,255,0.06)"},
                {"range": [75, 100],"color": "rgba(255,255,255,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": prob},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        margin=dict(l=20, r=20, t=20, b=0),
        height=200,
    )
    return fig


def proba_bar(proba_dict: dict):
    labels = list(proba_dict.keys())
    values = [round(v * 100, 1) for v in proba_dict.values()]
    cols_list = [PERSONALITY_INFO[l]["color"] for l in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=cols_list, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=13, color="#e2e8f0"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#94a3b8", tickfont=dict(size=13)),
        yaxis=dict(range=[0, 110], color="#475569", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10)),
        margin=dict(l=10, r=10, t=20, b=10),
        height=220,
        font=dict(family="Inter"),
    )
    return fig


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem">
      <div style="display:inline-block;background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
                  border-radius:20px;padding:6px 18px;font-size:12px;color:#818cf8;
                  letter-spacing:.08em;text-transform:uppercase;margin-bottom:1rem">
        🧠 AI · Personality Intelligence
      </div>
      <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.6rem;font-weight:700;
                 background:linear-gradient(135deg,#818cf8,#c084fc,#67e8f9);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 letter-spacing:-.02em;margin-bottom:.5rem">
        NeuroType Predictor
      </h1>
      <p style="color:#64748b;font-size:15px;max-width:520px;margin:0 auto">
        Calibrate 26 personality dimensions and discover your cognitive-social archetype
        powered by a trained machine learning model.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Input sliders (by category, in tabs) ──
    st.markdown('<div class="section-hdr">① Calibrate Your Traits</div>', unsafe_allow_html=True)
    st.caption("Adjust each slider honestly — your results are only as accurate as your inputs.")

    user_input: dict = {}
    tabs = st.tabs([f"{CATEGORY_COLORS[c] and ''}{c}" for c in CATEGORIES])

    for tab, cat in zip(tabs, CATEGORIES):
        with tab:
            feats = [f for f, (c, _) in FEATURE_META.items() if c == cat]
            col1, col2 = st.columns(2)
            for i, feat in enumerate(feats):
                icon = FEATURE_META[feat][1]
                label = f"{icon} {feat.replace('_', ' ').title()}"
                target_col = col1 if i % 2 == 0 else col2
                with target_col:
                    user_input[feat] = st.slider(
                        label, min_value=0.0, max_value=10.0, value=5.0, step=0.1,
                        key=feat,
                    )

    # Ensure all features are present (in order)
    ordered_input = {f: user_input[f] for f in FEATURES}

    # ── Sidebar ──
    render_sidebar(ordered_input)

    st.divider()

    # ── Visualisation before prediction ──
    st.markdown('<div class="section-hdr">② Your Trait Profile</div>', unsafe_allow_html=True)
    v_col1, v_col2 = st.columns([1, 1.6])
    with v_col1:
        st.caption("Category radar")
        st.plotly_chart(radar_chart(ordered_input), use_container_width=True, config={"displayModeBar": False})
    with v_col2:
        with st.expander("📊 All 26 traits ranked", expanded=False):
            st.plotly_chart(bar_chart(ordered_input), use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── Predict ──
    st.markdown('<div class="section-hdr">③ Run Prediction</div>', unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_clicked = st.button("🔮  Analyse My Personality", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame([ordered_input])
        scaled = scaler.transform(input_df)
        raw_pred = model.predict(scaled)[0]
        proba_arr = model.predict_proba(scaled)[0]
        label = LABEL_MAP.get(raw_pred, "Unknown")
        confidence = float(proba_arr.max() * 100)
        proba_dict = {LABEL_MAP[i]: float(p) for i, p in enumerate(proba_arr)}
        info = PERSONALITY_INFO[label]

        # ── Result banner ──
        st.markdown(f"""
        <div class="result-banner">
          <div style="font-size:3.5rem;margin-bottom:.5rem">{info['emoji']}</div>
          <div class="result-title">{label}</div>
          <div class="result-sub">{info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics row ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prediction", label, info["emoji"])
        m2.metric("Confidence", f"{confidence:.1f}%")
        m3.metric("Top Category", max(
            CATEGORIES,
            key=lambda c: np.mean([ordered_input[f] for f, (cat, _) in FEATURE_META.items() if cat == c])
        ))
        m4.metric("Traits Analysed", "26")

        # ── Detailed results tabs ──
        r1, r2, r3 = st.tabs(["📈 Probability Breakdown", "🧬 Trait Deep Dive", "📋 Full Input Summary"])

        with r1:
            rc1, rc2 = st.columns([1.2, 1])
            with rc1:
                st.caption("All-class probabilities")
                st.plotly_chart(proba_bar(proba_dict), use_container_width=True, config={"displayModeBar": False})
            with rc2:
                st.caption("Confidence meter")
                st.plotly_chart(confidence_gauge(confidence, label), use_container_width=True, config={"displayModeBar": False})
                for lbl, prob in sorted(proba_dict.items(), key=lambda x: x[1], reverse=True):
                    col = PERSONALITY_INFO[lbl]["color"]
                    pct = round(prob * 100, 1)
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:8px 12px;background:rgba(255,255,255,0.04);
                                border-radius:8px;margin-bottom:6px;border:1px solid rgba(255,255,255,0.06)">
                      <span style="font-size:13px;color:#e2e8f0">{PERSONALITY_INFO[lbl]['emoji']} {lbl}</span>
                      <span style="font-size:14px;font-weight:600;color:{col}">{pct}%</span>
                    </div>
                    """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"#### Strengths of a {label}")
            s_cols = st.columns(len(info["strengths"]))
            for col, s in zip(s_cols, info["strengths"]):
                col.markdown(f"""
                <div style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);
                             border-radius:12px;padding:14px;text-align:center;font-size:13px;color:#c7d2fe">
                  ✦ {s}
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"#### Growth Opportunities")
            g_cols = st.columns(len(info["growth"]))
            for col, g in zip(g_cols, info["growth"]):
                col.markdown(f"""
                <div style="background:rgba(236,72,153,0.1);border:1px solid rgba(236,72,153,0.2);
                             border-radius:12px;padding:14px;text-align:center;font-size:13px;color:#f9a8d4">
                  ◈ {g}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Category Scores")
            for cat in CATEGORIES:
                feats = [f for f, (c, _) in FEATURE_META.items() if c == cat]
                avg = np.mean([ordered_input[f] for f in feats])
                col_hex = CATEGORY_COLORS[cat]
                pct = int(avg * 10)
                st.markdown(f"""
                <div style="margin-bottom:12px">
                  <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                    <span style="color:#e2e8f0;font-size:13px;font-weight:500">{cat}</span>
                    <span style="color:{col_hex};font-size:13px;font-weight:600">{avg:.1f} / 10</span>
                  </div>
                  <div style="background:rgba(255,255,255,0.06);border-radius:8px;height:8px">
                    <div style="background:{col_hex};width:{pct}%;height:8px;border-radius:8px"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with r3:
            styled_df = (
                input_df.T
                .rename(columns={0: "Your Score"})
                .assign(Category=lambda df: df.index.map(lambda x: FEATURE_META[x][0]),
                        Icon=lambda df: df.index.map(lambda x: FEATURE_META[x][1]))
                .reset_index()
                .rename(columns={"index": "Feature"})
                [["Icon", "Feature", "Category", "Your Score"]]
            )
            styled_df["Feature"] = styled_df["Feature"].str.replace("_", " ").str.title()
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Your Score": st.column_config.ProgressColumn(
                        "Your Score", min_value=0, max_value=10, format="%.1f"
                    )
                },
            )

        # ── Download ──
        st.divider()
        csv = input_df.copy()
        csv.insert(0, "Predicted_Type", label)
        csv.insert(1, "Confidence_Pct", round(confidence, 2))
        st.download_button(
            "⬇️  Download Results as CSV",
            data=csv.to_csv(index=False).encode(),
            file_name="neurotype_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

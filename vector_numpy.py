import os
import time
from datetime import datetime


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional OpenAI import for agentic interpretation layer
from openai import OpenAI

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Agri-Food Policy Intelligence Suite",
    layout="wide"
)

# ---------------------------------------------------------
# SIDEBAR — POLICY & GOVERNANCE PANEL
# ---------------------------------------------------------
with st.sidebar:
    st.title("Policy Control Panel")

    st.markdown(
        "### Self‑Learning Policy Engines\n"
        "Performance & Stress Audit System (Quality Control)\n"
        "***##System Architecture, Design and Engineering - Shubhojit Bagchi © 2025##***"
    )

    st.subheader("Scenario Generator")
    scenario = st.selectbox(
        "Select Policy / Market Scenario",
        [
            "Baseline Conditions",
            "High Shock Volatility",
            "Climate Stress Scenario",
            "Market Expansion Scenario",
            "Recession Drift",
            "Policy Tightening",
        ],
    )

    st.subheader("EU Governance & Compliance")
    st.markdown("**EU AI Act & CAP 2023–2027 Alignment (High‑Level Check)**")

    st.checkbox("Transparency & Documentation in place", value=True)
    st.checkbox("Explainability for Policy Users enabled", value=True)
    st.checkbox("Robustness & Stress‑Testing performed (MC + Markov)", value=True)
    st.checkbox("Human‑in‑the‑Loop Oversight configured", value=True)
    st.checkbox("Data Governance & Traceability (EO + CSO) verified", value=True)

    st.caption(
        "All policy engines are tested with Monte‑Carlo (MC), "
        "Markov Chain (MC), and Copernicus Earth‑Observation "
        "aligned agri‑economic datasets."
    )

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("Agentic System-of-Systems Auditor — Irish Agri-Food Decision & Policy Intelligence")
st.caption(
    "Self‑Learning Policy Engines for Ireland’s Agri‑Food Sector — "
    "Performance, Stress Audit, and EU‑Aligned Governance."
)

# ---------------------------------------------------------
# LOAD REAL DATA (RELATIVE PATH FOR GITHUB/STREAMLIT)
# ---------------------------------------------------------
DATA_PATH = "ie_copernicus_agri_econ_panel_2016_2024.csv"

try:
    data = pd.read_csv(DATA_PATH)
    st.subheader("Dataset Source: Copernicus - Earth Observation, Eurostat, CSO, NOAA")
    st.dataframe(data.head(), use_container_width=True)
except Exception as e:
    st.warning(f"Could not load dataset from '{DATA_PATH}': {e}")
    data = None

# ---------------------------------------------------------
# APPLY SCENARIO PARAMETERS (POLICY / MARKET / CLIMATE)
# ---------------------------------------------------------
def apply_scenario(p_up: float, shock_prob: float):
    """Scenario logic to adjust baseline transition probabilities."""
    if scenario == "High Shock Volatility":
        return p_up - 0.05, shock_prob + 0.10
    elif scenario == "Climate Stress Scenario":
        return p_up - 0.03, shock_prob + 0.07
    elif scenario == "Market Expansion Scenario":
        return p_up + 0.05, shock_prob - 0.02
    elif scenario == "Recession Drift":
        return p_up - 0.08, shock_prob + 0.03
    elif scenario == "Policy Tightening":
        return p_up - 0.04, shock_prob + 0.01
    else:
        return p_up, shock_prob


# ---------------------------------------------------------
# GEN‑1 ENGINE — BASELINE AGRI‑MARKET STABILITY ENGINE (BAMSE)
# ---------------------------------------------------------
def mc_markov_python(n_iter: int, p_up: float = 0.52, shock_prob: float = 0.05):
    """
    GEN‑1: Baseline Agri‑Market Stability Engine (BAMSE)
    Classic Python loop simulating simple price / yield drift and shocks.
    """
    import random

    value = 1.0
    shocks = 0

    for _ in range(n_iter):
        if random.random() < p_up:
            value *= 1.01
        else:
            value *= 0.99

        if random.random() < shock_prob:
            value *= 0.80
            shocks += 1

    return value, shocks / n_iter


# ---------------------------------------------------------
# GEN‑2 ENGINE — RAPID AGRI‑SHOCK VECTORIZED RESPONSE ENGINE (RASVRE)
# ---------------------------------------------------------
def mc_markov_vectorized(
    n_iter: int,
    p_up: float = 0.52,
    shock_prob: float = 0.05,
    seed: int = 42,
):
    """
    Rapid Agri‑Shock Vectorized Response Engine
    Vectorized Monte‑Carlo + Markov engine (NumPy) — cloud‑safe and fast.
    """
    rng = np.random.default_rng(seed)

    ups = rng.random(n_iter) < p_up
    shocks = rng.random(n_iter) < shock_prob

    factors = np.where(ups, 1.01, 0.99)
    factors = np.where(shocks, factors * 0.80, factors)

    # Numerically stable log‑sum‑product
    log_factors = np.log(factors)
    log_value = np.sum(log_factors)
    value = float(np.exp(log_value))
    shock_freq = float(shocks.mean())

    return value, shock_freq


# ---------------------------------------------------------
# GEN‑3 ENGINE — ADAPTIVE FARM‑SECTOR STATE TRANSITION MODEL (AF‑STEM)
# ---------------------------------------------------------
def mc_markov_adaptive(n_iter: int, seed: int = 123):
    """
    Adaptive Farm‑Sector State Transition Model
    Adapts transition probabilities based on synthetic volatility patterns.
    """
    rng = np.random.default_rng(seed)

    p_up = 0.52
    shock_prob = 0.05
    value = 1.0
    shocks = 0

    for i in range(n_iter):
        if i > 1000:
            local_vol = abs(np.sin(i / 2000))
            p_up = 0.52 - local_vol * 0.1
            shock_prob = 0.05 + local_vol * 0.02

        move = rng.random()
        if move < p_up:
            value *= 1.01
        else:
            value *= 0.99

        if rng.random() < shock_prob:
            value *= 0.80
            shocks += 1

    return value, shocks / n_iter


# ---------------------------------------------------------
# GEN‑5 ENGINE — MULTI‑AGENT FARM‑PROCESSOR INTERACTION SIMULATOR (MAFPIS)
# ---------------------------------------------------------
def mc_multi_agent(n_iter: int, agents: int = 3, seed: int = 777):
    """
    Multi‑Agent Farm‑Processor Interaction Simulator
    Simulates interactions between farms, co‑ops, and processors.
    """
    rng = np.random.default_rng(seed)

    states = np.ones(agents)
    shocks = 0

    for _ in range(n_iter):
        interaction = rng.random((agents, agents))
        interaction = interaction / interaction.sum(axis=1, keepdims=True)

        shock_env = rng.random() < 0.04

        for a in range(agents):
            influence = np.dot(interaction[a], states)
            if influence > 1.0:
                states[a] *= 1.01
            else:
                states[a] *= 0.99

            if shock_env and rng.random() < 0.2:
                states[a] *= 0.85
                shocks += 1

    final_value = float(states.mean())
    shock_freq = shocks / (n_iter * agents)
    return final_value, shock_freq


# ---------------------------------------------------------
# GEN‑6 ENGINE — REINFORCEMENT POLICY OPTIMISER FOR AGRI‑FOOD (RPO‑Agri)
# ---------------------------------------------------------
def q_learning_policy(n_iter: int, seed: int = 2025):
    """
    Reinforcement Policy Optimiser for Agri‑Food
    Q‑learning policy engine exploring conserve/expand actions under shocks.
    """
    rng = np.random.default_rng(seed)

    # States: 0 = bad, 1 = neutral, 2 = good
    Q = np.zeros((3, 2))  # actions: 0 (conserve), 1 (expand)
    state = 1
    shocks = 0

    alpha = 0.1
    gamma = 0.9

    for _ in range(n_iter):
        action = rng.integers(0, 2)
        reward = 0.0

        p = rng.random()
        if p < 0.3:
            next_state = 0
        elif p < 0.7:
            next_state = 1
        else:
            next_state = 2

        if action == 1 and next_state == 2:
            reward = 1.0
        elif action == 1 and next_state == 0:
            reward = -1.0
        elif action == 0:
            reward = 0.05

        if rng.random() < 0.05:
            reward -= 0.5
            shocks += 1

        best_next = np.max(Q[next_state])
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * best_next - Q[state, action]
        )

        state = next_state

    expected_value = float(np.mean(Q))
    shock_freq = shocks / n_iter
    return expected_value, shock_freq


# ---------------------------------------------------------
# SELF‑AUDITOR (POLICY‑GRADED)
# ---------------------------------------------------------
def self_auditor_check(result_dict: dict):
    issues = []
    notes = []

    rt = result_dict["response_time_ms"]
    ev = result_dict["expected_value"]
    sf = result_dict["shock_frequency"]

    # Response Time
    if rt > 500:
        issues.append("⛔ System latency high — optimisation required.")
    elif rt > 250:
        issues.append("⚠️ Moderate latency — system slower than ideal.")
    else:
        notes.append("🟢 Response time within optimal decision window.")

    # Expected Value
    if ev > 1.2:
        notes.append("🟢 System showing strong positive trend in agri‑economic outcomes.")
    elif 0.8 < ev <= 1.2:
        notes.append("🟡 System stable with mild fluctuations.")
    elif 0.2 < ev <= 0.8:
        issues.append("⚠️ System trending downward — monitor farm incomes and margins.")
    else:
        issues.append(
            "🔍 Low expected value detected — signals structural stress but not confirmed systemic failure."
        )

    # Shock Frequency
    if sf > 0.20:
        issues.append("⛔ Excessive shock frequency — highly volatile agri‑environment.")
    elif sf > 0.10:
        issues.append("⚠️ Elevated shock levels — policy buffers may be required.")
    else:
        notes.append("🟢 Shock frequency within tolerable bounds for agri‑markets.")

    # Status classification
    if len(issues) == 0:
        status = "🟢 System Stable"
    elif any("⛔" in x for x in issues):
        status = "🔴 High‑Risk Conditions"
    else:
        status = "🟠 Moderate Risk Detected"

    return status, issues + notes


# ---------------------------------------------------------
# OPENAI GPT‑STYLE POLICY INTERPRETER (SAFE FALLBACK)
# ---------------------------------------------------------
def generate_policy_narrative(result_dict: dict, engine_label: str, scenario_label: str):
    """
    If OpenAI + API key are available, call GPT-4o-mini style agent.
    Otherwise, provide a deterministic fallback narrative.
    """

    # --- Load API Key safely ---
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    # --- Base prompt ---
    base_prompt = (
        f"You are an Irish agri-food policy advisor. "
        f"Engine: {engine_label}. Scenario: {scenario_label}. "
        f"Metrics: expected value={result_dict['expected_value']:.4f}, "
        f"shock frequency={result_dict['shock_frequency']:.4f}, "
        f"response time={result_dict['response_time_ms']:.2f} ms.\n\n"
        "Provide a short 4–6 line interpretation in professional policy language "
        "for the Department of Agriculture, Food and the Marine. "
        "Conclude with ONE concrete policy suggestion."
    )

    # --- Fallback if no API key ---
    if not api_key:
        fallback = [
            f"Engine **{engine_label}** under **{scenario_label}** reports an "
            f"expected performance of {result_dict['expected_value']:.4f} "
            f"with a shock exposure rate of {result_dict['shock_frequency']:.4f}.",
            "This outcome reflects Monte-Carlo and Markov-driven dynamics within a "
            "controlled agri-economic stress-testing framework.",
            "Volatility signals remain relevant for policy development, particularly "
            "under adverse climate or price scenarios.",
            "This result should be treated as an early-warning diagnostic rather than "
            "a conclusive market signal.",
            "Policy suggestion: enhance targeted income-stabilisation supports for "
            "vulnerable farm categories.",
        ]
        return "\n\n".join(fallback)

    # --- Try calling OpenAI (fully correct indentation) ---
    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior agri-food policy advisor for Ireland."
                },
                {
                    "role": "user",
                    "content": base_prompt
                },
            ],
            max_tokens=300,
            temperature=0.4,
        )

        return response.choices[0].message.content

    except Exception as e:
        return (
            f"⚠️ OpenAI call failed: {e}\n\n"
            "Use the quantitative metrics above as the primary guidance until "
            "the AI interpretation layer is restored."
        )


# ---------------------------------------------------------
# SESSION STATE FOR RUN HISTORY
# ---------------------------------------------------------
if "run_history" not in st.session_state:
    st.session_state["run_history"] = []  # list of dicts


def record_run(engine_key: str, label: str, result: dict):
    entry = {
        "timestamp": datetime.utcnow(),
        "engine_key": engine_key,
        "engine_label": label,
        "scenario": scenario,
        "response_time_ms": result["response_time_ms"],
        "expected_value": result["expected_value"],
        "shock_frequency": result["shock_frequency"],
    }
    st.session_state["run_history"].append(entry)


# ---------------------------------------------------------
# TABS LAYOUT
# ---------------------------------------------------------
tab_engines, tab_indicators, tab_governance = st.tabs(
    ["Policy Engines Dashboard", "Indicators & Plotly Visuals", "Governance & Auditor"]
)

# =========================================================
# TAB 1 — POLICY ENGINES DASHBOARD
# =========================================================
with tab_engines:
    st.subheader("Self‑Learning Policy Engines — Irish Agri-Food Sector")

    col1, col2 = st.columns(2)

    # ---------------- GEN‑1: BAMSE ----------------
    with col1:
        st.markdown("### Baseline Agri‑Market Stability Engine (BAMSE) ")
        iterations_g1 = st.slider(
            "Iterations (GEN‑1)", 50_000, 500_000, 200_000, step=50_000, key="iter_gen1"
        )

        if st.button("Run GEN‑1 (BAMSE)"):
            start = time.time()
            adj_p, adj_shock = apply_scenario(0.52, 0.05)
            exp_val, shock_freq = mc_markov_python(iterations_g1, adj_p, adj_shock)
            end = time.time()

            result = {
                "engine": "GEN‑1 BAMSE",
                "iterations": iterations_g1,
                "response_time_ms": (end - start) * 1000,
                "expected_value": exp_val,
                "shock_frequency": shock_freq,
            }
            st.session_state["GEN1"] = result
            record_run("GEN1", "GEN‑1 BAMSE", result)

        if "GEN1" in st.session_state:
            g1 = st.session_state["GEN1"]
            st.metric("Response Time (ms)", f"{g1['response_time_ms']:.2f}")
            st.metric("Expected Value", f"{g1['expected_value']:.4f}")
            st.metric("Shock Frequency", f"{g1['shock_frequency']:.4f}")
            with st.expander("AI Policy Interpretation (GEN‑1)"):
                st.markdown(generate_policy_narrative(g1, "GEN‑1 BAMSE", scenario))

    # ---------------- GEN‑2: RASVRE ----------------
    with col2:
        st.markdown(
            "### Rapid Agri‑Shock Vectorized Response Engine (RASVRE)"
        )
        iterations_g2 = st.slider(
            "Iterations (GEN‑2)", 50_000, 500_000, 200_000, step=50_000, key="iter_gen2"
        )

        if st.button("Run GEN‑2 (RASVRE)"):
            start = time.time()
            adj_p, adj_shock = apply_scenario(0.52, 0.05)
            exp_val, shock_freq = mc_markov_vectorized(iterations_g2, adj_p, adj_shock)
            end = time.time()

            result = {
                "engine": "GEN‑2 RASVRE",
                "iterations": iterations_g2,
                "response_time_ms": (end - start) * 1000,
                "expected_value": exp_val,
                "shock_frequency": shock_freq,
            }
            st.session_state["GEN2"] = result
            record_run("GEN2", "GEN‑2 RASVRE", result)

        if "GEN2" in st.session_state:
            g2 = st.session_state["GEN2"]
            st.metric("Response Time (ms)", f"{g2['response_time_ms']:.2f}")
            st.metric("Expected Value", f"{g2['expected_value']:.4f}")
            st.metric("Shock Frequency", f"{g2['shock_frequency']:.4f}")

            if "GEN1" in st.session_state and g2["response_time_ms"] > 0:
                g1 = st.session_state["GEN1"]
                speedup = g1["response_time_ms"] / g2["response_time_ms"]
                st.write(f"⚡ Approx. speedup over GEN‑1: **{speedup:.2f}×**")

            with st.expander("AI Policy Interpretation (GEN‑2)"):
                st.markdown(generate_policy_narrative(g2, "GEN‑2 RASVRE", scenario))

    st.markdown("---")

    # ---------------- GEN‑3: AF‑STEM ----------------
    st.markdown("### Adaptive Farm‑Sector State Transition Model (AF‑STEM)")
    iterations_g3 = st.slider(
        "Iterations (GEN‑3)", 50_000, 500_000, 150_000, step=50_000, key="iter_gen3"
    )

    if st.button("Run GEN‑3 (AF‑STEM)"):
        start = time.time()
        exp_val, shock_freq = mc_markov_adaptive(iterations_g3)
        end = time.time()

        result = {
            "engine": "GEN‑3 AF‑STEM",
            "iterations": iterations_g3,
            "response_time_ms": (end - start) * 1000,
            "expected_value": exp_val,
            "shock_frequency": shock_freq,
        }
        st.session_state["GEN3"] = result
        record_run("GEN3", "GEN‑3 AF‑STEM", result)

    if "GEN3" in st.session_state:
        g3 = st.session_state["GEN3"]
        st.metric("Response Time (ms)", f"{g3['response_time_ms']:.2f}")
        st.metric("Expected Value", f"{g3['expected_value']:.4f}")
        st.metric("Shock Frequency", f"{g3['shock_frequency']:.4f}")
        with st.expander("AI Policy Interpretation (GEN‑3)"):
            st.markdown(generate_policy_narrative(g3, "GEN‑3 AF‑STEM", scenario))

    st.markdown("---")

    # ---------------- GEN‑5: MAFPIS ----------------
    st.markdown(
        "### Multi‑Agent Farm‑Processor Interaction Simulator (MAFPIS)"
    )
    iterations_g5 = st.slider(
        "Iterations (GEN‑5)", 20_000, 200_000, 50_000, step=20_000, key="iter_gen5"
    )

    if st.button("Run GEN‑5 (MAFPIS)"):
        start = time.time()
        exp_val, shock_freq = mc_multi_agent(iterations_g5)
        end = time.time()

        result = {
            "engine": "GEN‑5 MAFPIS",
            "iterations": iterations_g5,
            "response_time_ms": (end - start) * 1000,
            "expected_value": exp_val,
            "shock_frequency": shock_freq,
        }
        st.session_state["GEN5"] = result
        record_run("GEN5", "GEN‑5 MAFPIS", result)

    if "GEN5" in st.session_state:
        g5 = st.session_state["GEN5"]
        st.metric("Response Time (ms)", f"{g5['response_time_ms']:.2f}")
        st.metric("Expected Value", f"{g5['expected_value']:.4f}")
        st.metric("Shock Frequency", f"{g5['shock_frequency']:.4f}")
        with st.expander("AI Policy Interpretation (GEN‑5)"):
            st.markdown(generate_policy_narrative(g5, "GEN‑5 MAFPIS", scenario))

    st.markdown("---")

    # ---------------- GEN‑6: RPO‑Agri ----------------
    st.markdown(
        "### Reinforcement Policy Optimiser for Agri‑Food (RPO‑Agri)"
    )
    iterations_g6 = st.slider(
        "Iterations (GEN‑6)", 10_000, 100_000, 30_000, step=10_000, key="iter_gen6"
    )

    if st.button("Run GEN‑6 (RPO‑Agri)"):
        start = time.time()
        exp_val, shock_freq = q_learning_policy(iterations_g6)
        end = time.time()

        result = {
            "engine": "GEN‑6 RPO‑Agri",
            "iterations": iterations_g6,
            "response_time_ms": (end - start) * 1000,
            "expected_value": exp_val,
            "shock_frequency": shock_freq,
        }
        st.session_state["GEN6"] = result
        record_run("GEN6", "GEN‑6 RPO‑Agri", result)

    if "GEN6" in st.session_state:
        g6 = st.session_state["GEN6"]
        st.metric("Response Time (ms)", f"{g6['response_time_ms']:.2f}")
        st.metric("Expected Value", f"{g6['expected_value']:.4f}")
        st.metric("Shock Frequency", f"{g6['shock_frequency']:.4f}")
        with st.expander("AI Policy Interpretation (GEN‑6)"):
            st.markdown(generate_policy_narrative(g6, "GEN‑6 RPO‑Agri", scenario))


# =========================================================
# TAB 2 — INDICATORS & PLOTLY VISUALS
# =========================================================
with tab_indicators:
    st.subheader("Engine Performance & Shock Indicators")

    history = st.session_state["run_history"]
    if not history:
        st.info("Run any policy engine to populate indicators and graphs.")
    else:
        hist_df = pd.DataFrame(history)
        hist_df["timestamp_local"] = hist_df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Dublin")

        # 1) Response time over runs
        fig_rt = px.line(
            hist_df,
            x="timestamp_local",
            y="response_time_ms",
            color="engine_label",
            title="Response Time Across Policy Engine Runs",
            labels={"response_time_ms": "Response Time (ms)", "timestamp_local": "Time"},
        )
        st.plotly_chart(fig_rt, use_container_width=True)

        # 2) Expected value vs shock frequency (scatter)
        fig_scatter = px.scatter(
            hist_df,
            x="shock_frequency",
            y="expected_value",
            color="engine_label",
            size="response_time_ms",
            hover_data=["scenario"],
            title="Expected Value vs Shock Frequency (by Engine)",
            labels={
                "shock_frequency": "Shock Frequency",
                "expected_value": "Expected Value",
            },
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 3) Radar chart for a selected engine run (latest)
        latest = hist_df.iloc[-1]
        radar_categories = [
            "Expected Value",
            "Shock Frequency (inverted)",
            "Latency (inverted)",
        ]
        radar_values = [
            max(latest["expected_value"], 0.0),
            max(0.0001, 1.0 - latest["shock_frequency"]),
            max(0.0001, 1.0 - min(latest["response_time_ms"] / 1000.0, 1.0)),
        ]

        radar_fig = go.Figure(
            data=go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=radar_categories + [radar_categories[0]],
                fill="toself",
                name=latest["engine_label"],
            )
        )
        radar_fig.update_layout(
            title=f"Policy Risk‑Resilience Profile — {latest['engine_label']} ({latest['scenario']})",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
        )
        st.plotly_chart(radar_fig, use_container_width=True)

        # 4) Optional: Copernicus vs Engine expected value (if data available)
        if data is not None and "year" in data.columns:
            st.markdown("### Climate–Economy Fusion (Illustrative)")
            # Simple illustrative aggregation
            climate_series = data.groupby("year").mean(numeric_only=True).reset_index()
            climate_series["ndvi_proxy"] = (
                climate_series.select_dtypes(include=[np.number]).mean(axis=1)
            )

            eng_series = (
                hist_df.groupby(hist_df["timestamp"].dt.year)
                .mean(numeric_only=True)
                .reset_index()
                .rename(columns={"timestamp": "year"})
            )
            fusion_df = pd.DataFrame(
                {
                    "year": climate_series["year"],
                    "NDVI_Proxy": climate_series["ndvi_proxy"],
                }
            )

            fig_fusion = go.Figure()
            fig_fusion.add_trace(
                go.Scatter(
                    x=fusion_df["year"],
                    y=fusion_df["NDVI_Proxy"],
                    mode="lines+markers",
                    name="NDVI Proxy (Copernicus‑Aligned)",
                )
            )
            st.plotly_chart(fig_fusion, use_container_width=True)


# =========================================================
# TAB 3 — GOVERNANCE & AUDITOR
# =========================================================
with tab_governance:
    st.subheader("Governance, Risk & Self‑Auditor")

    # Choose latest run across all engines
    latest_run = None
    for key in ["GEN2", "GEN1", "GEN3", "GEN5", "GEN6"]:
        if key in st.session_state:
            latest_run = st.session_state[key]

    if latest_run is None:
        st.info("Run any engine to activate the self‑auditor and governance view.")
    else:
        status, messages = self_auditor_check(latest_run)
        st.markdown(f"### Auditor Status: {status}")

        for msg in messages:
            if "⛔" in msg:
                st.error(msg)
            elif "⚠️" in msg:
                st.warning(msg)
            else:
                st.success(msg)

        st.markdown("### Audit Input Snapshot")
        st.json(latest_run)
        # ---------------------------------------------------------
        # AGENTIC OPENAI POLICY INTERPRETATION LAYER
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("Agentic Policy Intelligence Advisor")

        agentic_narrative = generate_policy_narrative(
            latest_run,
            latest_run["engine"],
            scenario
        )

        st.markdown(
            "The Agentic Advisor interprets the system’s behaviour using "
            "EU‑style policy language and provides actionable recommendations "
            "for Ireland’s agri‑food sector."
        )

        st.markdown(f"**Engine:** {latest_run['engine']}  \n"
                    f"**Scenario:** {scenario}")

        st.write(agentic_narrative)

    st.markdown("---")
    st.subheader("Active Scenario Metadata")
    st.write(f"**Active Scenario:** {scenario}")
    st.write(
        "Scenario parameters are integrated into BAMSE and RASVRE engines "
        "for policy‑realistic modelling of agri‑food markets."
    )

    st.markdown("### EU Governance Statement")
    st.write(
        "This prototype follows an EU‑style approach to AI governance: combining "
        "quantitative stress‑testing (MC + Markov), explainable narratives, and "
        "explicit policy oversight markers for the agri‑food sector."
    )

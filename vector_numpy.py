import streamlit as st
import pandas as pd
import numpy as np
import time

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Meta System Gen-2", layout="wide")

st.title("📡 Meta System Gen-2 — Vectorized GEN-2 (Cloud Safe)")
st.caption("Real-time Monte-Carlo + Markov Risk Engine | GEN-1 vs GEN-2 Benchmark | Self-Auditor")

# ---------------------------------------------------------
# LOAD REAL DATA (RELATIVE PATH FOR GITHUB/STREAMLIT)
# ---------------------------------------------------------
DATA_PATH = "ie_copernicus_agri_econ_panel_2016_2024.csv"

try:
    data = pd.read_csv(DATA_PATH)
    st.subheader("📊 Real Data Loaded: Irish Agri-Econ Panel (2016–2024)")
    st.dataframe(data.head(), use_container_width=True)
except Exception as e:
    st.warning(f"Could not load dataset from '{DATA_PATH}': {e}")
    data = None

# ---------------------------------------------------------
# GEN-1 ENGINE — PURE PYTHON (BASELINE)
# ---------------------------------------------------------
def mc_markov_python(n_iter: int, p_up: float = 0.52, shock_prob: float = 0.05):
    import random
    value = 1.0
    shocks = 0

    for _ in range(n_iter):
        # Markov-like up/down move
        if random.random() < p_up:
            value *= 1.01
        else:
            value *= 0.99

        # Rare negative shock
        if random.random() < shock_prob:
            value *= 0.80
            shocks += 1

    return value, shocks / n_iter

# ---------------------------------------------------------
# GEN-2 ENGINE — NUMPY VECTORIZED (STREAMLIT-CLOUD SAFE)
# ---------------------------------------------------------
def mc_markov_vectorized(n_iter: int, p_up: float = 0.52, shock_prob: float = 0.05, seed: int = 42):
    """
    Vectorized Monte-Carlo + Markov shock engine.
    Uses NumPy arrays instead of Python loops.
    Cloud-safe (no Cython/Numba) but significantly faster than GEN-1.
    """
    rng = np.random.default_rng(seed)

    # Draw random uniforms for up/down and shocks
    ups = rng.random(n_iter) < p_up           # True = up, False = down
    shocks = rng.random(n_iter) < shock_prob  # True = shock

    # Base movement factor: 1.01 if up else 0.99
    factors = np.where(ups, 1.01, 0.99)

    # Apply shock multiplier where shock occurs
    factors = np.where(shocks, factors * 0.80, factors)

    # ----- NEW numerically stable log-sum-product engine -----
    # Compute log factors to avoid underflow
    log_factors = np.log(factors)

    # Sum logs instead of multiplying tiny numbers
    log_value = np.sum(log_factors)

    # Convert back from log-space
    value = float(np.exp(log_value))

    shock_freq = float(shocks.mean())

    return value, shock_freq

# ---------------------------------------------------------
# SELF-AUDITOR
# ---------------------------------------------------------
def self_auditor_check(result_dict):
    issues = []
    notes = []

    rt = result_dict["response_time_ms"]
    ev = result_dict["expected_value"]
    sf = result_dict["shock_frequency"]

    # --- Response Time Classification ---
    if rt > 500:
        issues.append("⛔ System latency high — optimisation required.")
    elif rt > 250:
        issues.append("⚠️ Moderate latency — system running slower than ideal.")
    else:
        notes.append("🟢 Response time within optimal range.")

    # --- Expected Value Classification (GEN‑Safe Logic) ---
    # Instead of treating low values as collapse, classify them normally.
    if ev > 1.2:
        notes.append("🟢 System showing strong positive trend.")
    elif 0.8 < ev <= 1.2:
        notes.append("🟡 System stable with mild fluctuations.")
    elif 0.2 < ev <= 0.8:
        issues.append("⚠️ System trending downward — monitor behaviour.")
    else:
        issues.append("🔍 Low expected value detected — indicates stress but not systemic failure.")

    # --- Shock Frequency Classification ---
    if sf > 0.20:
        issues.append("⛔ Excessive shock frequency — unstable environment.")
    elif sf > 0.10:
        issues.append("⚠️ Elevated shock levels — conditions volatile.")
    else:
        notes.append("🟢 Shock frequency within normal bounds.")

    # Determine status
    if len(issues) == 0:
        status = "🟢 System Stable"
    elif any("⛔" in x for x in issues):
        status = "🔴 High-Risk Conditions"
    else:
        status = "🟠 Moderate Risk Detected"

    return status, issues + notes

# ---------------------------------------------------------
# APP LAYOUT — 2×2 GRID (GEN-1 | GEN-2) + SELF-AUDITOR
# ---------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

# ============================================
# GEN-1 BLOCK
# ============================================
with col1:
    st.header("GEN-1 Baseline (Pure Python)")
    iterations_g1 = st.slider("Iterations (GEN-1)", 50_000, 500_000, 200_000, step=50_000, key="iter_gen1")

    if st.button("Run GEN-1 Model"):
        start = time.time()
        exp_val, shock_freq = mc_markov_python(iterations_g1)
        end = time.time()

        st.session_state["GEN1"] = {
            "engine": "GEN-1 (Python Loop)",
            "iterations": iterations_g1,
            "response_time_ms": (end - start) * 1000,
            "expected_value": exp_val,
            "shock_frequency": shock_freq,
        }

    if "GEN1" in st.session_state:
        g1 = st.session_state["GEN1"]
        st.metric("Response Time (ms)", f"{g1['response_time_ms']:.2f}")
        st.metric("Expected Value", f"{g1['expected_value']:.4f}")
        st.metric("Shock Frequency", f"{g1['shock_frequency']:.4f}")

# ============================================
# GEN-2 BLOCK (NUMPY VECTORIZED)
# ============================================
with col2:
    st.header("GEN-2 Accelerated (NumPy Vectorized)")
    iterations_g2 = st.slider("Iterations (GEN-2)", 50_000, 500_000, 200_000, step=50_000, key="iter_gen2")

    if st.button("Run GEN-2 Model"):
        start = time.time()
        exp_val, shock_freq = mc_markov_vectorized(iterations_g2)
        end = time.time()

        st.session_state["GEN2"] = {
            "engine": "GEN-2 (NumPy Vectorized)",
            "iterations": iterations_g2,
            "response_time_ms": (end - start) * 1000,
            "expected_value": exp_val,
            "shock_frequency": shock_freq,
        }

    if "GEN2" in st.session_state:
        g2 = st.session_state["GEN2"]
        st.metric("Response Time (ms)", f"{g2['response_time_ms']:.2f}")
        st.metric("Expected Value", f"{g2['expected_value']:.4f}")
        st.metric("Shock Frequency", f"{g2['shock_frequency']:.4f}")

        # Optional: show speedup vs GEN-1
        if "GEN1" in st.session_state:
            g1 = st.session_state["GEN1"]
            if g2["response_time_ms"] > 0:
                speedup = g1["response_time_ms"] / g2["response_time_ms"]
                st.write(f"⚡ Approx. speedup over GEN-1: **{speedup:.2f}×**")

# ---------------------------------------------------------
# SELF-AUDITOR SECTION
# ---------------------------------------------------------
st.markdown("---")
st.header("🛡️ System Self-Auditor")

# Prefer auditing GEN-2, fallback to GEN-1
if "GEN2" in st.session_state:
    last = st.session_state["GEN2"]
elif "GEN1" in st.session_state:
    last = st.session_state["GEN1"]
else:
    st.info("Run GEN-1 or GEN-2 to activate the Self-Auditor.")
    last = None

if last:
    status, issues = self_auditor_check(last)
    st.subheader("Auditor Status")
    st.write(status)

    if issues:
        for issue in issues:
            st.error(issue)
    else:
        st.success("No issues detected.")

    st.subheader("Audit Input Snapshot")
    st.json(last)

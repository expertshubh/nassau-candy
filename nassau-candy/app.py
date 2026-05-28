import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from preprocess import (
    load_and_clean, get_feature_matrix,
    PRODUCT_FACTORY, FACTORY_COORDS, REGION_COORDS
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy — Factory Optimizer",
    page_icon="🍬",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0f0f1a; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #f9c74f; font-family: 'Georgia', serif; }
    h2, h3 { color: #90e0ef; }
    .metric-card {
        background: #1a1a2e; border-radius: 12px;
        padding: 1rem 1.2rem; border-left: 4px solid #f9c74f;
        margin-bottom: 0.5rem;
    }
    .recommend-card {
        background: #16213e; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #43d17a;
        margin-bottom: 0.6rem;
    }
    .warn-card {
        background: #2a1a1a; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #ef476f;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & data ──────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model        = joblib.load(os.path.join("models", "best_model.pkl"))
    encoders     = joblib.load(os.path.join("models", "encoders.pkl"))
    feature_cols = joblib.load(os.path.join("models", "feature_cols.pkl"))
    return model, encoders, feature_cols

@st.cache_data
def load_data():
    df, _ = load_and_clean(os.path.join("data", "Nassau_Candy_Distributor.csv"))
    return df

model, encoders, feature_cols = load_assets()
df = load_data()

ALL_FACTORIES = list(FACTORY_COORDS.keys())
ALL_PRODUCTS  = list(PRODUCT_FACTORY.keys())
ALL_REGIONS   = list(REGION_COORDS.keys())
ALL_MODES     = ["Standard Class", "Second Class", "First Class", "Same Day"]

# ── Helper: predict lead time ──────────────────────────────────────────────
def predict_lead_time(product, factory, region, ship_mode, units=2, cost=3.0):
    def enc(col, val):
        le = encoders[col]
        if val in le.classes_:
            return le.transform([val])[0]
        return 0

    f_lat, f_lon = FACTORY_COORDS[factory]
    r_lat, r_lon = REGION_COORDS[region]
    R = 3958.8
    def rad(x): return x * math.pi / 180
    dlat = rad(r_lat - f_lat); dlon = rad(r_lon - f_lon)
    a = math.sin(dlat/2)**2 + math.cos(rad(f_lat))*math.cos(rad(r_lat))*math.sin(dlon/2)**2
    dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    row = {
        "Product Name_enc":          enc("Product Name", product),
        "Factory_enc":               enc("Factory", factory),
        "Region_enc":                enc("Region", region),
        "Ship Mode_enc":             enc("Ship Mode", ship_mode),
        "Shipping Distance (miles)": dist,
        "Units":                     units,
        "Cost":                      cost,
    }
    X = pd.DataFrame([row])[feature_cols]
    return round(float(model.predict(X)[0]), 1), round(dist, 0)


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("# 🍬 Nassau Candy — Factory Optimization Dashboard")
st.markdown("*Predict shipping lead times · Simulate reassignments · Get ranked recommendations*")
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎛️ Filters")
    sel_product   = st.selectbox("Product",   ALL_PRODUCTS, index=0)
    sel_region    = st.selectbox("Region",    ALL_REGIONS,  index=0)
    sel_ship_mode = st.selectbox("Ship Mode", ALL_MODES,    index=0)
    sel_units     = st.slider("Units per Order", 1, 20, 2)
    st.divider()
    st.markdown("### ⚖️ Optimization Priority")
    priority = st.slider("Speed  ◀──▶  Profit", 0, 100, 50,
                         help="0 = prioritize fastest shipping, 100 = prioritize profit")

current_factory = PRODUCT_FACTORY[sel_product]

# ── KPI row ────────────────────────────────────────────────────────────────
avg_cost    = df[df["Product Name"] == sel_product]["Cost"].mean()
avg_profit  = df[df["Product Name"] == sel_product]["Gross Profit"].mean()
avg_sales   = df[df["Product Name"] == sel_product]["Sales"].mean()
curr_lt, curr_dist = predict_lead_time(sel_product, current_factory,
                                       sel_region, sel_ship_mode,
                                       sel_units, avg_cost)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Current Factory",        current_factory)
k2.metric("Predicted Lead Time",    f"{curr_lt} days")
k3.metric("Avg Gross Profit / Order", f"${avg_profit:.2f}")
k4.metric("Shipping Distance",      f"{curr_dist:,.0f} mi")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏭 Factory Simulator",
    "🔄 What-If Analysis",
    "🏆 Recommendations",
    "⚠️ Risk & Impact",
])

# ── TAB 1: Factory Simulator ───────────────────────────────────────────────
with tab1:
    st.subheader(f"Lead Time Across All Factories — {sel_product}")
    st.caption(f"Region: {sel_region}  |  Ship Mode: {sel_ship_mode}")

    rows = []
    for fac in ALL_FACTORIES:
        lt, dist = predict_lead_time(sel_product, fac, sel_region,
                                     sel_ship_mode, sel_units, avg_cost)
        rows.append({"Factory": fac, "Predicted Lead Time (days)": lt,
                     "Distance (miles)": dist,
                     "Current": "✅ Current" if fac == current_factory else ""})

    sim_df = pd.DataFrame(rows).sort_values("Predicted Lead Time (days)")

    colors = ["#f9c74f" if r == "✅ Current" else "#90e0ef"
              for r in sim_df["Current"]]

    fig = go.Figure(go.Bar(
        x=sim_df["Factory"],
        y=sim_df["Predicted Lead Time (days)"],
        marker_color=colors,
        text=sim_df["Predicted Lead Time (days)"].apply(lambda x: f"{x}d"),
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
        font_color="white", height=380,
        xaxis_title="Factory", yaxis_title="Lead Time (days)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sim_df.set_index("Factory"), use_container_width=True)


# ── TAB 2: What-If Analysis ────────────────────────────────────────────────
with tab2:
    st.subheader("Compare Current vs Alternative Factory")

    col_l, col_r = st.columns(2)
    with col_l:
        alt_factory = st.selectbox("Alternative Factory",
                                   [f for f in ALL_FACTORIES if f != current_factory])
    with col_r:
        alt_region  = st.selectbox("Destination Region", ALL_REGIONS,
                                   index=ALL_REGIONS.index(sel_region))

    alt_lt, alt_dist = predict_lead_time(sel_product, alt_factory,
                                         alt_region, sel_ship_mode,
                                         sel_units, avg_cost)
    delta_lt   = curr_lt - alt_lt
    delta_dist = curr_dist - alt_dist

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Lead Time",     f"{curr_lt} days")
    c2.metric("Alternative Lead Time", f"{alt_lt} days",
              delta=f"{-delta_lt:.1f} days", delta_color="inverse")
    c3.metric("Distance Change",       f"{alt_dist:,.0f} mi",
              delta=f"{-delta_dist:,.0f} mi", delta_color="inverse")

    # Comparison bar chart
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Current",     x=["Lead Time (days)", "Distance (miles)"],
                          y=[curr_lt, curr_dist], marker_color="#f9c74f"))
    fig2.add_trace(go.Bar(name="Alternative", x=["Lead Time (days)", "Distance (miles)"],
                          y=[alt_lt,  alt_dist],  marker_color="#90e0ef"))
    fig2.update_layout(barmode="group", paper_bgcolor="#0f0f1a",
                       plot_bgcolor="#0f0f1a", font_color="white", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    if delta_lt > 0:
        st.success(f"✅ Switching to **{alt_factory}** saves **{delta_lt:.1f} days** of lead time!")
    elif delta_lt < 0:
        st.warning(f"⚠️ Switching to **{alt_factory}** adds **{abs(delta_lt):.1f} days** of lead time.")
    else:
        st.info("No change in lead time.")


# ── TAB 3: Recommendations ─────────────────────────────────────────────────
with tab3:
    st.subheader("🏆 Top Factory Reassignment Recommendations")
    st.caption("Ranked by lead time improvement + profit stability")

    all_recs = []
    for product in ALL_PRODUCTS:
        cur_fac  = PRODUCT_FACTORY[product]
        prod_cost = df[df["Product Name"] == product]["Cost"].mean()
        prod_gp   = df[df["Product Name"] == product]["Gross Profit"].mean()
        cur_lt, _ = predict_lead_time(product, cur_fac, sel_region,
                                      sel_ship_mode, sel_units, prod_cost)
        for fac in ALL_FACTORIES:
            if fac == cur_fac:
                continue
            new_lt, new_dist = predict_lead_time(product, fac, sel_region,
                                                 sel_ship_mode, sel_units, prod_cost)
            improvement = cur_lt - new_lt
            # Blend speed & profit based on priority slider
            score = (improvement * (1 - priority/100)) + (prod_gp * (priority/100) * 0.1)
            all_recs.append({
                "Product":          product,
                "From Factory":     cur_fac,
                "To Factory":       fac,
                "Lead Time Saving": round(improvement, 1),
                "Gross Profit":     round(prod_gp, 2),
                "Score":            round(score, 2),
                "New Distance (mi)":int(new_dist),
            })

    rec_df = pd.DataFrame(all_recs).sort_values("Score", ascending=False).head(10)
    rec_df.index = range(1, len(rec_df)+1)

    for _, row in rec_df.iterrows():
        badge = "🟢" if row["Lead Time Saving"] > 0 else "🔴"
        st.markdown(f"""
        <div class="recommend-card">
        {badge} <b>{row['Product']}</b><br>
        Move from <b>{row['From Factory']}</b> → <b>{row['To Factory']}</b><br>
        Lead Time Saving: <b>{row['Lead Time Saving']} days</b> &nbsp;|&nbsp;
        Avg Gross Profit: <b>${row['Gross Profit']}</b> &nbsp;|&nbsp;
        Score: <b>{row['Score']}</b>
        </div>
        """, unsafe_allow_html=True)

    st.download_button("📥 Download Recommendations CSV",
                       rec_df.to_csv(index=False),
                       "recommendations.csv", "text/csv")


# ── TAB 4: Risk & Impact ───────────────────────────────────────────────────
with tab4:
    st.subheader("⚠️ Risk & Profit Impact Panel")

    # Profit distribution per product
    prod_stats = df.groupby("Product Name").agg(
        Avg_GP=("Gross Profit", "mean"),
        Std_GP=("Gross Profit", "std"),
        Orders=("Order ID", "count"),
    ).reset_index()
    prod_stats["Risk"] = prod_stats["Std_GP"] / prod_stats["Avg_GP"]

    fig3 = px.scatter(
        prod_stats, x="Avg_GP", y="Risk",
        size="Orders", color="Risk",
        hover_name="Product Name",
        color_continuous_scale="RdYlGn_r",
        labels={"Avg_GP": "Avg Gross Profit ($)", "Risk": "Profit Volatility"},
        title="Profit Volatility vs Average Gross Profit per Product",
    )
    fig3.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                       font_color="white", height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # High-risk warnings
    st.markdown("#### 🚨 High-Risk Reassignment Alerts")
    high_risk = prod_stats[prod_stats["Risk"] > prod_stats["Risk"].median()]
    for _, row in high_risk.iterrows():
        st.markdown(f"""
        <div class="warn-card">
        ⚠️ <b>{row['Product Name']}</b> has high profit volatility (risk score: {row['Risk']:.2f}).
        Reassigning this product carries financial risk — review carefully before switching factories.
        </div>
        """, unsafe_allow_html=True)

    # Model performance
    st.markdown("#### 📊 Model Performance Summary")
    model_results_path = os.path.join("models", "model_results.csv")
    if os.path.exists(model_results_path):
        mr = pd.read_csv(model_results_path)
        st.dataframe(mr, use_container_width=True)
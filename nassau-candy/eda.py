import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from preprocess import load_and_clean, PRODUCT_FACTORY

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy — EDA Report",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .main { background: #0a0a0f; }
    h1 { color: #f9c74f; }
    h2, h3 { color: #90e0ef; }
    .insight-box {
        background: #1a1a2e; border-radius: 10px;
        padding: 1rem 1.4rem; border-left: 4px solid #f9c74f;
        margin: 0.5rem 0 1rem 0; color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df, _ = load_and_clean(os.path.join("data", "Nassau_Candy_Distributor.csv"))
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)
    return df

df = load()

st.markdown("# 📊 Nassau Candy Distributor — EDA Report")
st.markdown("*Exploratory Data Analysis of 10,194 orders across 15 products and 5 factories*")
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## 1. Dataset Overview")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Orders",    f"{len(df):,}")
c2.metric("Total Revenue",   f"${df['Sales'].sum():,.0f}")
c3.metric("Total Profit",    f"${df['Gross Profit'].sum():,.0f}")
c4.metric("Unique Products", df["Product Name"].nunique())
c5.metric("Avg Lead Time",   f"{df['Lead Time'].mean():.0f} days")

st.markdown("### Sample Data")
st.dataframe(df[["Order ID","Order Date","Ship Date","Lead Time",
                  "Product Name","Factory","Region","Ship Mode",
                  "Sales","Gross Profit"]].head(10), use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <b>Key Insight:</b> The dataset contains 10,194 orders spanning 15 unique Wonka products
across 5 factories and 4 US regions. Average lead time is extremely high (~1,320 days),
suggesting significant shipping inefficiencies that this project aims to address.
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: LEAD TIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## 2. Lead Time Analysis")

col1, col2 = st.columns(2)

with col1:
    # Lead time by ship mode
    lt_mode = df.groupby("Ship Mode")["Lead Time"].mean().reset_index()
    lt_mode.columns = ["Ship Mode", "Avg Lead Time (days)"]
    lt_mode = lt_mode.sort_values("Avg Lead Time (days)", ascending=False)
    fig = px.bar(lt_mode, x="Ship Mode", y="Avg Lead Time (days)",
                 color="Avg Lead Time (days)", color_continuous_scale="YlOrRd",
                 title="Average Lead Time by Ship Mode")
    fig.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Lead time by region
    lt_region = df.groupby("Region")["Lead Time"].mean().reset_index()
    lt_region.columns = ["Region", "Avg Lead Time (days)"]
    lt_region = lt_region.sort_values("Avg Lead Time (days)", ascending=False)
    fig2 = px.bar(lt_region, x="Region", y="Avg Lead Time (days)",
                  color="Avg Lead Time (days)", color_continuous_scale="Blues",
                  title="Average Lead Time by Region")
    fig2.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", font_color="white")
    st.plotly_chart(fig2, use_container_width=True)

# Lead time by factory
lt_factory = df.groupby("Factory")["Lead Time"].mean().reset_index()
lt_factory.columns = ["Factory", "Avg Lead Time (days)"]
lt_factory = lt_factory.sort_values("Avg Lead Time (days)")
fig3 = px.bar(lt_factory, x="Factory", y="Avg Lead Time (days)",
              color="Avg Lead Time (days)", color_continuous_scale="RdYlGn_r",
              title="Average Lead Time by Factory")
fig3.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", font_color="white", height=350)
st.plotly_chart(fig3, use_container_width=True)

best_factory  = lt_factory.iloc[0]["Factory"]
worst_factory = lt_factory.iloc[-1]["Factory"]
st.markdown(f"""
<div class="insight-box">
💡 <b>Key Insight:</b> <b>{best_factory}</b> has the shortest average lead time while
<b>{worst_factory}</b> has the longest. Reassigning products from slow factories to faster
ones could significantly reduce shipping delays.
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: SALES & PROFIT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## 3. Sales & Profit Analysis")

col3, col4 = st.columns(2)

with col3:
    # Sales by division
    sales_div = df.groupby("Division")["Sales"].sum().reset_index()
    fig4 = px.pie(sales_div, names="Division", values="Sales",
                  title="Total Sales by Division",
                  color_discrete_sequence=["#f9c74f","#90e0ef","#ef476f"])
    fig4.update_layout(paper_bgcolor="#0f0f1a", font_color="white")
    st.plotly_chart(fig4, use_container_width=True)

with col4:
    # Profit by product (top 10)
    profit_prod = df.groupby("Product Name")["Gross Profit"].sum().reset_index()
    profit_prod = profit_prod.sort_values("Gross Profit", ascending=True).tail(10)
    fig5 = px.bar(profit_prod, x="Gross Profit", y="Product Name",
                  orientation="h", title="Total Gross Profit by Product (Top 10)",
                  color="Gross Profit", color_continuous_scale="Greens")
    fig5.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", font_color="white")
    st.plotly_chart(fig5, use_container_width=True)

# Sales by region
sales_region = df.groupby("Region")["Sales"].sum().reset_index()
sales_region = sales_region.sort_values("Sales", ascending=False)
fig6 = px.bar(sales_region, x="Region", y="Sales",
              color="Sales", color_continuous_scale="Viridis",
              title="Total Sales by Region")
fig6.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", font_color="white")
st.plotly_chart(fig6, use_container_width=True)

top_product = df.groupby("Product Name")["Gross Profit"].sum().idxmax()
top_region  = sales_region.iloc[0]["Region"]
st.markdown(f"""
<div class="insight-box">
💡 <b>Key Insight:</b> <b>{top_product}</b> generates the most gross profit.
The <b>{top_region}</b> region drives the highest total sales, making it a
priority region for shipping optimization.
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: FACTORY PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## 4. Factory Performance")

factory_stats = df.groupby("Factory").agg(
    Orders       =("Order ID",    "count"),
    Avg_Lead_Time=("Lead Time",   "mean"),
    Total_Sales  =("Sales",       "sum"),
    Total_Profit =("Gross Profit","sum"),
    Avg_Cost     =("Cost",        "mean"),
).reset_index().round(2)
factory_stats.columns = ["Factory","Orders","Avg Lead Time","Total Sales","Total Profit","Avg Cost"]

fig7 = px.scatter(factory_stats,
                  x="Avg Lead Time", y="Total Profit",
                  size="Orders", color="Factory",
                  hover_name="Factory",
                  title="Factory: Lead Time vs Total Profit (bubble = order volume)",
                  size_max=60)
fig7.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                   font_color="white", height=420)
st.plotly_chart(fig7, use_container_width=True)
st.dataframe(factory_stats, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: SHIPPING DISTANCE VS LEAD TIME
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## 5. Shipping Distance vs Lead Time")

fig8 = px.scatter(df.sample(2000, random_state=42),
                  x="Shipping Distance (miles)", y="Lead Time",
                  color="Ship Mode", opacity=0.5,
                  title="Shipping Distance vs Lead Time (sample of 2,000 orders)",
                  trendline="ols")
fig8.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                   font_color="white", height=420)
st.plotly_chart(fig8, use_container_width=True)

corr = df[["Shipping Distance (miles)","Lead Time"]].corr().iloc[0,1]
st.markdown(f"""
<div class="insight-box">
💡 <b>Key Insight:</b> Correlation between shipping distance and lead time = <b>{corr:.3f}</b>.
{"A positive correlation confirms that longer distances lead to longer shipping times — supporting the case for factory reassignment." if corr > 0 else "The weak correlation suggests lead time is affected by other factors beyond distance alone."}
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: PRODUCT-REGION HEATMAP
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## 6. Product–Region Lead Time Heatmap")

heatmap_data = df.groupby(["Product Name","Region"])["Lead Time"].mean().reset_index()
heatmap_pivot = heatmap_data.pivot(index="Product Name", columns="Region", values="Lead Time")

fig9 = px.imshow(heatmap_pivot,
                 color_continuous_scale="RdYlGn_r",
                 title="Avg Lead Time: Product × Region (red = slow, green = fast)",
                 aspect="auto")
fig9.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                   font_color="white", height=500)
st.plotly_chart(fig9, use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <b>Key Insight:</b> The heatmap reveals specific product-region combinations with
consistently high lead times — these are the highest-priority candidates for factory
reassignment in the optimization engine.
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## 7. ML Model Performance")

model_path = os.path.join("models", "model_results.csv")
if os.path.exists(model_path):
    mr = pd.read_csv(model_path)
    fig10 = px.bar(mr, x="Model", y="RMSE", color="Model",
                   title="Model Comparison — RMSE (lower is better)",
                   color_discrete_sequence=["#f9c74f","#90e0ef","#ef476f"])
    fig10.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                        font_color="white")
    st.plotly_chart(fig10, use_container_width=True)
    st.dataframe(mr, use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <b>Key Insight:</b> Linear Regression performed best (lowest RMSE), indicating that
lead time has a largely linear relationship with the input features — particularly
shipping distance and ship mode.
</div>
""", unsafe_allow_html=True)

st.divider()
st.markdown("*EDA Report — Nassau Candy Distributor Factory Optimization Project*")
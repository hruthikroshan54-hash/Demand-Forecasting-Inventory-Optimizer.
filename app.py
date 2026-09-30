"""Demand Forecasting & Inventory Optimizer: Streamlit app."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from src.data import load_sample_data, validate_sales_data
from src.forecast import backtest, forecast_future, prepare_sku_frame
from src.inventory import inventory_plan

st.set_page_config(page_title="Demand Forecasting & Inventory Optimizer", layout="wide")


# ---------- cached compute ----------
@st.cache_data(show_spinner=False)
def run_sku(df: pd.DataFrame, sku: str, horizon: int):
    frame = prepare_sku_frame(df, sku)
    bt = backtest(frame, horizon=horizon)
    fc = forecast_future(frame, horizon)
    return frame, bt, fc


@st.cache_data(show_spinner=False)
def read_upload(file_bytes: bytes) -> pd.DataFrame:
    import io
    return validate_sales_data(pd.read_csv(io.BytesIO(file_bytes)))


# ---------- sidebar ----------
st.sidebar.title("Settings")
source = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"])

if source == "Upload CSV":
    up = st.sidebar.file_uploader("CSV with columns: date, sku, units (optional: promo)", type="csv")
    if up is None:
        st.info("Upload a CSV with columns `date, sku, units` (optional `promo`) or switch to Sample data.")
        st.stop()
    try:
        data = read_upload(up.getvalue())
    except Exception as e:  # readable message instead of a stack trace
        st.error(f"Could not read the file: {e}")
        st.stop()
else:
    data = load_sample_data()

sku = st.sidebar.selectbox("Product (SKU)", sorted(data["sku"].unique()))
horizon = st.sidebar.slider("Forecast horizon (days)", 14, 90, 28, step=7)

st.sidebar.subheader("Inventory parameters")
lead_time = st.sidebar.number_input("Supplier lead time (days)", 1, 60, 7)
service_pct = st.sidebar.slider("Target service level (%)", 85.0, 99.9, 95.0, step=0.5)
order_cost = st.sidebar.number_input("Cost per order (₹)", 1.0, 100000.0, 500.0, step=50.0)
unit_cost = st.sidebar.number_input("Unit cost (₹)", 0.1, 100000.0, 40.0, step=1.0)
holding_pct = st.sidebar.slider("Annual holding cost (% of unit cost)", 5, 50, 20)
on_hand = st.sidebar.number_input("Current stock on hand (units)", 0, 1_000_000, 800, step=50)

# ---------- compute ----------
st.title("Demand Forecasting & Inventory Optimizer")
st.caption("ML demand forecast → safety stock, reorder point and order quantity")

try:
    with st.spinner("Training model and running backtest..."):
        frame, bt, fc = run_sku(data, sku, horizon)
except ValueError as e:
    st.error(str(e))
    st.stop()

mean_daily = float(fc["forecast"].mean())
sigma = bt["sigma"]
plan = inventory_plan(mean_daily, sigma, lead_time, service_pct / 100, order_cost, unit_cost, holding_pct / 100)

# ---------- KPI row ----------
best = bt["metrics"].iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Forecast demand ({horizon}d)", f"{fc['forecast'].sum():,.0f} units")
k2.metric("Safety stock", f"{plan['safety_stock']:,.0f} units")
k3.metric("Reorder point", f"{plan['reorder_point']:,.0f} units")
k4.metric("Order quantity (EOQ)", f"{plan['eoq']:,.0f} units")

if on_hand <= plan["reorder_point"]:
    st.error(f"Reorder now: stock ({on_hand:,}) is at or below the reorder point "
             f"({plan['reorder_point']:,.0f}). Suggested order: {plan['eoq']:,.0f} units.")
else:
    days_left = (on_hand - plan["reorder_point"]) / max(mean_daily, 1e-9)
    st.success(f"Stock is healthy. Reorder point reached in about {days_left:,.0f} days at the forecast demand rate.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Forecast", "Inventory plan", "Model comparison", "All products", "About"])

# ---------- Forecast ----------
with tab1:
    z80 = norm.ppf(0.90)
    hist = frame.tail(120)
    upper = fc["forecast"] + z80 * sigma
    lower = (fc["forecast"] - z80 * sigma).clip(lower=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["units"], name="Actual", line=dict(color="#64748b")))
    fig.add_trace(go.Scatter(x=fc["date"], y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["date"], y=lower, fill="tonexty", fillcolor="rgba(37,99,235,0.15)",
                             line=dict(width=0), name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], name="Forecast", line=dict(color="#2563eb", width=2.5)))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Units per day",
                      legend=dict(orientation="h", y=1.08), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("The shaded range is approximate: it uses the average forecast error from the backtest. "
               "Future promotions are assumed off.")
    st.download_button("Download forecast CSV", fc.to_csv(index=False), file_name=f"forecast_{horizon}d.csv")

# ---------- Inventory ----------
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Policy")
        st.table(pd.DataFrame({
            "Metric": ["Average daily demand", "Forecast error (std/day)", "Safety stock", "Reorder point",
                       "Economic order quantity", "Orders per year", "Days between orders"],
            "Value": [f"{mean_daily:,.1f} units", f"{sigma:,.1f} units", f"{plan['safety_stock']:,.0f} units",
                      f"{plan['reorder_point']:,.0f} units", f"{plan['eoq']:,.0f} units",
                      f"{plan['orders_per_year']:,.1f}", f"{plan['cycle_days']:,.1f}"],
        }).set_index("Metric"))
    with c2:
        st.subheader("Estimated annual cost")
        total = plan["annual_ordering_cost"] + plan["annual_holding_cost"]
        st.metric("Ordering + holding cost", f"₹{total:,.0f}")
        st.write(f"Ordering: ₹{plan['annual_ordering_cost']:,.0f}  \nHolding: ₹{plan['annual_holding_cost']:,.0f}")
        st.markdown(
            "**Formulas**\n\n"
            "- Safety stock = z × σ × √lead time\n"
            "- Reorder point = daily demand × lead time + safety stock\n"
            "- EOQ = √(2 × annual demand × order cost / holding cost per unit)"
        )

# ---------- Model comparison ----------
with tab3:
    st.subheader(f"Rolling-origin backtest ({bt['folds']} × {horizon}-day windows)")
    st.dataframe(bt["metrics"].round(2), hide_index=True, width="stretch")
    st.caption("WAPE = total absolute error / total actual demand (lower is better).")
    last = bt["predictions"][bt["predictions"]["fold"] == bt["folds"]]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=last["date"], y=last["actual"], name="Actual", line=dict(color="#0f172a", width=2)))
    fig2.add_trace(go.Scatter(x=last["date"], y=last["gbm"], name="Gradient Boosting", line=dict(color="#2563eb")))
    fig2.add_trace(go.Scatter(x=last["date"], y=last["seasonal_naive"], name="Seasonal Naive",
                              line=dict(color="#f59e0b", dash="dot")))
    fig2.add_trace(go.Scatter(x=last["date"], y=last["moving_avg"], name="Moving Avg",
                              line=dict(color="#94a3b8", dash="dash")))
    fig2.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Units per day",
                       legend=dict(orientation="h", y=1.1), hovermode="x unified")
    st.plotly_chart(fig2, width="stretch")

# ---------- All products ----------
with tab4:
    st.subheader("Inventory plan for every product")
    rows = []
    for s in sorted(data["sku"].unique()):
        try:
            _, b, f = run_sku(data, s, horizon)
        except ValueError:
            continue
        md = float(f["forecast"].mean())
        p = inventory_plan(md, b["sigma"], lead_time, service_pct / 100, order_cost, unit_cost, holding_pct / 100)
        rows.append({"Product": s, f"Forecast {horizon}d (units)": round(f["forecast"].sum()),
                     "Avg daily": round(md, 1), "Safety stock": round(p["safety_stock"]),
                     "Reorder point": round(p["reorder_point"]), "EOQ": round(p["eoq"]),
                     "Backtest WAPE %": round(b["metrics"].set_index("Model").loc["Gradient Boosting (ML)", "WAPE %"], 1)})
    if rows:
        table = pd.DataFrame(rows)
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Download plan CSV", table.to_csv(index=False), file_name="inventory_plan.csv")
    else:
        st.warning("No product has enough history for the chosen horizon.")

# ---------- About ----------
with tab5:
    st.markdown("""
**How it works**

1. Daily sales are turned into features: day of week, month, promotions, lagged sales (7/14/28 days) and rolling averages.
2. A gradient boosting model (scikit-learn `HistGradientBoostingRegressor`) forecasts day by day.
3. It is compared with two baselines using a rolling-origin backtest, which is the honest way to test a forecast.
4. The forecast and its error feed classic inventory maths: safety stock, reorder point and EOQ.

**Using your own data:** upload a CSV with `date, sku, units` (optional `promo` as 0/1).
Each SKU needs at least 120 days of history plus the forecast horizon.

**Limits:** forecasts get less certain the further out they go; future promotions and price changes are not modelled;
the safety stock assumes lead time is constant and forecast errors are roughly normal.
""")

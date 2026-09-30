# Demand Forecasting & Inventory Optimizer

An end-to-end operations analytics project: an ML model forecasts product demand, and the forecast drives inventory decisions (**safety stock, reorder point and order quantity**) in an interactive Streamlit app.

**Live demo:** _add your Streamlit link here after deploying_

![CI](https://github.com/YOUR-USERNAME/demand-forecasting-inventory/actions/workflows/ci.yml/badge.svg)

<!-- Add a screenshot: save it as docs/screenshot.png and uncomment the next line -->
<!-- ![App screenshot](docs/screenshot.png) -->

## Business problem

Retailers lose money two ways: stock-outs (lost sales) and overstock (cash tied up, wastage). Both come from poor demand estimates. This project answers three questions for each product:

1. How much will we sell over the next 2-12 weeks?
2. How much safety stock do we need for a target service level?
3. When should we reorder, and how much?

## Approach

| Step | What it does | Tools |
|---|---|---|
| Features | Day of week, month, promotions, lagged sales (7/14/28 days), rolling averages | pandas |
| Model | Gradient boosting forecasts day by day, feeding predictions back in | scikit-learn `HistGradientBoostingRegressor` |
| Validation | Rolling-origin backtest against two baselines (seasonal naive, moving average) | MAE, RMSE, WAPE |
| Inventory logic | Forecast error sets safety stock; reorder point and EOQ follow | SciPy, standard formulas |
| App | Forecast chart, inventory plan, model comparison, all-products table, CSV export | Streamlit, Plotly |

**Formulas**

- Safety stock = z × σ × √(lead time), where σ is the backtest forecast error per day
- Reorder point = average daily demand × lead time + safety stock
- EOQ = √(2 × annual demand × order cost ÷ holding cost per unit)

## Results

Average over 5 products, 28-day horizon, 3 rolling backtest windows (lower is better):

| Model | MAE | RMSE | WAPE % |
|---|---|---|---|
| **Gradient Boosting (ML)** | 14.33 | 18.94 | **12.60** |
| 28-day Moving Average | 16.52 | 20.66 | 14.19 |
| Seasonal Naive | 19.03 | 24.41 | 16.81 |

The ML model cuts forecast error by about 11% versus the best baseline. Note that the included data is **synthetic**, so treat these numbers as a demonstration of the method, not a real-world claim. Rerun `python scripts/evaluate.py` to reproduce, or use real data (below) and report those results instead.

## Run locally

```bash
git clone https://github.com/YOUR-USERNAME/demand-forecasting-inventory.git
cd demand-forecasting-inventory
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run the tests: `pip install -r requirements-dev.txt && pytest -q`

## Use your own data

Upload a CSV in the app sidebar with these columns:

| Column | Required | Example |
|---|---|---|
| `date` | yes | 2025-03-14 |
| `sku` | yes | Store12-Milk |
| `units` | yes | 84 |
| `promo` | no | 0 or 1 |

Each SKU needs at least 120 days of history plus the forecast horizon. Good public datasets: Kaggle **Rossmann Store Sales**, **Walmart Store Sales Forecasting**, **Store Item Demand Forecasting**. Rename their columns to match the table above.

## Project structure

```
app.py                  Streamlit app
src/data.py             Loading, validation, synthetic data generator
src/forecast.py         Features, model, baselines, backtest
src/inventory.py        Safety stock, reorder point, EOQ
scripts/evaluate.py     Backtest all SKUs, writes results/
tests/                  Unit tests + app smoke test
data/sample_sales.csv   Synthetic demo data (regenerate: python -m src.data)
```

## Limitations and next steps

- Future promotions and price changes are not modelled (promos assumed off in the forecast).
- Lead time is treated as constant; a fuller model would include supplier variability.
- Forecast uncertainty is approximated from backtest error, not from a probabilistic model.
- Ideas: add holiday features, quantile regression for prediction intervals, multi-echelon inventory, an ABC-XYZ classification tab.

## Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub (public).
2. Go to share.streamlit.io, sign in with GitHub, click **Create app**.
3. Select the repo, branch `main`, and main file `app.py`, then **Deploy**.
4. Paste the live URL at the top of this README.

## License

MIT

"""
Generates three practice datasets that mirror the column schema of popular public datasets:

  1. data/ecommerce_churn.csv   -> schema of Kaggle "E-Commerce Customer Churn" (E Commerce Dataset.xlsx)
  2. data/store_sales.csv       -> schema of Kaggle "Store Item Demand Forecasting Challenge" (train.csv)
  3. data/paysim_transactions.csv -> schema of Kaggle "PaySim Synthetic Financial Dataset" (PS_20174392719_1491204439457_Log.csv)

These are SYNTHETIC. To use the real datasets, download them from Kaggle, save them in /data
with the file names above (CSV), and re-run the notebooks - the column names are identical.

Run:  python src/generate_data.py
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)


# ----------------------------------------------------------------------------------------------
# 1. E-commerce churn
# ----------------------------------------------------------------------------------------------
def make_churn(n=5630, seed=42):
    rng = np.random.default_rng(seed)
    tenure = np.round(rng.gamma(1.1, 9.0, n)).clip(0, 61)
    device = rng.choice(["Mobile Phone", "Computer", "Phone"], n, p=[0.49, 0.29, 0.22])
    tier = rng.choice([1, 2, 3], n, p=[0.65, 0.04, 0.31])
    wh = rng.integers(5, 37, n).astype(float)
    out = rng.random(n) < 0.002
    wh[out] = rng.choice([126, 127], out.sum())
    pay = rng.choice(["Debit Card", "UPI", "CC", "Cash on Delivery", "E wallet", "COD", "Credit Card"], n,
                     p=np.array([2314, 414, 273, 514, 614, 365, 1501]) / 5995)
    gender = rng.choice(["Male", "Female"], n, p=[0.6, 0.4])
    hours = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.002, 0.10, 0.37, 0.43, 0.09, 0.008]).astype(float)
    ndev = rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.03, 0.10, 0.30, 0.40, 0.14, 0.03])
    cat = rng.choice(["Laptop & Accessory", "Mobile Phone", "Fashion", "Mobile", "Grocery", "Others"], n,
                     p=np.array([2050, 1271, 826, 809, 410, 264]) / 5630)
    sat = rng.choice([1, 2, 3, 4, 5], n, p=np.array([1164, 586, 1351, 1074, 1455]) / 5630)
    marital = rng.choice(["Married", "Single", "Divorced"], n, p=[0.53, 0.31, 0.16])
    addr = (1 + rng.poisson(2.2, n)).clip(1, 22)
    complain = (rng.random(n) < 0.28).astype(int)
    hike = rng.integers(11, 27, n).astype(float)
    coupon = rng.poisson(1.7, n).clip(0, 16).astype(float)
    orders = (1 + rng.poisson(2.0, n)).clip(1, 16).astype(float)
    dso = rng.poisson(4.5, n).clip(0, 46).astype(float)
    cash = (rng.normal(175, 45, n) + 3 * orders).clip(100, 325).round(2)

    z = (1.5 * complain + 1.4 * (tenure <= 2) - 0.07 * np.minimum(tenure, 30) + 0.28 * (sat - 3)
         + 0.12 * (addr - 3) + 0.5 * (tier == 3) + 0.6 * (marital == "Single") - 0.4 * (marital == "Married")
         + 0.03 * (np.minimum(wh, 40) - 15) + 0.18 * (ndev - 4) - 0.012 * (cash - 175) - 0.06 * (dso - 4))
    lo, hi = -10, 10
    for _ in range(60):  # bisection on the intercept so churn rate is ~16.8 %
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > 0.168:
            hi = mid
        else:
            lo = mid
    churn = (rng.random(n) < 1 / (1 + np.exp(-(z + mid)))).astype(int)

    df = pd.DataFrame({
        "CustomerID": np.arange(50001, 50001 + n), "Churn": churn, "Tenure": tenure,
        "PreferredLoginDevice": device, "CityTier": tier, "WarehouseToHome": wh,
        "PreferredPaymentMode": pay, "Gender": gender, "HourSpendOnApp": hours,
        "NumberOfDeviceRegistered": ndev, "PreferedOrderCat": cat, "SatisfactionScore": sat,
        "MaritalStatus": marital, "NumberOfAddress": addr, "Complain": complain,
        "OrderAmountHikeFromlastYear": hike, "CouponUsed": coupon, "OrderCount": orders,
        "DaySinceLastOrder": dso, "CashbackAmount": cash})
    for col in ["Tenure", "WarehouseToHome", "HourSpendOnApp", "OrderAmountHikeFromlastYear",
                "CouponUsed", "OrderCount", "DaySinceLastOrder"]:
        df.loc[rng.random(n) < 0.045, col] = np.nan
    df.to_csv(os.path.join(DATA, "ecommerce_churn.csv"), index=False)
    print("churn:", df.shape, "churn rate = %.3f" % df.Churn.mean())


# ----------------------------------------------------------------------------------------------
# 2. Store sales
# ----------------------------------------------------------------------------------------------
def make_sales(seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", "2022-12-31", freq="D")
    t = np.arange(len(dates))
    doy = dates.dayofyear.values
    dow_f = np.array([0.85, 0.85, 0.90, 0.95, 1.10, 1.30, 1.25])[dates.dayofweek.values]
    yearly = 1 + 0.25 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    festive = np.where(((dates.month == 11) & (dates.day >= 20)) | (dates.month == 12), 1.15, 1.0)
    trend = 1 + 0.0003 * t
    promo = np.ones(len(dates))
    promo[rng.choice(len(dates), 12, replace=False)] = 1.8      # promotional spikes
    promo[rng.choice(len(dates), 6, replace=False)] = 0.3       # stock-outs / outages
    base = 40 * trend * dow_f * yearly * festive * promo
    rows = []
    for s, sm in zip([1, 2, 3], [1.0, 1.25, 0.8]):
        for i, im in zip([1, 2, 3], [1.0, 0.7, 1.4]):
            sales = np.round(base * sm * im * rng.lognormal(0, 0.08, len(dates))).astype(int)
            rows.append(pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "store": s, "item": i, "sales": sales}))
    df = pd.concat(rows, ignore_index=True).sort_values(["date", "store", "item"]).reset_index(drop=True)
    df.to_csv(os.path.join(DATA, "store_sales.csv"), index=False)
    print("sales:", df.shape)


# ----------------------------------------------------------------------------------------------
# 3. PaySim-style fraud transactions
# ----------------------------------------------------------------------------------------------
def make_fraud(n=60000, seed=11):
    rng = np.random.default_rng(seed)
    typ = rng.choice(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"], n, p=[0.34, 0.22, 0.35, 0.08, 0.01])
    fraud = np.zeros(n, dtype=int)
    fraud[(typ == "TRANSFER") & (rng.random(n) < 0.07)] = 1
    fraud[(typ == "CASH_OUT") & (rng.random(n) < 0.0125)] = 1
    f = fraud == 1

    # time: 30 days of hourly steps. Legit activity is daytime-heavy, fraud skews to the night
    day = rng.integers(0, 31, n)
    w_legit = np.array([1, 1, 1, 1, 1, 2, 3, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 7, 5, 3, 2], float)
    w_fraud = np.array([8, 8, 8, 7, 7, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 6, 7], float)
    hour = np.where(f, rng.choice(24, n, p=w_fraud / w_fraud.sum()), rng.choice(24, n, p=w_legit / w_legit.sum()))
    step = 1 + day * 24 + hour

    mu = {"PAYMENT": 8.5, "CASH_IN": 11.5, "CASH_OUT": 10.5, "TRANSFER": 11.3, "DEBIT": 8.5}
    sg = {"PAYMENT": 1.0, "CASH_IN": 0.8, "CASH_OUT": 1.2, "TRANSFER": 1.3, "DEBIT": 1.0}
    amount = np.array([rng.lognormal(mu[t], sg[t]) for t in typ])
    amount[f] = rng.lognormal(12.8, 1.0, f.sum())
    amount = amount.round(2)

    old_o = np.zeros(n); new_o = np.zeros(n)
    outgoing = np.isin(typ, ["PAYMENT", "CASH_OUT", "TRANSFER", "DEBIT"])
    for i in np.where(outgoing)[0]:
        if f[i]:
            if rng.random() < 0.8:
                old_o[i] = amount[i]                                  # full account drain
            else:
                old_o[i] = amount[i] * (1 + rng.uniform(0, 0.3))
        else:
            r = rng.random()
            if r < 0.12:
                old_o[i] = 0; new_o[i] = 0; continue                  # balance not recorded
            old_o[i] = amount[i] if r < 0.17 else amount[i] * (1 + rng.exponential(3))
        new_o[i] = max(old_o[i] - amount[i], 0)
    ci = np.where(typ == "CASH_IN")[0]
    old_o[ci] = rng.lognormal(11, 1.5, len(ci)); new_o[ci] = old_o[ci] + amount[ci]

    old_d = np.zeros(n); new_d = np.zeros(n)
    for i in range(n):
        if typ[i] == "PAYMENT":
            continue
        if f[i]:
            old_d[i] = 0 if rng.random() < 0.55 else rng.lognormal(8, 1.5)
            new_d[i] = old_d[i] if rng.random() < 0.5 else old_d[i] + amount[i]
        else:
            old_d[i] = rng.lognormal(11, 2) if rng.random() < 0.6 else 0
            new_d[i] = old_d[i] if rng.random() < 0.15 else old_d[i] + amount[i]

    name_o = ["C" + str(x) for x in rng.integers(10**8, 10**9, n)]
    name_d = [("M" if t == "PAYMENT" else "C") + str(x) for t, x in zip(typ, rng.integers(10**8, 10**9, n))]
    df = pd.DataFrame({
        "step": step, "type": typ, "amount": amount, "nameOrig": name_o,
        "oldbalanceOrg": old_o.round(2), "newbalanceOrg": new_o.round(2), "nameDest": name_d,
        "oldbalanceDest": old_d.round(2), "newbalanceDest": new_d.round(2), "isFraud": fraud})
    df["isFlaggedFraud"] = ((df["type"] == "TRANSFER") & (df["amount"] > 200000)).astype(int)
    df = df.sort_values("step").reset_index(drop=True)
    df.to_csv(os.path.join(DATA, "paysim_transactions.csv"), index=False)
    print("fraud:", df.shape, "fraud rate = %.4f" % df.isFraud.mean())


if __name__ == "__main__":
    make_churn()
    make_sales()
    make_fraud()

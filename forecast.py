"""
Predict Future Sales (Kaggle, 1C): прогноз месячных продаж по магазину и товару.

Шаги:
  1. Месячная агрегация продаж на сетке (месяц x магазин x товар).
  2. Baseline: скользящее среднее за последние N месяцев.
  3. LightGBM с лагами и скользящими признаками.
  4. Сравнение по RMSE на валидационном месяце 33 (октябрь 2015).
  5. Переобучение на месяцах до 33 включительно и submission.csv для месяца 34.

Запуск:
  python forecast.py --data ./data --out ./output

В папке data должны лежать файлы соревнования:
  sales_train.csv, test.csv, items.csv, item_categories.csv, shops.csv
"""
import argparse
import json
import os
from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd

VALID_MONTH = 33
TEST_MONTH = 34
FIRST_TRAIN_MONTH = 12  # первые 12 месяцев уходят на лаг 12
LAGS = [1, 2, 3, 6, 12]
CLIP = (0, 20)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def load(data_dir):
    sales = pd.read_csv(os.path.join(data_dir, "sales_train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    items = pd.read_csv(os.path.join(data_dir, "items.csv"))
    cats = pd.read_csv(os.path.join(data_dir, "item_categories.csv"))
    return sales, test, items, cats


def clean(sales, test):
    sales = sales[(sales.item_price > 0) & (sales.item_price < 100000)]
    sales = sales[(sales.item_cnt_day > -1000) & (sales.item_cnt_day < 1000)]
    # Дубликаты магазинов с разными id (одинаковые названия в shops.csv);
    # приводим к id, которые есть в test.csv
    shop_map = {0: 57, 1: 58, 11: 10}
    sales = sales.assign(shop_id=sales.shop_id.replace(shop_map))
    test = test.assign(shop_id=test.shop_id.replace(shop_map))
    return sales, test


def build_matrix(sales, test, items, cats):
    # Сетка: все пары магазин x товар, активные в каждом месяце
    grid = []
    for m in sorted(sales.date_block_num.unique()):
        part = sales[sales.date_block_num == m]
        grid.append(np.array(list(product([m], part.shop_id.unique(), part.item_id.unique())), dtype=np.int32))
    grid = pd.DataFrame(np.vstack(grid), columns=["date_block_num", "shop_id", "item_id"])

    sales = sales.assign(revenue=sales.item_price * sales.item_cnt_day)
    monthly = (
        sales.groupby(["date_block_num", "shop_id", "item_id"])
        .agg(item_cnt_month=("item_cnt_day", "sum"))
        .reset_index()
    )
    df = grid.merge(monthly, on=["date_block_num", "shop_id", "item_id"], how="left")
    df["item_cnt_month"] = df["item_cnt_month"].fillna(0).clip(*CLIP).astype(np.float32)

    t = test[["shop_id", "item_id"]].drop_duplicates().copy()
    t["date_block_num"] = TEST_MONTH
    t["item_cnt_month"] = np.nan
    df = pd.concat([df, t], ignore_index=True)

    # Средняя цена товара по всей истории (известна заранее)
    price = sales.groupby("item_id").item_price.mean().rename("item_price_mean").astype(np.float32)
    df = df.merge(price, on="item_id", how="left")

    cats = cats.assign(cat_type=cats.item_category_name.str.split("-").str[0].str.strip())
    cats["cat_type"] = cats.cat_type.astype("category").cat.codes.astype(np.int16)
    df = df.merge(items[["item_id", "item_category_id"]], on="item_id", how="left")
    df = df.merge(cats[["item_category_id", "cat_type"]], on="item_category_id", how="left")

    for c in ["date_block_num", "shop_id", "item_category_id", "cat_type"]:
        df[c] = df[c].astype(np.int16)
    df["item_id"] = df["item_id"].astype(np.int32)
    return df


def add_lag(df, src, keys, lags, prefix):
    tmp = df[["date_block_num"] + keys + [src]].drop_duplicates(["date_block_num"] + keys)
    for lag in lags:
        shifted = tmp.copy()
        shifted["date_block_num"] = shifted["date_block_num"] + lag
        shifted = shifted.rename(columns={src: f"{prefix}_lag{lag}"})
        df = df.merge(shifted, on=["date_block_num"] + keys, how="left")
    return df


def add_features(df):
    keys = ["shop_id", "item_id"]

    # Лаги целевой переменной для пары магазин-товар
    df = add_lag(df, "item_cnt_month", keys, LAGS, "cnt")

    # Средние по группам за месяц, сдвинутые на 1 месяц
    groups = {
        "item": ["item_id"],
        "shop": ["shop_id"],
        "cat": ["item_category_id"],
        "shop_cat": ["shop_id", "item_category_id"],
    }
    for name, g in groups.items():
        col = f"{name}_mean"
        means = (
            df[df.date_block_num < TEST_MONTH]
            .groupby(["date_block_num"] + g).item_cnt_month.mean().astype(np.float32)
            .rename(col).reset_index()
        )
        df = df.merge(means, on=["date_block_num"] + g, how="left")
        lags = [1, 2, 3] if name == "item" else [1]
        df = add_lag(df, col, g, lags, col)
        df = df.drop(columns=col)

    lag_cols = [f"cnt_lag{l}" for l in LAGS]
    df[lag_cols] = df[lag_cols].fillna(0)

    # Скользящие признаки по лагам (только прошлое, без утечки)
    df["cnt_roll_mean3"] = df[["cnt_lag1", "cnt_lag2", "cnt_lag3"]].mean(axis=1)
    df["cnt_roll_std3"] = df[["cnt_lag1", "cnt_lag2", "cnt_lag3"]].std(axis=1)
    df["cnt_trend"] = df["cnt_lag1"] - df["cnt_lag2"]

    # Время с первой продажи товара и пары магазин-товар
    hist = df[(df.date_block_num < TEST_MONTH) & (df.item_cnt_month > 0)]
    first_item = hist.groupby("item_id").date_block_num.min().rename("item_first")
    first_pair = hist.groupby(keys).date_block_num.min().rename("pair_first")
    df = df.merge(first_item, on="item_id", how="left").merge(first_pair, on=keys, how="left")
    df["item_age"] = (df.date_block_num - df.item_first).where(df.item_first < df.date_block_num, -1)
    df["pair_age"] = (df.date_block_num - df.pair_first).where(df.pair_first < df.date_block_num, -1)
    df = df.drop(columns=["item_first", "pair_first"])

    df["month"] = (df.date_block_num % 12).astype(np.int8)

    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def moving_average_baseline(df, target_month, window):
    """Среднее продаж пары магазин-товар за window месяцев до target_month (нули, где продаж не было)."""
    hist = df[(df.date_block_num >= target_month - window) & (df.date_block_num < target_month)]
    ma = hist.groupby(["shop_id", "item_id"]).item_cnt_month.sum() / window
    target = df[df.date_block_num == target_month][["shop_id", "item_id"]]
    return target.merge(ma.rename("pred").reset_index(), on=["shop_id", "item_id"], how="left")["pred"].fillna(0).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./output")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sales, test, items, cats = load(args.data)
    sales, test = clean(sales, test)
    df = build_matrix(sales, test, items, cats)
    print(f"Матрица: {len(df):,} строк")

    valid = df[df.date_block_num == VALID_MONTH]
    y_valid = valid.item_cnt_month.values
    results = {}

    # 1. Baseline: скользящее среднее
    for w in [1, 3, 6]:
        p = moving_average_baseline(df, VALID_MONTH, w)
        results[f"moving_average_{w}m"] = rmse(y_valid, np.clip(p, *CLIP))
        print(f"Скользящее среднее {w} мес.: RMSE = {results[f'moving_average_{w}m']:.4f}")

    # 2. LightGBM
    df = add_features(df)
    df = df[df.date_block_num >= FIRST_TRAIN_MONTH]
    features = [c for c in df.columns if c not in ("item_cnt_month", "date_block_num", "ID")]
    cat_features = ["shop_id", "item_category_id", "cat_type", "month"]

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=255,
        min_data_in_leaf=100,
        feature_fraction=0.75,
        bagging_fraction=0.75,
        bagging_freq=1,
        lambda_l2=1.0,
        verbose=-1,
        seed=42,
    )

    tr = df[df.date_block_num < VALID_MONTH]
    va = df[df.date_block_num == VALID_MONTH]
    dtr = lgb.Dataset(tr[features], tr.item_cnt_month, categorical_feature=cat_features)
    dva = lgb.Dataset(va[features], va.item_cnt_month, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=1000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    p_valid = np.clip(model.predict(va[features], num_iteration=model.best_iteration), *CLIP)
    results["lightgbm"] = rmse(va.item_cnt_month, p_valid)
    results["lightgbm_best_iteration"] = int(model.best_iteration)
    print(f"LightGBM: RMSE = {results['lightgbm']:.4f} (итераций: {model.best_iteration})")

    imp = pd.Series(model.feature_importance("gain"), index=features).sort_values(ascending=False)
    imp.to_csv(os.path.join(args.out, "feature_importance.csv"), header=["gain"])

    # 3. Финальная модель на всех месяцах до 33 включительно
    full = df[df.date_block_num <= VALID_MONTH]
    dfull = lgb.Dataset(full[features], full.item_cnt_month, categorical_feature=cat_features)
    final = lgb.train(params, dfull, num_boost_round=max(int(model.best_iteration * 1.05), 1))

    te = df[df.date_block_num == TEST_MONTH].drop_duplicates(["shop_id", "item_id"])
    pred = np.clip(final.predict(te[features]), *CLIP)
    sub = test[["ID", "shop_id", "item_id"]].merge(
        te[["shop_id", "item_id"]].assign(item_cnt_month=pred), on=["shop_id", "item_id"], how="left"
    )
    sub[["ID", "item_cnt_month"]].sort_values("ID").to_csv(os.path.join(args.out, "submission.csv"), index=False)

    # Baseline-сабмит для сравнения на лидерборде
    base = moving_average_baseline(df, TEST_MONTH, 3)
    base_sub = test[["ID", "shop_id", "item_id"]].merge(
        te[["shop_id", "item_id"]].assign(item_cnt_month=np.clip(base, *CLIP)), on=["shop_id", "item_id"], how="left"
    )
    base_sub[["ID", "item_cnt_month"]].sort_values("ID").to_csv(
        os.path.join(args.out, "submission_baseline.csv"), index=False
    )

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Готово: {os.path.join(args.out, 'submission.csv')}")


if __name__ == "__main__":
    main()

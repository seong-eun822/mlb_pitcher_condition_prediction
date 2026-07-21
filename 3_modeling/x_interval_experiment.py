"""X구간 실험 — 경기 초반을 어디까지 봐야 이후 whiff%를 가장 잘 맞추나.

pitch(n구) / inning(n이닝) / batter(n타자) × n 조합 8가지를 학습·비교한다.
feature_aggregator.build_and_save로 구간별 parquet를 만들고, 각각
XGBoost / CatBoost / LightGBM으로 학습해 val 성능을 비교한다.
"""

import os
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

from feature_aggregator import build_and_save


# ── 실험 조합 ───────────────────────────────────────────────

EXPERIMENTS = [
    # (mode,    n,  label)
    ('pitch',  10, 'pitch10'),
    ('pitch',  15, 'pitch15'),
    ('pitch',  20, 'pitch20'),
    ('inning',  1, 'inning1'),
    ('inning',  2, 'inning2'),
    ('batter',  3, 'batter3'),
    ('batter',  6, 'batter6'),
    ('batter',  9, 'batter9'),
]


# ── 단일 데이터셋 평가 ──────────────────────────────────────

def evaluate_dataset(feat_path: str, random_state: int = 42) -> dict:
    """주어진 parquet으로 XGBoost + CatBoost 학습 후 성능 반환."""
    df = pd.read_parquet(feat_path)

    meta_cols    = ['game_pk', 'pitcher', 'season', 'y_whiff']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    train = df[df['season'].isin([2021, 2022, 2023])]
    val   = df[df['season'] == 2024]
    test  = df[df['season'] == 2025]

    X_train, y_train = train[feature_cols], train['y_whiff']
    X_val,   y_val   = val[feature_cols],   val['y_whiff']
    X_test,  y_test  = test[feature_cols],  test['y_whiff']

    out = {
        'n_train':    len(train),
        'n_val':      len(val),
        'n_test':     len(test),
        'n_features': len(feature_cols),
    }

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, n_jobs=-1, verbosity=0,
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    for split, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        pred = xgb_model.predict(X)
        out[f'xgb_{split}_rmse'] = mean_squared_error(y, pred) ** 0.5
        out[f'xgb_{split}_r2']   = r2_score(y, pred)

    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        random_seed=random_state, verbose=False,
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    for split, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        pred = cb_model.predict(X)
        out[f'cb_{split}_rmse'] = mean_squared_error(y, pred) ** 0.5
        out[f'cb_{split}_r2']   = r2_score(y, pred)

    # LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)
    params = dict(
        objective='regression', metric='rmse',
        learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, n_jobs=-1, verbose=-1,
    )
    cb = lgb.early_stopping(50, verbose=False)
    lgb_model = lgb.train(
        params, lgb_train, num_boost_round=500,
        valid_sets=[lgb_val], callbacks=[cb],
    )
    for split, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        pred = lgb_model.predict(X)
        out[f'lgb_{split}_rmse'] = mean_squared_error(y, pred) ** 0.5
        out[f'lgb_{split}_r2']   = r2_score(y, pred)

    return out


# ── 전체 실험 ───────────────────────────────────────────────

def run_all(
    starters_path: str,
    lookup_path: str,
    feature_dir: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """모든 X구간 조합에 대해 feature 생성 + 학습 + 결과 반환."""
    records = []
    for mode, n, label in EXPERIMENTS:
        print(f'[{label}] feature 생성...')
        feat_path = build_and_save(
            starters_path=starters_path,
            lookup_path=lookup_path,
            out_dir=feature_dir,
            mode=mode, n=n,
            overwrite=overwrite,
        )

        print(f'[{label}] 모델 학습...')
        r = evaluate_dataset(feat_path)
        r['name'] = label
        r['mode'] = mode
        r['n']    = n
        records.append(r)
        print(f'  XGB Val RMSE={r["xgb_val_rmse"]:.4f}  R²={r["xgb_val_r2"]:.4f}')
        print(f'  CB  Val RMSE={r["cb_val_rmse"]:.4f}  R²={r["cb_val_r2"]:.4f}')
        print(f'  LGB Val RMSE={r["lgb_val_rmse"]:.4f}  R²={r["lgb_val_r2"]:.4f}')
        print()
    return pd.DataFrame(records)

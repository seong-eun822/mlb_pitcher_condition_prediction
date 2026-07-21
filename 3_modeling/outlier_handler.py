"""이상치 처리 전략 비교 실험.

구속 X feature의 극단값, whiff% 타겟의 비현실적 값(0.05 미만 / 0.60 초과)을
잘라내면(clip / 행 제거) 예측이 나아지는지 4가지로 비교한다.

  baseline     처리 없음
  clip_speed   X 구속 feature 상하 1% clip
  remove_y     Y 극단값 경기 제거 (whiff% < 0.05 or > 0.60)
  clip+remove  둘 조합
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score


META_COLS = ['game_pk', 'pitcher', 'season', 'y_whiff']

SPEED_COLS = [
    'avg_speed_all', 'avg_speed_Fastball', 'avg_speed_Breaking', 'avg_speed_Offspeed',
    'std_speed_all', 'std_speed_Fastball', 'std_speed_Breaking', 'std_speed_Offspeed',
]


# ── 이상치 처리 함수들 ───────────────────────────────────────

def clip_speed(df: pd.DataFrame, q_lo: float = 0.01, q_hi: float = 0.99) -> pd.DataFrame:
    """구속 관련 X feature 상하 q% clip."""
    df = df.copy()
    for col in SPEED_COLS:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        lo, hi = valid.quantile(q_lo), valid.quantile(q_hi)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def remove_y_outlier(df: pd.DataFrame, lo: float = 0.05, hi: float = 0.60) -> pd.DataFrame:
    """whiff% 극단값 경기 제거 (EDA 기반 기준)."""
    mask = (df['y_whiff'] >= lo) & (df['y_whiff'] <= hi)
    return df[mask].copy()


# ── 단일 실험 실행 ──────────────────────────────────────────

def run_experiment(
    feature_path: str,
    do_clip_speed: bool = False,
    do_remove_y: bool = False,
    y_lo: float = 0.05,
    y_hi: float = 0.60,
    random_state: int = 42,
) -> dict:
    df = pd.read_parquet(feature_path)

    # ── 이상치 처리 적용 ────────────────────────────────────
    n_before = len(df)
    if do_clip_speed:
        df = clip_speed(df)
    if do_remove_y:
        df = remove_y_outlier(df, lo=y_lo, hi=y_hi)
    n_after = len(df)

    # ── Train / Val / Test 분리 ──────────────────────────────
    feature_cols = [c for c in df.columns if c not in META_COLS]

    train = df[df['season'].isin([2021, 2022, 2023])]
    val   = df[df['season'] == 2024]
    test  = df[df['season'] == 2025]

    X_train, y_train = train[feature_cols], train['y_whiff']
    X_val,   y_val   = val[feature_cols],   val['y_whiff']
    X_test,  y_test  = test[feature_cols],  test['y_whiff']

    out = {
        'do_clip_speed': do_clip_speed,
        'do_remove_y':   do_remove_y,
        'n_removed':     n_before - n_after,
        'n_train':       len(train),
        'n_val':         len(val),
        'n_test':        len(test),
        'n_features':    len(feature_cols),
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
    lgb_model = lgb.train(
        params, lgb_train, num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    for split, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        pred = lgb_model.predict(X)
        out[f'lgb_{split}_rmse'] = mean_squared_error(y, pred) ** 0.5
        out[f'lgb_{split}_r2']   = r2_score(y, pred)

    return out


# ── 전체 실험 정의 ───────────────────────────────────────────

EXPERIMENTS = [
    # (name,          clip_speed, remove_y)
    ('baseline',     False,      False),
    ('clip_speed',   True,       False),
    ('remove_y',     False,      True),
    ('clip+remove',  True,       True),
]


def run_all(feature_path: str) -> pd.DataFrame:
    records = []
    for name, clip_speed, remove_y in EXPERIMENTS:
        print(f'[{name}] 실행 중...')
        r = run_experiment(feature_path, do_clip_speed=clip_speed, do_remove_y=remove_y)
        r['name'] = name
        records.append(r)
        print(f'  제거 샘플: {r["n_removed"]}개')
        print(f'  XGB Val RMSE={r["xgb_val_rmse"]:.4f}  R²={r["xgb_val_r2"]:.4f}')
        print(f'  CB  Val RMSE={r["cb_val_rmse"]:.4f}  R²={r["cb_val_r2"]:.4f}')
        print(f'  LGB Val RMSE={r["lgb_val_rmse"]:.4f}  R²={r["lgb_val_r2"]:.4f}')
        print()
    return pd.DataFrame(records)

"""결측(NaN) 처리 전략 비교 실험.

delta feature는 해당 구종을 안 던진 경기에 NaN이 생긴다. 이 NaN을
어떻게 다룰지(그대로 두기 / 0 대체 / 컬럼 삭제)에 따른 성능을 비교한다.
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ── 결측 처리 함수들 ────────────────────────────────────────

def impute_offspeed_zero(df: pd.DataFrame) -> pd.DataFrame:
    """미등판 구종(Offspeed/Breaking) delta → 0 impute.
    안 던졌으면 직전 시즌 대비 변화 없음(0)으로 간주.
    기준값 자체가 없는 경우(prev_* NaN)는 그대로 NaN 유지.
    """
    df = df.copy()
    delta_cols = [c for c in df.columns if c.startswith('delta_')]
    for col in delta_cols:
        # 구종 그룹 추출
        group = col.split('_')[-1]  # Fastball / Breaking / Offspeed
        avg_col = col.replace('delta_', 'avg_').replace(f'_{group}', f'_{group}')
        # avg_* 가 NaN → 해당 경기 미등판 → delta 0
        # avg_* 가 존재하는데 delta NaN → 기준값(prev_*) 없음 → 유지
        if avg_col in df.columns:
            mask_no_pitch = df[avg_col].isna() & df[col].isna()
            df.loc[mask_no_pitch, col] = 0.0
    return df


def drop_cols_by_nan(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """NaN 비율이 threshold 이상인 컬럼 제거."""
    df = df.copy()
    meta = ['game_pk', 'pitcher', 'season', 'y_woba']
    drop_cols = [
        c for c in df.columns
        if c not in meta and df[c].isna().mean() >= threshold
    ]
    df = df.drop(columns=drop_cols)
    return df, drop_cols


# ── 실험 실행 함수 ──────────────────────────────────────────

def run_experiment(
    feature_path: str,
    offspeed_zero: bool = False,
    drop_threshold: float = None,
    random_state: int = 42,
) -> dict:
    """
    Parameters
    ----------
    feature_path   : features_batter9.parquet 경로
    offspeed_zero  : True → 미등판 구종 delta NaN을 0으로 impute
    drop_threshold : None → 컬럼 제거 없음
                     0.3  → NaN 30% 이상 컬럼 제거
                     0.5  → NaN 50% 이상 컬럼 제거
    """
    df = pd.read_parquet(feature_path)

    # ── 결측 처리 적용 ──────────────────────────────────────
    dropped_cols = []
    if offspeed_zero:
        df = impute_offspeed_zero(df)
    if drop_threshold is not None:
        df, dropped_cols = drop_cols_by_nan(df, threshold=drop_threshold)

    # ── Train / Val / Test 분리 ─────────────────────────────
    meta_cols    = ['game_pk', 'pitcher', 'season', 'y_woba']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    train = df[df['season'].isin([2021, 2022, 2023])]
    val   = df[df['season'] == 2024]
    test  = df[df['season'] == 2025]

    X_train, y_train = train[feature_cols], train['y_woba']
    X_val,   y_val   = val[feature_cols],   val['y_woba']
    X_test,  y_test  = test[feature_cols],  test['y_woba']

    results = {
        'offspeed_zero':  offspeed_zero,
        'drop_threshold': drop_threshold,
        'n_features':     len(feature_cols),
        'dropped_cols':   dropped_cols,
    }

    # ── XGBoost ─────────────────────────────────────────────
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, n_jobs=-1, verbosity=0,
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    for split, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        pred = xgb_model.predict(X)
        results[f'xgb_{split}_rmse'] = mean_squared_error(y, pred) ** 0.5
        results[f'xgb_{split}_r2']   = r2_score(y, pred)

    # ── CatBoost ────────────────────────────────────────────
    cb_model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        random_seed=random_state, verbose=False,
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    for split, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        pred = cb_model.predict(X)
        results[f'cb_{split}_rmse'] = mean_squared_error(y, pred) ** 0.5
        results[f'cb_{split}_r2']   = r2_score(y, pred)

    return results


# ── 전체 실험 실행 ──────────────────────────────────────────

EXPERIMENTS = [
    # (name,          offspeed_zero, drop_threshold)
    ('baseline',      False,         None),
    ('zero',          True,          None),
    ('drop50',        False,         0.5),
    ('drop30',        False,         0.3),
    ('zero+drop50',   True,          0.5),
]


def run_all(feature_path: str) -> pd.DataFrame:
    """EXPERIMENTS의 5가지 결측 전략을 모두 실행해 성능 비교표를 반환."""
    records = []
    for name, offspeed_zero, drop_threshold in EXPERIMENTS:
        print(f'[{name}] 실행 중...')
        r = run_experiment(feature_path, offspeed_zero=offspeed_zero, drop_threshold=drop_threshold)
        r['name'] = name
        records.append(r)
        print(f'  XGB Val RMSE={r["xgb_val_rmse"]:.4f}  R²={r["xgb_val_r2"]:.4f}')
        print(f'  CB  Val RMSE={r["cb_val_rmse"]:.4f}  R²={r["cb_val_r2"]:.4f}')
    return pd.DataFrame(records)


if __name__ == '__main__':
    DRIVE       = '/content/drive/MyDrive/MLB_pitcher'
    FEAT_PATH   = os.path.join(DRIVE, 'data', '4_features', 'features_batter9.parquet')
    summary     = run_all(FEAT_PATH)
    print('\n=== 결과 요약 ===')
    cols = ['name', 'n_features',
            'xgb_val_rmse', 'xgb_val_r2',
            'cb_val_rmse',  'cb_val_r2']
    print(summary[cols].to_string(index=False))

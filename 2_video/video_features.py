# -*- coding: utf-8 -*-
"""
영상 생체역학 feature 파이프라인 모듈
====================================
좌표 합치기 / 각도 계산 / 경기 집계 로직을 함수화 (04_video_pipeline.ipynb가 import).
정형 쪽 feature_aggregator.py 와 동일한 패턴.

흐름:
    batch_slot*_coords.csv  --merge_coords-->  좌표+경기정보
                            --compute_angles--> 공별 생체역학 각도(좌투 미러링·정규화·이상치)
                            --aggregate-->      경기 단위 집계 feature (n_pitches × agg)

슬롯 = 시즌 (0~4 = 2021~2025). 전체 데이터는 slots=[0,1,2,3,4].

사용 예 (노트북에서):
    from video_features import merge_coords, compute_angles, aggregate
    coords = merge_coords(output_dir, play_ids_csv, slots=[0, 1, 2, 3, 4])
    angles = compute_angles(coords)
    feat   = aggregate(angles, n_pitches=15, agg='full9')
"""

import os
import glob
import numpy as np
import pandas as pd


# ── 관절/각도 상수 ──────────────────────────────────────────────
JOINTS = ['ear', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']

ANGLE_COLS = [
    'stride_norm', 'arm_slot', 'shoulder_tilt', 'hip_tilt',
    'trunk_dist_norm', 'trunk_angle', 'separation',
    'release_height_norm', 'arm_extension_norm',
]

# 집계 통계 9종 (순서 무관 분포 통계 — "초반 N구" 정체성에 맞음)
# ※ 추세(전반/후반)는 제외: 초반 예측이라는 문제 정의와 충돌 + 투구순서 정보 부재
AGG9 = ['mean', 'std', 'min', 'max', 'q25', 'q50', 'q75', 'range', 'skew']


# ── 1. 좌표 합치기 + 경기정보 조인 (구 09) ──────────────────────
def merge_coords(output_dir, play_ids_csv, slots=(0, 1, 2, 3, 4)):
    """batch_slot*_coords.csv 들을 합치고 play_id->game_pk/pitcher/season 조인.

    Returns: DataFrame (좌표 + pitcher/game_pk/season, pitcher 컬럼명 통일)
    """
    frames = []
    for slot in slots:
        files = sorted(glob.glob(os.path.join(output_dir, f'batch_slot{slot}_*_coords.csv')))
        for f in files:
            d = pd.read_csv(f)
            if len(d):
                d['src_slot'] = slot
                frames.append(d)
    if not frames:
        raise FileNotFoundError(f'좌표 파일 없음: {output_dir} slots={slots}')

    coords = pd.concat(frames, ignore_index=True)
    coords = coords.drop_duplicates(subset='video_name', keep='first').reset_index(drop=True)

    play = pd.read_csv(play_ids_csv, dtype={'play_id': str})
    play['game_pk'] = play['game_pk'].astype('int64')
    play['season'] = play['season'].astype('int64')

    merged = coords.merge(
        play[['play_id', 'pitcher_id', 'game_pk', 'season']],
        left_on='video_name', right_on='play_id', how='left',
    )
    merged = merged.dropna(subset=['game_pk', 'pitcher_id']).reset_index(drop=True)
    merged['game_pk'] = merged['game_pk'].astype('int64')
    merged['pitcher_id'] = merged['pitcher_id'].astype('int64')
    # 정형 데이터와 키 이름 통일
    merged = merged.rename(columns={'pitcher_id': 'pitcher'}).drop(columns=['play_id'])
    return merged


# ── 2. 좌투 미러링 ──────────────────────────────────────────────
def mirror_lefties(df):
    """좌투(L)를 우투 기준으로 통일: x좌표 미러 + left/right 관절 swap.
    안 하면 좌투 각도가 좌우 반전돼 우투와 비교 불가."""
    df = df.copy()
    x_cols = [c for c in df.columns if c.endswith('_x')]
    is_left = (df['hand'] == 'L')
    if is_left.any():
        row_xmax = df.loc[is_left, x_cols].max(axis=1)
        for c in x_cols:
            df.loc[is_left, c] = row_xmax - df.loc[is_left, c]
        for j in JOINTS:
            for ax in ['x', 'y']:
                lc, rc = f'left_{j}_{ax}', f'right_{j}_{ax}'
                if lc in df.columns and rc in df.columns:
                    tmp = df.loc[is_left, lc].copy()
                    df.loc[is_left, lc] = df.loc[is_left, rc].values
                    df.loc[is_left, rc] = tmp.values
    return df


# ── 3. 공별 각도 계산 (구 10) ──────────────────────────────────
def _P(df, joint, ax):
    return df[f'{joint}_{ax}'].to_numpy(dtype='float64')


def _joint_angle(df, a, b, c):
    """b를 꼭짓점으로 한 a-b-c 각도(도)."""
    v1x, v1y = _P(df, a, 'x') - _P(df, b, 'x'), _P(df, a, 'y') - _P(df, b, 'y')
    v2x, v2y = _P(df, c, 'x') - _P(df, b, 'x'), _P(df, c, 'y') - _P(df, b, 'y')
    n1 = np.sqrt(v1x**2 + v1y**2)
    n2 = np.sqrt(v2x**2 + v2y**2)
    cos = (v1x*v2x + v1y*v2y) / np.where((n1*n2) < 1e-6, np.nan, n1*n2)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def compute_angles(coords, mirror=True):
    """공 1개당 생체역학 각도/거리 9종 계산. 어깨너비로 정규화.

    Returns: DataFrame (video_name/pitcher/game_pk/season/hand + ANGLE_COLS)
    """
    dfm = mirror_lefties(coords) if mirror else coords.copy()
    ang = pd.DataFrame(index=dfm.index)

    sw = np.sqrt((_P(dfm, 'right_shoulder', 'x') - _P(dfm, 'left_shoulder', 'x'))**2 +
                 (_P(dfm, 'right_shoulder', 'y') - _P(dfm, 'left_shoulder', 'y'))**2)
    sw = np.where(sw < 1e-6, np.nan, sw)

    stride = np.sqrt((_P(dfm, 'left_ankle', 'x') - _P(dfm, 'right_ankle', 'x'))**2 +
                     (_P(dfm, 'left_ankle', 'y') - _P(dfm, 'right_ankle', 'y'))**2)
    ang['stride_norm'] = stride / sw
    ang['arm_slot'] = _joint_angle(dfm, 'right_wrist', 'right_elbow', 'right_shoulder')
    ang['shoulder_tilt'] = np.degrees(np.arctan2(
        _P(dfm, 'right_shoulder', 'y') - _P(dfm, 'left_shoulder', 'y'),
        _P(dfm, 'right_shoulder', 'x') - _P(dfm, 'left_shoulder', 'x')))
    ang['hip_tilt'] = np.degrees(np.arctan2(
        _P(dfm, 'right_hip', 'y') - _P(dfm, 'left_hip', 'y'),
        _P(dfm, 'right_hip', 'x') - _P(dfm, 'left_hip', 'x')))

    hipc_x = (_P(dfm, 'left_hip', 'x') + _P(dfm, 'right_hip', 'x')) / 2
    hipc_y = (_P(dfm, 'left_hip', 'y') + _P(dfm, 'right_hip', 'y')) / 2
    shlc_x = (_P(dfm, 'left_shoulder', 'x') + _P(dfm, 'right_shoulder', 'x')) / 2
    shlc_y = (_P(dfm, 'left_shoulder', 'y') + _P(dfm, 'right_shoulder', 'y')) / 2
    ang['trunk_dist_norm'] = np.sqrt((shlc_x - hipc_x)**2 + (shlc_y - hipc_y)**2) / sw
    ang['trunk_angle'] = np.degrees(np.arctan2(shlc_y - hipc_y, shlc_x - hipc_x))

    # separation: 어깨-골반 각도차를 0~180으로 wrap
    sep = np.abs(ang['shoulder_tilt'] - ang['hip_tilt']) % 360
    ang['separation'] = np.where(sep > 180, 360 - sep, sep)

    ang['release_height_norm'] = (_P(dfm, 'right_shoulder', 'y') - _P(dfm, 'right_wrist', 'y')) / sw
    ang['arm_extension_norm'] = (_P(dfm, 'right_wrist', 'x') - _P(dfm, 'right_shoulder', 'x')) / sw

    keep = ['video_name', 'pitcher', 'game_pk', 'season', 'hand']
    keep = [c for c in keep if c in dfm.columns]
    return pd.concat([dfm[keep].reset_index(drop=True),
                      ang.reset_index(drop=True)], axis=1)


# ── 4. 이상치 + 경기 집계 (구 11) ──────────────────────────────
def winsorize(angles, low=0.01, high=0.99):
    """정규화값 폭발(어깨너비≈0)을 1~99% 클리핑. 결측 행 제거."""
    df = angles.dropna(subset=ANGLE_COLS).reset_index(drop=True)
    for c in ANGLE_COLS:
        lo, hi = df[c].quantile(low), df[c].quantile(high)
        df[c] = df[c].clip(lo, hi)
    return df


KEYS = ['game_pk', 'pitcher', 'season']


def _agg9_series(g):
    out = {}
    for c in ANGLE_COLS:
        x = g[c]
        out[f'{c}_mean'] = x.mean()
        out[f'{c}_std'] = x.std()
        out[f'{c}_min'] = x.min()
        out[f'{c}_max'] = x.max()
        out[f'{c}_q25'] = x.quantile(0.25)
        out[f'{c}_q50'] = x.quantile(0.50)
        out[f'{c}_q75'] = x.quantile(0.75)
        out[f'{c}_range'] = x.max() - x.min()
        out[f'{c}_skew'] = x.skew()
    return pd.Series(out)


def aggregate(angles, n_pitches=15, agg='full9', min_pitches=3, do_winsorize=True):
    """공 N개 → 경기 1개 집계.

    n_pitches: 경기 앞 N구 (None이면 전체). ※ N=15는 사실상 전체라 정확, N<15는 근사.
    agg: 'mean' | 'mean_std' | 'full9'
    """
    df = winsorize(angles) if do_winsorize else angles.dropna(subset=ANGLE_COLS)
    df = df.sort_values(['game_pk', 'pitcher', 'video_name']).reset_index(drop=True)

    g0 = df.groupby(KEYS, sort=False)
    head = g0.head(n_pitches) if n_pitches else df
    grp = head.groupby(KEYS, sort=False)

    cnt = grp.size()
    valid = cnt[cnt >= min_pitches].index

    if agg == 'mean':
        feat = grp[ANGLE_COLS].mean()
        feat.columns = [f'{c}_mean' for c in feat.columns]
    elif agg == 'mean_std':
        m = grp[ANGLE_COLS].mean(); m.columns = [f'{c}_mean' for c in m.columns]
        s = grp[ANGLE_COLS].std();  s.columns = [f'{c}_std'  for c in s.columns]
        feat = pd.concat([m, s], axis=1)
    elif agg == 'full9':
        feat = grp.apply(_agg9_series)
    else:
        raise ValueError(f'agg는 mean/mean_std/full9 중 하나: {agg}')

    feat = feat.loc[feat.index.isin(valid)].reset_index()
    feat['n_pitches_used'] = grp.size().loc[feat.set_index(KEYS).index].values
    return feat


def build_all(output_dir, play_ids_csv, out_dir, slots=(0, 1, 2, 3, 4),
              n_list=(3, 5, 10, 15), agg_list=('mean', 'mean_std', 'full9')):
    """end-to-end: 좌표→각도→ (N×agg) parquet 일괄 생성. all_angles.csv도 저장."""
    os.makedirs(out_dir, exist_ok=True)
    coords = merge_coords(output_dir, play_ids_csv, slots=slots)
    angles = compute_angles(coords)
    angles.to_csv(os.path.join(out_dir, 'all_angles.csv'), index=False, encoding='utf-8')

    import itertools
    for n, agg in itertools.product(n_list, agg_list):
        feat = aggregate(angles, n_pitches=n, agg=agg)
        fname = f'features_bio_pitch{n}_{agg}.parquet'
        feat.to_parquet(os.path.join(out_dir, fname), index=False)
        n_feat = len([c for c in feat.columns if c not in KEYS + ['n_pitches_used']])
        print(f'  {fname:44s}  경기 {len(feat):>4} · 피처 {n_feat}')
    return angles

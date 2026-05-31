"""
X 구간별로 feature + Y 를 집계하는 모듈.

- 03_feature_engineering.ipynb의 로직을 함수화
- starters_all.parquet + prev_season_lookup.parquet 로부터
  (mode, n) 파라미터로 features_<mode><n>.parquet 생성
"""

import os
import pandas as pd
import numpy as np
import duckdb


FASTBALL = ['FF', 'SI', 'FC']
BREAKING = ['SL', 'CU', 'KC']
OFFSPEED = ['CH', 'FS']

PITCH_GROUP_SQL = """
    CASE pitch_type
        WHEN 'FF' THEN 'Fastball' WHEN 'SI' THEN 'Fastball' WHEN 'FC' THEN 'Fastball'
        WHEN 'SL' THEN 'Breaking' WHEN 'CU' THEN 'Breaking' WHEN 'KC' THEN 'Breaking'
        WHEN 'CH' THEN 'Offspeed' WHEN 'FS' THEN 'Offspeed'
        ELSE 'Other'
    END
"""

IS_STRIKE_SQL = """
    CASE WHEN description IN ('called_strike','swinging_strike',
                              'swinging_strike_blocked','foul','foul_tip') THEN 1
         ELSE 0 END
"""


# ── X 구간 SQL 필터 생성 ────────────────────────────────────

def _x_zone_sql(mode: str, n: int) -> str:
    """X구간(투구 데이터)을 정의하는 SQL.
    starters 테이블에 pitch_group, is_strike, season 컬럼이 있다고 가정.
    """
    if mode == 'pitch':
        # 경기별 처음 n구
        return f"""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY game_pk, pitcher
                        ORDER BY at_bat_number, pitch_number
                    ) AS pitch_rank
                FROM starters
            )
            SELECT * FROM ranked WHERE pitch_rank <= {n}
        """
    elif mode == 'inning':
        return f"""
            SELECT * FROM starters WHERE inning <= {n}
        """
    elif mode == 'batter':
        return f"""
            WITH game_min AS (
                SELECT game_pk, pitcher, MIN(at_bat_number) AS min_ab
                FROM starters GROUP BY game_pk, pitcher
            )
            SELECT s.* FROM starters s
            JOIN game_min g
              ON s.game_pk = g.game_pk AND s.pitcher = g.pitcher
            WHERE s.at_bat_number <= g.min_ab + {n} - 1
        """
    else:
        raise ValueError(f'mode는 pitch/inning/batter 중 하나여야 함: {mode}')


# ── X feature 집계 ──────────────────────────────────────────

def _aggregate_x_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """x_zone 뷰로부터 경기 단위 feature DataFrame을 생성."""

    # 구종 그룹별 집계
    pitch_group_features = con.execute("""
        SELECT
            game_pk, pitcher, season, pitch_group,
            COUNT(*) AS pitch_count,
            AVG(release_speed) AS avg_speed,
            STDDEV(release_speed) AS std_speed,
            AVG(release_spin_rate) AS avg_spin,
            AVG(release_extension) AS avg_ext,
            AVG(release_pos_x) AS avg_pos_x,
            AVG(release_pos_z) AS avg_pos_z
        FROM x_zone
        WHERE pitch_group != 'Other'
        GROUP BY game_pk, pitcher, season, pitch_group
    """).df()

    # 경기 단위 전체 집계
    game_features = con.execute("""
        SELECT
            game_pk, pitcher, season,
            COUNT(*) AS total_pitches,
            AVG(release_speed) AS avg_speed_all,
            STDDEV(release_speed) AS std_speed_all,
            AVG(is_strike) AS strike_ratio,
            AVG(CASE WHEN pitch_group='Fastball' THEN 1.0 ELSE 0.0 END) AS fastball_ratio,
            AVG(CASE WHEN pitch_group='Breaking' THEN 1.0 ELSE 0.0 END) AS breaking_ratio,
            AVG(CASE WHEN pitch_group='Offspeed' THEN 1.0 ELSE 0.0 END) AS offspeed_ratio,
            AVG(release_pos_x) AS avg_pos_x,
            AVG(release_pos_z) AS avg_pos_z,
            AVG(release_extension) AS avg_ext,
            AVG(arm_angle) AS avg_arm_angle
        FROM x_zone
        GROUP BY game_pk, pitcher, season
    """).df()

    # wide format pivot
    pivot = pitch_group_features.pivot_table(
        index=['game_pk', 'pitcher', 'season'],
        columns='pitch_group',
        values=['avg_speed', 'std_speed', 'avg_spin', 'avg_ext', 'avg_pos_x', 'avg_pos_z'],
        aggfunc='first'
    )
    pivot.columns = [f'{v}_{g}' for v, g in pivot.columns]
    pivot = pivot.reset_index()

    return game_features.merge(pivot, on=['game_pk', 'pitcher', 'season'], how='left')


# ── delta feature 병합 ──────────────────────────────────────

def _merge_delta(features: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """직전 시즌 lookup → 구종 그룹별 평균 → delta feature 계산."""
    lookup_pivot = (
        lookup[['pitcher', 'season']].drop_duplicates()
        .set_index(['pitcher', 'season'])
    )

    for metric, prefix in [
        ('prev_avg_speed', 'prev_speed'),
        ('prev_avg_spin',  'prev_spin'),
        ('prev_avg_ext',   'prev_ext'),
        ('prev_avg_pos_x', 'prev_pos_x'),
        ('prev_avg_pos_z', 'prev_pos_z'),
    ]:
        if metric not in lookup.columns:
            continue
        pivot = lookup.pivot_table(
            index=['pitcher', 'season'],
            columns='pitch_type',
            values=metric,
            aggfunc='mean',
        )
        existing = pivot.columns.tolist()
        for group, types in [('Fastball', FASTBALL), ('Breaking', BREAKING), ('Offspeed', OFFSPEED)]:
            cols = [c for c in types if c in existing]
            col = f'{prefix}_{group}'
            lookup_pivot[col] = pivot[cols].mean(axis=1) if cols else pd.NA

    lookup_pivot = lookup_pivot.reset_index()
    features = features.merge(lookup_pivot, on=['pitcher', 'season'], how='left')

    # delta 계산
    for metric, today_prefix, prev_prefix in [
        ('speed', 'avg_speed', 'prev_speed'),
        ('spin',  'avg_spin',  'prev_spin'),
        ('ext',   'avg_ext',   'prev_ext'),
        ('pos_x', 'avg_pos_x', 'prev_pos_x'),
        ('pos_z', 'avg_pos_z', 'prev_pos_z'),
    ]:
        for group in ['Fastball', 'Breaking', 'Offspeed']:
            today_col = f'{today_prefix}_{group}'
            prev_col  = f'{prev_prefix}_{group}'
            delta_col = f'delta_{metric}_{group}'
            if today_col in features.columns and prev_col in features.columns:
                features[delta_col] = features[today_col] - features[prev_col]

    return features


# ── Y 구간 wOBA 계산 ────────────────────────────────────────

def _calc_y(con: duckdb.DuckDBPyConnection, mode: str, n: int, min_swings: int = 20) -> pd.DataFrame:
    """Y구간(X구간 이후) whiff% (헛스윙률) 계산.

    whiff% = swinging_strike 수 / 전체 스윙 수
    전체 스윙 = swinging_strike + swinging_strike_blocked + foul + foul_tip + hit_into_play

    - X구간 정의는 _x_zone_sql과 동일 → 그 외 = Y구간
    - 스윙 수가 min_swings 미만이면 제외 (분모 안정성)
    """
    swing_agg = """
        SUM(CASE WHEN description IN ('swinging_strike','swinging_strike_blocked')
                 THEN 1 ELSE 0 END) AS whiffs,
        SUM(CASE WHEN description IN ('swinging_strike','swinging_strike_blocked',
                                       'foul','foul_tip','hit_into_play')
                 THEN 1 ELSE 0 END) AS swings
    """

    if mode == 'pitch':
        df = con.execute(f"""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY game_pk, pitcher
                                       ORDER BY at_bat_number, pitch_number) AS pitch_rank
                FROM starters
            )
            SELECT game_pk, pitcher, season, {swing_agg}
            FROM ranked WHERE pitch_rank > ?
            GROUP BY game_pk, pitcher, season
        """, [n]).df()

    elif mode == 'inning':
        df = con.execute(f"""
            SELECT game_pk, pitcher, season, {swing_agg}
            FROM starters WHERE inning > ?
            GROUP BY game_pk, pitcher, season
        """, [n]).df()

    elif mode == 'batter':
        df = con.execute(f"""
            WITH game_min AS (
                SELECT game_pk, pitcher, MIN(at_bat_number) AS min_ab
                FROM starters GROUP BY game_pk, pitcher
            )
            SELECT s.game_pk, s.pitcher, s.season, {swing_agg.replace('description', 's.description')}
            FROM starters s
            JOIN game_min g
              ON s.game_pk = g.game_pk AND s.pitcher = g.pitcher
            WHERE s.at_bat_number > g.min_ab + ? - 1
            GROUP BY s.game_pk, s.pitcher, s.season
        """, [n]).df()
    else:
        raise ValueError(f'mode: {mode}')

    df = df[df['swings'] >= min_swings].copy()
    df['y_whiff'] = (df['whiffs'] / df['swings']).round(4)
    return df[['game_pk', 'pitcher', 'season', 'swings', 'y_whiff']]


# ── 메인 함수 ───────────────────────────────────────────────

def build_features(
    starters_path: str,
    lookup_path: str,
    mode: str,
    n: int,
    db_path: str = ':memory:',
    min_y_ab: int = 5,
    include_delta: bool = True,
) -> pd.DataFrame:
    """주어진 (mode, n) X구간에 대해 features + Y 데이터셋 생성.

    Parameters
    ----------
    include_delta : delta feature(직전 시즌 대비 편차) 포함 여부.
                   False이면 절대값 feature만 사용 (paired t-test 비교용).

    Returns
    -------
    DataFrame : game_pk, pitcher, season, ...features..., y_whiff
    """
    con = duckdb.connect(db_path)

    # starters 로드 + 구종 그룹 / strike 컬럼 추가
    con.execute(f"""
        CREATE OR REPLACE TABLE starters AS
        SELECT *, {PITCH_GROUP_SQL} AS pitch_group, {IS_STRIKE_SQL} AS is_strike
        FROM read_parquet('{starters_path}')
    """)

    # x_zone 뷰 등록
    con.execute(f"CREATE OR REPLACE VIEW x_zone AS {_x_zone_sql(mode, n)}")

    # X feature 집계
    features = _aggregate_x_features(con)

    # delta feature 병합 (include_delta=False면 절대값만 유지)
    if include_delta:
        lookup = pd.read_parquet(lookup_path)
        features = _merge_delta(features, lookup)

    # Y 계산
    y_df = _calc_y(con, mode, n, min_swings=min_y_ab)

    # 병합
    final = features.merge(
        y_df[['game_pk', 'pitcher', 'season', 'y_whiff']],
        on=['game_pk', 'pitcher', 'season'], how='inner',
    )
    final = final.dropna(subset=['y_whiff'])

    con.close()
    return final


def build_and_save(
    starters_path: str,
    lookup_path: str,
    out_dir: str,
    mode: str,
    n: int,
    overwrite: bool = False,
    include_delta: bool = True,
) -> str:
    """build_features 결과를 features_<mode><n>.parquet 로 저장."""
    suffix = '' if include_delta else '_nodelta'
    out_path = os.path.join(out_dir, f'features_{mode}{n}{suffix}.parquet')
    if os.path.exists(out_path) and not overwrite:
        print(f'[skip] 이미 존재: {out_path}')
        return out_path

    df = build_features(starters_path, lookup_path, mode, n, include_delta=include_delta)
    df.to_parquet(out_path, index=False)
    print(f'[saved] {out_path}  ({len(df):,}행, {len(df.columns)}컬럼)')
    return out_path

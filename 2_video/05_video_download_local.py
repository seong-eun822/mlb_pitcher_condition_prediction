"""
MLB 투수 영상 다운로드 - 로컬 PC용
실행 방법:
    PC1: python 05_video_download_local.py --slot 2
    PC2: python 05_video_download_local.py --slot 3
    PC3: python 05_video_download_local.py --slot 4
    (코랩 A=0, 코랩 B=1)

사전 준비:
    1. play_ids_all.csv 를 이 스크립트와 같은 폴더에 복사
       (드라이브에서: data/play_ids_all.csv)
    2. pip install requests beautifulsoup4 pandas
"""

import os
import re
import time
import zipfile
import shutil
import argparse
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

# ══════════════════════════════════════════════════════
# 설정
PLAY_ID_CSV  = 'play_ids_all.csv'    # 같은 폴더에 복사해둘 것
VIDEO_DIR    = 'video_zips'          # zip 저장 폴더
TEMP_DIR     = 'tmp_videos'          # 임시 폴더
BATCH_SIZE   = 200
TOTAL        = 1_185_600
PC_COUNT     = 5                     # 코랩2 + 로컬3
# ══════════════════════════════════════════════════════

REQ_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36'
}

def get_cdn_url(play_id):
    url  = f'https://baseballsavant.mlb.com/sporty-videos?playId={play_id}'
    res  = requests.get(url, headers=REQ_HEADERS, timeout=15)
    soup = BeautifulSoup(res.text, 'html.parser')
    video = soup.find('video')
    if video and video.find('source'):
        return video.find('source').get('src')
    return None

def download_mp4(cdn_url, save_path):
    with requests.get(cdn_url, headers=REQ_HEADERS, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def main(slot: int):
    slot_size  = TOTAL // PC_COUNT             # 237,120
    idx_start  = slot * slot_size
    idx_end    = idx_start + slot_size if slot < PC_COUNT - 1 else TOTAL

    progress_csv = f'progress_slot_{slot}.csv'

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR,  exist_ok=True)

    all_play_df = pd.read_csv(PLAY_ID_CSV)
    slot_df     = all_play_df.iloc[idx_start:idx_end].reset_index(drop=True)

    if os.path.exists(progress_csv):
        progress = pd.read_csv(progress_csv)
        done_ids = set(progress[progress['status'] == 'done']['play_id'].tolist())
        print(f'체크포인트 로드: 완료 {len(done_ids):,}개')
    else:
        progress = pd.DataFrame(columns=['play_id', 'season', 'batch_id', 'status'])
        done_ids = set()
        print('체크포인트 없음 → 처음부터 시작')

    remaining = slot_df[~slot_df['play_id'].isin(done_ids)].reset_index(drop=True)

    print(f'슬롯         : {slot} (로컬)')
    print(f'담당 구간    : {idx_start:,} ~ {idx_end:,}')
    print(f'전체 담당    : {len(slot_df):,}개')
    print(f'남은 play_id : {len(remaining):,}개')
    print(f'예상 소요    : 약 {len(remaining) * 4 / 3600:.1f}시간')

    existing_zips = list(Path(VIDEO_DIR).glob(f'batch_slot{slot}_*.zip'))
    start_batch   = len(existing_zips)
    new_records   = []

    for batch_idx, batch_start in enumerate(range(0, len(remaining), BATCH_SIZE)):
        batch_num  = start_batch + batch_idx + 1
        batch_name = f'batch_slot{slot}_{batch_num:04d}'
        zip_path   = os.path.join(VIDEO_DIR, f'{batch_name}.zip')

        if os.path.exists(zip_path):
            print(f'[{batch_name}] 이미 존재 → 건너뜀')
            continue

        batch_df  = remaining.iloc[batch_start:batch_start + BATCH_SIZE]
        batch_dir = os.path.join(TEMP_DIR, batch_name)
        os.makedirs(batch_dir, exist_ok=True)

        print(f'\n[{batch_name}] {len(batch_df)}개 다운로드 시작...')
        batch_done = 0
        batch_fail = 0

        for i, (_, row) in enumerate(batch_df.iterrows()):
            play_id  = row['play_id']
            season   = row['season']
            mp4_path = os.path.join(batch_dir, f'{play_id}.mp4')

            try:
                cdn_url = get_cdn_url(play_id)
                if cdn_url:
                    download_mp4(cdn_url, mp4_path)
                    new_records.append({'play_id': play_id, 'season': season,
                                        'batch_id': batch_name, 'status': 'done'})
                    batch_done += 1
                else:
                    new_records.append({'play_id': play_id, 'season': season,
                                        'batch_id': batch_name, 'status': 'fail'})
                    batch_fail += 1
            except Exception:
                new_records.append({'play_id': play_id, 'season': season,
                                    'batch_id': batch_name, 'status': 'fail'})
                batch_fail += 1

            if (i + 1) % 10 == 0:
                print(f'  {i+1}/{len(batch_df)} | 성공 {batch_done} / 실패 {batch_fail}')

            time.sleep(1.0)

        print(f'[{batch_name}] zip 압축 중...')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for mp4 in Path(batch_dir).glob('*.mp4'):
                zf.write(mp4, mp4.name)

        zip_mb = os.path.getsize(zip_path) / 1024 / 1024
        print(f'[{batch_name}] 완료 ({zip_mb:.1f} MB) 성공 {batch_done} / 실패 {batch_fail}')

        shutil.rmtree(batch_dir)

        progress = pd.concat([progress, pd.DataFrame(new_records)], ignore_index=True)
        progress.to_csv(progress_csv, index=False)
        new_records = []

    print('\n전체 다운로드 완료!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slot', type=int, required=True, choices=[2, 3, 4],
                        help='로컬 PC 슬롯 (2, 3, 4) — 코랩이 0, 1 담당')
    args = parser.parse_args()
    main(args.slot)

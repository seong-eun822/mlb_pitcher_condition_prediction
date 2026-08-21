"""프로젝트 경로 설정 — Colab / 로컬 / 팀원 환경을 하나로 흡수한다.

사용법 (모든 노트북 첫 셀):

    import sys, os
    while not os.path.exists('config.py'):
        os.chdir('..')
    sys.path.insert(0, os.getcwd())
    from config import *
    check()

설계:
  1층 — 코드 루트(ROOT)는 자동 감지. 노트북이 몇 단계 깊이에 있든 무관.
  2층 — 데이터 위치(DATA_ROOT)는 사람마다 다르므로 `.env`로 분리한다.
        `.env`는 gitignore되므로 팀원끼리 값이 달라도 충돌하지 않는다.

데이터 폴더 구조가 환경마다 다른 문제도 여기서 흡수한다.
Colab 드라이브는 `0_data/4_features/`, 로컬은 루트에 parquet이 흩어져 있어
`find_data()`가 양쪽을 모두 탐색한다.
"""

import os
from pathlib import Path

# ── 1층: 실행 환경과 코드 루트 ────────────────────────────────

try:
    from google.colab import drive  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def _find_root() -> Path:
    """이 파일이 있는 폴더를 루트로 본다. 실패 시 cwd에서 위로 탐색."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        pass
    here = Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / "requirements.txt").exists():
            return cand
    raise RuntimeError(
        "프로젝트 루트를 찾을 수 없습니다. "
        "저장소 안에서 노트북을 실행하고 있는지 확인하세요."
    )


if IN_COLAB:
    # 노트북 첫 셀이 이미 드라이브를 마운트하고 이 파일이 있는 폴더로
    # 이동시켜 두었다. config.py의 실제 위치를 루트로 삼는다.
    ROOT = Path(__file__).resolve().parent
else:
    ROOT = _find_root()


# ── 2층: 데이터 위치 (사람마다 다름) ──────────────────────────


def _load_dotenv() -> None:
    """루트의 .env를 읽어 환경변수로 올린다. 이미 설정된 값은 덮지 않는다."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


_load_dotenv()

# 우선순위: 환경변수 > .env > 기본값(ROOT/0_data)
DATA_ROOT = Path(os.environ.get("PITCHER_DATA_ROOT", ROOT / "0_data"))


# ── 경로 정의 ────────────────────────────────────────────────

# 원본
STATCAST_DIR = DATA_ROOT / "statcast"          # statcast_2021~2025.parquet
RAW_DIR = DATA_ROOT / "data"                   # 크롤링 csv, 좌표/각도
VIDEO_OUT_DIR = DATA_ROOT / "output"           # 영상 배치 산출물

# 가공 — 환경마다 위치가 다르다.
# Colab 드라이브는 `0_data/4_features/`에 모아두었지만, 로컬은 프로젝트
# 루트에 parquet이 흩어져 있다. 실제로 존재하는 쪽을 자동으로 고른다.


def _pick(*candidates: Path) -> Path:
    """존재하는 첫 폴더를 반환. 없으면 첫 번째(표준 위치)를 반환한다."""
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


FEATURE_DIR = _pick(DATA_ROOT / "4_features", ROOT)
INTERIM_DIR = _pick(DATA_ROOT / "2_interim", ROOT)

# 산출물 — 4_output 하위를 성격별로 나눈다.
OUTPUT_DIR = ROOT / "4_output"
FIG_DIR = OUTPUT_DIR / "figures"        # 그림
EXP_DIR = OUTPUT_DIR / "experiments"    # 실험 결과 (채택·기각 근거)
FINAL_DIR = OUTPUT_DIR / "final"        # 최종 모델 관련 산출물
MODEL_DIR = OUTPUT_DIR / "models"       # 학습된 모델 (.pkl, gitignore)

TARGETS_PATH = DATA_ROOT / "game_targets.parquet"

# 기존 노트북 호환 별칭 — 새 코드에서는 위 이름을 쓸 것
STAT_DIR = FEATURE_DIR       # 정형 feature
BIO_DIR = FEATURE_DIR        # 영상 feature (같은 폴더)
FEAT_DIR = FEATURE_DIR
BASE_DIR = DATA_ROOT
IMAGE_DIR = OUTPUT_DIR / "final_images"
STATCAST_GLOB = str(STATCAST_DIR / "statcast_*.parquet")

for _d in (OUTPUT_DIR, FIG_DIR, EXP_DIR, FINAL_DIR, MODEL_DIR, IMAGE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── 데이터 파일 탐색 ──────────────────────────────────────────

# 파일이 있을 수 있는 위치들. 환경마다 구조가 달라 순서대로 확인한다.
_SEARCH_DIRS = [FEATURE_DIR, INTERIM_DIR, DATA_ROOT, ROOT, RAW_DIR]

# 산출물 탐색 경로 — 하위 폴더로 옮기기 전 코드와의 호환을 위해
# 옛 위치(4_output 직속)도 함께 확인한다.
_OUTPUT_SEARCH = [EXP_DIR, FINAL_DIR, MODEL_DIR, FIG_DIR, OUTPUT_DIR]


def find_output(filename: str) -> Path:
    """산출물 파일을 4_output 하위에서 찾는다. 없으면 EXP_DIR 경로를 반환."""
    for d in _OUTPUT_SEARCH:
        p = d / filename
        if p.exists():
            return p
    return EXP_DIR / filename


def find_data(filename: str, required: bool = True) -> Path:
    """데이터 파일을 여러 후보 폴더에서 찾는다.

    Colab(0_data/4_features/)과 로컬(루트에 흩어짐)의 구조 차이를 흡수한다.

        df = pd.read_parquet(find_data('features_pitch15.parquet'))
    """
    for d in _SEARCH_DIRS:
        p = d / filename
        if p.exists():
            return p
    if not required:
        return FEATURE_DIR / filename
    raise FileNotFoundError(
        f"\n데이터 파일을 찾을 수 없습니다: {filename}\n"
        f"탐색한 위치:\n"
        + "\n".join(f"  - {d}" for d in _SEARCH_DIRS)
        + "\n\n→ .env 파일에 PITCHER_DATA_ROOT 를 지정하세요 (.env.example 참고).\n"
        "→ 또는 1_statcast/ 노트북으로 데이터를 재생성하세요."
    )


def check(require_data: bool = True) -> None:
    """노트북 첫 셀에서 호출. 환경과 데이터 상태를 출력한다."""
    print(f'환경   : {"Colab" if IN_COLAB else "로컬"}')
    print(f"ROOT   : {ROOT}")
    print(f"DATA   : {DATA_ROOT}")

    if not DATA_ROOT.exists():
        msg = (
            f"\n데이터 폴더가 없습니다: {DATA_ROOT}\n"
            f"→ .env 파일에 PITCHER_DATA_ROOT=<0_data 경로> 를 지정하세요.\n"
            f"  (.env.example 을 .env 로 복사한 뒤 수정)\n"
            f"→ 데이터는 용량(약 614MB) 때문에 저장소에 포함되지 않습니다."
        )
        if require_data:
            raise FileNotFoundError(msg)
        print(msg)
        return

    print(f"OUTPUT : {OUTPUT_DIR}")
    print("데이터 : OK")

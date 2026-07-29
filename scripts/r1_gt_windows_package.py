"""Write / verify the Windows R1 Independent GT launcher package (ASCII BAT + CRLF)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANONICAL_BAT = REPO / "scripts" / "windows" / "START_GT_REVIEW.bat"
CANONICAL_REPAIR_BAT = REPO / "scripts" / "windows" / "START_TRAIN_REPAIR.bat"
DEFAULT_WIN_DIR = Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Independent GT")

ALLOWED = frozenset(
    {
        "START_GT_REVIEW.bat",
        "START_TRAIN_REPAIR.bat",
        "START_ACTIVE_LEARNING_REVIEW.bat",
        "README_TR.txt",
        "ACTIVE_LEARNING_README_TR.txt",
        "REVIEW_PROGRESS.html",
        "NO_MODEL_RESULT_YET.txt",
    }
)

# BAT must stay ASCII command text (Turkish belongs in README_TR only).
FORBIDDEN_BAT_SUBSTRINGS = (
    "—",
    "–",
    "yalnız",
    "Tarayıcı",
    "nohup",
    "pkill",
    "conda activate",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assert_bat_safe(text: str, *, require_repair: bool = False) -> None:
    if text.startswith("\ufeff"):
        raise ValueError("BAT_MUST_NOT_HAVE_BOM")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"BAT_NOT_ASCII:{exc}") from exc
    for bad in FORBIDDEN_BAT_SUBSTRINGS:
        if bad in text:
            raise ValueError(f"BAT_FORBIDDEN_TOKEN:{bad}")
    if "wsl -e bash -lc" in text and "nohup" in text:
        raise ValueError("BAT_EPHEMERAL_NOHUP_PATTERN")
    if re.search(r"&\s*\"\s*$", text, re.M) or '2>&1 &"' in text:
        raise ValueError("BAT_BACKGROUND_AMPERSAND_IN_WSL")
    if "Ubuntu-22.04" not in text:
        raise ValueError("BAT_MISSING_DISTRO")
    if "start_r1_gt_review.sh" not in text:
        raise ValueError("BAT_MISSING_WRAPPER")
    if "CHECK_HEALTH" not in text and "Invoke-RestMethod" not in text:
        raise ValueError("BAT_MISSING_HEALTH_GATE")
    if "Browser was NOT opened" not in text:
        raise ValueError("BAT_MISSING_FAIL_PATH")
    if require_repair:
        if "train-empty-complete" not in text:
            raise ValueError("BAT_MISSING_REPAIR_MODE")
        if "--repair" not in text:
            raise ValueError("BAT_MISSING_REPAIR_FLAG")


def _normalize_bat(raw: str) -> str:
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def load_canonical_bat() -> str:
    text = _normalize_bat(CANONICAL_BAT.read_text(encoding="utf-8"))
    assert_bat_safe(text)
    return text


def load_canonical_repair_bat() -> str:
    text = _normalize_bat(CANONICAL_REPAIR_BAT.read_text(encoding="utf-8"))
    assert_bat_safe(text, require_repair=True)
    return text


def bat_windows_bytes(text: str | None = None) -> bytes:
    body = load_canonical_bat() if text is None else text.replace("\r\n", "\n")
    assert_bat_safe(body)
    return body.replace("\n", "\r\n").encode("ascii")


def repair_bat_windows_bytes(text: str | None = None) -> bytes:
    body = load_canonical_repair_bat() if text is None else text.replace("\r\n", "\n")
    assert_bat_safe(body, require_repair=True)
    return body.replace("\n", "\r\n").encode("ascii")


def write_windows_package(
    win_dir: Path | None = None,
    *,
    include_repair: bool = True,
) -> dict[str, str]:
    win_dir = win_dir or DEFAULT_WIN_DIR
    win_dir.mkdir(parents=True, exist_ok=True)
    for p in list(win_dir.iterdir()):
        if p.is_file() and p.name not in ALLOWED:
            p.unlink()

    bat_bytes = bat_windows_bytes()
    bat_path = win_dir / "START_GT_REVIEW.bat"
    bat_path.write_bytes(bat_bytes)

    repair_sha = ""
    if include_repair:
        repair_bytes = repair_bat_windows_bytes()
        repair_path = win_dir / "START_TRAIN_REPAIR.bat"
        repair_path.write_bytes(repair_bytes)
        repair_sha = sha256_bytes(repair_path.read_bytes())
        if repair_sha != sha256_bytes(repair_bat_windows_bytes()):
            raise RuntimeError("REPAIR_BAT_SHA_MISMATCH")

    readme = """R1 TRAIN REPAIR (FIX2)
=======================

Amac: Yalniz 37 hatali TRAIN karesini duzeltmek.
Dev ve holdout'taki 40 kare KORUNUR — dokunmayin.

NASIL BASLATILIR (bu tur)
-------------------------
1) START_TRAIN_REPAIR.bat dosyasina CIFTE TIKLAYIN.
2) "R1-GT-Repair-Server" penceresi acilir — KAPATMAYIN.
3) Tarayici yalniz health OK olunca acilir.
4) Terminale komut YAZMAYIN.

NE YAPACAKSINIZ
---------------
- Turuncu kutular = model onerisi (henuz GT degil).
- Camgobeği kutular = sizin kabul ettiginiz insan.
- "Karedeki tum onerileri insan olarak kabul et" ile toplu kabul.
- Sonra yanlis kutulari silin, eksikleri cizin, kutulari duzeltin.
- Metadata: Y/W/R/G/U/O (Shift+click ile coklu secim).
- Pending oneri varken Complete CALISMAZ.
- Insan yoksa once onerileri reddedin, sonra "gorunur insan yok" kutusunu isaretleyin.
- Repair sayaci: 0/37 -> 37/37 (ana 80'den ayri).

DIGER
-----
START_GT_REVIEW.bat = eski tam 80-kare inceleme (bu turda GEREKMEZ).

Freeze / fine-tuning / R2 YOK. Bitince Cursor'a haber verin.
"""
    (win_dir / "README_TR.txt").write_text(readme.replace("\n", "\r\n"), encoding="utf-8")
    (win_dir / "NO_MODEL_RESULT_YET.txt").write_text(
        "NO MODEL RESULT YET\r\n"
        "This folder is for independent human GT labeling / train repair.\r\n"
        "No detector acceptance / fine-tuning / R2 package here.\r\n",
        encoding="ascii",
    )
    (win_dir / "REVIEW_PROGRESS.html").write_text(
        "<!DOCTYPE html>\r\n"
        '<html lang="tr"><head><meta charset="utf-8"/>\r\n'
        "<title>R1 GT Train Repair</title>\r\n"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;"
        "background:#111;color:#eee}a{color:#8cf}</style></head><body>\r\n"
        "<h1>TRAIN REPAIR — 37 KARE</h1>\r\n"
        "<p>Cift tik: <b>START_TRAIN_REPAIR.bat</b></p>\r\n"
        '<p><a href="http://127.0.0.1:8766/?repair=train-empty-complete">'
        "http://127.0.0.1:8766/</a></p>\r\n"
        "<p>Turuncu=proposal, camgobeği=accepted human. Freeze yok.</p>\r\n"
        "</body></html>\r\n",
        encoding="utf-8",
    )

    mirror_sha = sha256_bytes(bat_path.read_bytes())
    repo_sha = sha256_bytes(bat_windows_bytes())
    if mirror_sha != repo_sha:
        raise RuntimeError(f"BAT_SHA_MISMATCH mirror={mirror_sha} repo={repo_sha}")
    return {
        "bat_sha256": mirror_sha,
        "repair_bat_sha256": repair_sha,
        "bat_path": str(bat_path),
        "canonical_bat": str(CANONICAL_BAT),
        "canonical_repair_bat": str(CANONICAL_REPAIR_BAT),
    }


__all__ = [
    "ALLOWED",
    "CANONICAL_BAT",
    "CANONICAL_REPAIR_BAT",
    "DEFAULT_WIN_DIR",
    "assert_bat_safe",
    "bat_windows_bytes",
    "load_canonical_bat",
    "load_canonical_repair_bat",
    "repair_bat_windows_bytes",
    "sha256_bytes",
    "write_windows_package",
]

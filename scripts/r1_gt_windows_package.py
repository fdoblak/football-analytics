"""Write / verify the Windows R1 Independent GT launcher package (ASCII BAT + CRLF)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANONICAL_BAT = REPO / "scripts" / "windows" / "START_GT_REVIEW.bat"
DEFAULT_WIN_DIR = Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Independent GT")

ALLOWED = frozenset(
    {
        "START_GT_REVIEW.bat",
        "README_TR.txt",
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


def assert_bat_safe(text: str) -> None:
    if text.startswith("\ufeff"):
        raise ValueError("BAT_MUST_NOT_HAVE_BOM")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"BAT_NOT_ASCII:{exc}") from exc
    for bad in FORBIDDEN_BAT_SUBSTRINGS:
        if bad in text:
            raise ValueError(f"BAT_FORBIDDEN_TOKEN:{bad}")
    # Dangerous quoting / ephemeral background patterns from FIX1 root cause.
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


def load_canonical_bat() -> str:
    raw = CANONICAL_BAT.read_text(encoding="utf-8")
    # Normalize to LF in repo; Windows write uses CRLF.
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    assert_bat_safe(text)
    return text


def bat_windows_bytes(text: str | None = None) -> bytes:
    body = load_canonical_bat() if text is None else text.replace("\r\n", "\n")
    assert_bat_safe(body)
    return body.replace("\n", "\r\n").encode("ascii")


def write_windows_package(win_dir: Path | None = None) -> dict[str, str]:
    win_dir = win_dir or DEFAULT_WIN_DIR
    win_dir.mkdir(parents=True, exist_ok=True)
    for p in list(win_dir.iterdir()):
        if p.is_file() and p.name not in ALLOWED:
            p.unlink()

    bat_bytes = bat_windows_bytes()
    bat_path = win_dir / "START_GT_REVIEW.bat"
    bat_path.write_bytes(bat_bytes)

    readme = """R1 Bagimsiz Insan GT Incelemesi
================================

Bu arac, kendi mac videonuzda sinirli sayida kareyi ETIKETLEMENIZ icindir.
Model dogrulugu / fine-tuning / R2 henuz baslamaz.

NASIL BASLATILIR
----------------
1) START_GT_REVIEW.bat dosyasina cift tiklayin.
2) Ayrica "R1-GT-Server" baslikli bir pencere acilir — bunu KAPATMAYIN.
3) Tarayici yalniz sunucu hazir olunca (health OK) acilir.
4) "127.0.0.1 baglantiyi reddetti" gorurseniz R1-GT-Server penceresine
   ve Linux loguna bakin:
   /home/fdoblak/workspace/independent_gt_review/own_video_97b298e4/server_wrapper.log
5) Terminale komut YAZMAYIN — cift tiklamak yeterlidir.

KURALLAR (kisa)
---------------
- Bir kutu = bir insan (siki, bastan ayaga).
- Iki kisiyi tek kutuya almayin.
- Forma numarasi okunmuyorsa yazmayin.
- DEV ve HOLDOUT kordur: otomatik kutu yok; sifirdan cizin.
- TRAIN'deki mavi oneri kutulari GT degildir.

KLAVYE
------
Y sari oyuncu / W beyaz oyuncu / R hakem / G kaleci / U unknown
O saha disi / C complete+next / Delete secili kutuyu sil

DURDURMA
--------
R1-GT-Server penceresini kapatmak sunucuyu durdurur.
Launcher penceresinde ENTER ile launcher kapanir (server ayri kalabilir).

NOT
---
Henuz freeze / acceptance yok. Bitince Cursor'a haber verin.
"""
    # README may use ASCII-fold Turkish to avoid encoding surprises on Notepad;
    # content is still Turkish-readable without diacritics for launcher clarity.
    (win_dir / "README_TR.txt").write_text(readme.replace("\n", "\r\n"), encoding="utf-8")
    (win_dir / "NO_MODEL_RESULT_YET.txt").write_text(
        "NO MODEL RESULT YET\r\n"
        "This folder is only for independent human GT labeling.\r\n"
        "No detector acceptance / fine-tuning / R2 package here.\r\n",
        encoding="ascii",
    )
    (win_dir / "REVIEW_PROGRESS.html").write_text(
        "<!DOCTYPE html>\r\n"
        '<html lang="tr"><head><meta charset="utf-8"/>\r\n'
        "<title>R1 GT Progress</title>\r\n"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;"
        "background:#111;color:#eee}a{color:#8cf}</style></head><body>\r\n"
        "<h1>R1 Independent GT</h1>\r\n"
        "<p>Cift tik: <b>START_GT_REVIEW.bat</b></p>\r\n"
        '<p><a href="http://127.0.0.1:8766/">http://127.0.0.1:8766/</a></p>\r\n'
        "<p>Server hazir olunca tarayici acilir. Model sonucu yok.</p>\r\n"
        "</body></html>\r\n",
        encoding="utf-8",
    )

    mirror_sha = sha256_bytes(bat_path.read_bytes())
    repo_sha = sha256_bytes(bat_windows_bytes())
    if mirror_sha != repo_sha:
        raise RuntimeError(f"BAT_SHA_MISMATCH mirror={mirror_sha} repo={repo_sha}")
    return {
        "bat_sha256": mirror_sha,
        "bat_path": str(bat_path),
        "canonical_bat": str(CANONICAL_BAT),
    }


__all__ = [
    "ALLOWED",
    "CANONICAL_BAT",
    "DEFAULT_WIN_DIR",
    "assert_bat_safe",
    "bat_windows_bytes",
    "load_canonical_bat",
    "sha256_bytes",
    "write_windows_package",
]

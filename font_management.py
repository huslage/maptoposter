"""
Font Management Module
Handles font loading, Google Fonts integration, and caching.
"""

import os
import re
from pathlib import Path
from typing import Optional

import requests

FONTS_DIR = "fonts"
FONTS_CACHE_DIR = Path(FONTS_DIR) / "cache"


def download_google_font(font_family: str, weights: list = None) -> Optional[dict]:
    """
    Download a font family from Google Fonts and cache it locally.
    Returns dict with font paths for different weights, or None if download fails.
    """
    if weights is None:
        weights = [300, 400, 700]

    FONTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    font_name_safe = font_family.replace(" ", "_").lower()
    font_files = {}

    try:
        weights_str = ";".join(map(str, weights))
        response = requests.get(
            "https://fonts.googleapis.com/css2",
            params={"family": f"{font_family}:wght@{weights_str}"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()

        weight_url_map = {}
        for block in re.split(r"@font-face\s*\{", response.text)[1:]:
            weight_match = re.search(r"font-weight:\s*(\d+)", block)
            if not weight_match:
                continue
            url_match = re.search(r"url\((https://[^)]+\.(woff2|ttf))\)", block)
            if url_match:
                weight_url_map[int(weight_match.group(1))] = url_match.group(1)

        weight_map = {300: "light", 400: "regular", 700: "bold"}

        for weight in weights:
            weight_key = weight_map.get(weight, "regular")
            weight_url = weight_url_map.get(weight)

            if not weight_url and weight_url_map:
                closest = min(weight_url_map.keys(), key=lambda x: abs(x - weight))
                weight_url = weight_url_map[closest]

            if not weight_url:
                continue

            file_ext = "woff2" if weight_url.endswith(".woff2") else "ttf"
            font_path = FONTS_CACHE_DIR / f"{font_name_safe}_{weight_key}.{file_ext}"

            if not font_path.exists():
                print(f"    {weight_key}...", end="", flush=True)
                try:
                    r = requests.get(weight_url, timeout=10)
                    r.raise_for_status()
                    font_path.write_bytes(r.content)
                    print("✓")
                except Exception:
                    print("✗")
                    continue

            font_files[weight_key] = str(font_path)

        if "regular" not in font_files and font_files:
            font_files["regular"] = list(font_files.values())[0]
        if "bold" not in font_files and "regular" in font_files:
            font_files["bold"] = font_files["regular"]
        if "light" not in font_files and "regular" in font_files:
            font_files["light"] = font_files["regular"]

        return font_files if font_files else None

    except Exception:
        return None


def load_fonts(font_family: Optional[str] = None) -> Optional[dict]:
    """
    Load fonts from local directory or download from Google Fonts.
    Returns dict with 'bold', 'regular', 'light' keys, or None on failure.
    """
    if font_family and font_family.lower() != "roboto":
        fonts = download_google_font(font_family)
        if fonts:
            return fonts

    fonts = {
        "bold": os.path.join(FONTS_DIR, "Roboto-Bold.ttf"),
        "regular": os.path.join(FONTS_DIR, "Roboto-Regular.ttf"),
        "light": os.path.join(FONTS_DIR, "Roboto-Light.ttf"),
    }

    for path in fonts.values():
        if not os.path.exists(path):
            return None

    return fonts

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Create and activate virtual environment (already present as pyvenv.cfg)
python -m venv .
source bin/activate  # macOS/Linux
# or: bin/Activate.ps1  (Windows PowerShell)

pip install -r requirements.txt
```

## Running the Script

```bash
# Basic usage
python create_map_poster.py --city "New York" --country "USA" --theme noir --distance 12000

# Short flags: -c city, -C country, -t theme, -d distance (meters)
python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000

# PDF output
python create_map_poster.py -c "Paris" -C "France" -t noir --format pdf

# Custom physical size — A3 portrait in mm, or in inches
python create_map_poster.py -c "Venice" -C "Italy" -t blueprint --width 297 --height 420 --unit mm
python create_map_poster.py -c "London" -C "UK" -t noir --width 18 --height 24 --unit in

# Google Font (downloads & caches in fonts/google_fonts/)
python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige --font "Playfair Display"

# List available themes
python create_map_poster.py --list-themes
```

Output is saved to `posters/` as `{city}_{theme}_{YYYYMMDD_HHMMSS}.{png|pdf}`.

## Architecture

Single-file application (`create_map_poster.py`) with this pipeline:

1. **CLI** (`argparse`) → parses city, country, theme, distance
2. **Geocoding** (`geopy`/Nominatim) → city name → lat/lon coordinates
3. **Data fetching** (`osmnx`) → downloads street network, water, and parks from OpenStreetMap
4. **Rendering** (`matplotlib`) → composites layers in z-order:
   - z=0: background color
   - z=1: water polygons
   - z=2: parks polygons
   - z=3: road network (via `ox.plot_graph`)
   - z=10: gradient fades (top and bottom, 25% of height each)
   - z=11: text labels (city, country, coordinates, attribution)
5. **Output** → saves 300 DPI PNG to `posters/`

The global `THEME` dict is loaded from a JSON file in `themes/` and referenced throughout rendering. Road colors and widths are assigned per-edge by OSM `highway` tag via `get_edge_colors_by_type()` and `get_edge_widths_by_type()`.

## Themes

Each theme is a JSON file in `themes/` with these required keys:

```json
{
  "name": "...", "description": "...",
  "bg": "#hex", "text": "#hex", "gradient_color": "#hex",
  "water": "#hex", "parks": "#hex",
  "road_motorway": "#hex", "road_primary": "#hex",
  "road_secondary": "#hex", "road_tertiary": "#hex",
  "road_residential": "#hex", "road_default": "#hex"
}
```

To add a new theme, create a JSON file in `themes/` — it's automatically discovered by `get_available_themes()`.

## Distance Guide

| Distance | Use case |
|----------|----------|
| 4000–6000m | Small/dense cities (Venice, Amsterdam) |
| 8000–12000m | Medium cities (Paris, Barcelona) |
| 15000–20000m | Large metros (Tokyo, Mumbai) |

## Key Functions

| Function | Purpose |
|----------|---------|
| `get_coordinates()` | Geocoding via Nominatim; includes 1s rate-limit delay |
| `create_poster()` | Main render pipeline; accepts `width_in`, `height_in`, `fmt` |
| `get_edge_colors_by_type()` | Maps OSM highway tag → theme color |
| `get_edge_widths_by_type()` | Maps OSM highway tag → line width (0.4–1.2) |
| `create_gradient_fade()` | RGBA gradient overlay at top/bottom |
| `load_theme()` | Reads `themes/{name}.json`; falls back to embedded default |
| `load_fonts()` | Returns paths for bundled Roboto bold/regular/light |
| `download_google_font()` | Fetches TTF variants from Google Fonts API; caches in `fonts/google_fonts/` |

### Font sizes and canvas scaling

Font sizes (pt) are multiplied by `width_in / 12.0` so text stays proportional when the canvas size changes. The base sizes at the default 12×16 in canvas are:

| Role | Size (pt) | Font weight |
|------|-----------|-------------|
| City name | 60 | bold |
| Country name | 22 | light |
| Coordinates | 14 | regular |
| Attribution | 8 | light |

### Google Fonts download

`download_google_font(font_name)` hits `fonts.googleapis.com/css?family=Name:300,400,700` with an old IE User-Agent to receive TTF download URLs (vs WOFF2 for modern browsers). Files are cached in `fonts/google_fonts/{Font_Name}/`. If weight 300 is absent the regular (400) is used for the light role.

## Adding a New Map Layer

```python
# In create_poster(), after the parks fetch:
try:
    railways = ox.features_from_point(point, tags={'railway': 'rail'}, dist=dist)
except:
    railways = None

# Then plot before roads (choose appropriate zorder):
if railways is not None and not railways.empty:
    railways.plot(ax=ax, color=THEME['railway'], linewidth=0.5, zorder=2.5)
```

Also add `"railway": "#hex"` to each theme JSON and add a fallback in `load_theme()`.

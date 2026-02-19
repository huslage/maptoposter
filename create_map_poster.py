import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time
import json
import os
import re
import pickle
import hashlib
import gzip
import math
from scipy.ndimage import gaussian_filter, zoom
import requests
from datetime import datetime
import argparse

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"
GRAPH_CACHE_DIR = "cache/graphs"
SRTM_CACHE_DIR = "cache/srtm"

# Default figure dimensions in inches
DEFAULT_WIDTH_IN = 12.0
DEFAULT_HEIGHT_IN = 16.0


def load_fonts():
    """
    Load bundled Roboto fonts from the fonts directory.
    Returns dict with font paths for bold, regular, light weights.
    """
    fonts = {
        'bold': os.path.join(FONTS_DIR, 'Roboto-Bold.ttf'),
        'regular': os.path.join(FONTS_DIR, 'Roboto-Regular.ttf'),
        'light': os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
    }

    for weight, path in fonts.items():
        if not os.path.exists(path):
            print(f"⚠ Font not found: {path}")
            return None

    return fonts


# Old Android User-Agent causes Google Fonts to serve static TTF files.
# IE 6 UA triggers EOT (Embedded OpenType) which FreeType cannot read.
# Modern UA triggers WOFF2 which requires brotli to decompress.
_TTF_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 4.4.2; Nexus 5 Build/KOT49H) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.70 Mobile Safari/537.36"
)


def _is_loadable_font(path):
    """Return True if fontTools can open this file as a font."""
    from fontTools.ttLib import TTFont
    try:
        TTFont(path)
        return True
    except Exception:
        return False


def _normalize_to_static_ttf(raw_path, weight_int, output_path):
    """
    Write a static plain TTF to output_path from raw_path.

    fontTools handles TTF, OTF, and WOFF as input. Setting font.flavor = None
    before saving strips the WOFF wrapper so the output is a plain TTF that
    FreeType and matplotlib's PDF backend can embed.

    If raw_path is a variable font (fvar + wght axis), instantiates at
    weight_int using fontTools.varLib.instancer first.

    Raises RuntimeError on failure.
    """
    from fontTools.ttLib import TTFont
    try:
        font = TTFont(raw_path)
    except Exception as e:
        raise RuntimeError(f"fontTools could not open font: {e}")

    if "fvar" in font:
        from fontTools.varLib.instancer import instantiateVariableFont
        wght_axis = next((a for a in font["fvar"].axes if a.axisTag == "wght"), None)
        if wght_axis:
            w = float(max(wght_axis.minValue, min(wght_axis.maxValue, weight_int)))
            font = instantiateVariableFont(font, {"wght": w})

    # Strip WOFF/WOFF2 wrapper → plain TTF output readable by FreeType
    font.flavor = None
    font.save(output_path)


def _fetch_font_url_for_weight(font_slug, weight, headers):
    """
    Return the font file URL for a single weight via the CSS v2 API, or None.

    Requests each weight individually — batch requests silently return only
    one @font-face block for variable fonts.
    """
    css_url = f"https://fonts.googleapis.com/css2?family={font_slug}:wght@{weight}"
    try:
        resp = requests.get(css_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        url_match = re.search(r"url\(([^)]+)\)", resp.text)
        if url_match:
            return url_match.group(1).strip("'\"")
    except requests.RequestException:
        pass
    return None


def download_google_font(font_name):
    """
    Download a font from Google Fonts and cache static TTF files locally.

    Probes all nine standard weights (100–900) individually so we discover
    exactly what the font provides, then maps bold/regular/light to the
    closest available weight. Variable fonts are instantiated at the target
    weight via fonttools so matplotlib's PDF backend can embed them.

    Cached files live in fonts/google_fonts/<Font_Name>/.
    Returns a dict with keys 'bold', 'regular', 'light' → file paths.
    Raises RuntimeError if the font cannot be found or downloaded.
    """
    cache_dir = os.path.join(FONTS_DIR, "google_fonts", font_name.replace(" ", "_"))
    os.makedirs(cache_dir, exist_ok=True)

    headers = {"User-Agent": _TTF_USER_AGENT}
    font_slug = font_name.replace(" ", "+")

    print(f"Fetching Google Font '{font_name}'...")

    probe_weights = ["100", "200", "300", "400", "500", "600", "700", "800", "900"]
    weight_to_url = {}
    for w in probe_weights:
        url = _fetch_font_url_for_weight(font_slug, w, headers)
        if url:
            weight_to_url[w] = url

    if not weight_to_url:
        raise RuntimeError(
            f"Font '{font_name}' not found on Google Fonts. "
            "Check the spelling (e.g. 'Inter', 'Open Sans', 'Montserrat')."
        )

    available = sorted(weight_to_url, key=int)

    def closest(target):
        return min(available, key=lambda w: abs(int(w) - int(target)))

    role_weights = {
        "bold":    closest("700"),
        "regular": closest("400"),
        "light":   closest("300"),
    }

    for role, weight in role_weights.items():
        targets = {"bold": "700", "regular": "400", "light": "300"}
        if weight != targets[role]:
            print(f"  ⚠ Weight {targets[role]} not available; using {weight} for {role}")

    result = {}
    for role, weight in role_weights.items():
        filepath = os.path.join(cache_dir, f"{role}_{weight}.ttf")

        # Discard cached files that fontTools can no longer open (e.g. stale
        # EOT files written by the old IE User-Agent before this fix).
        if os.path.exists(filepath) and not _is_loadable_font(filepath):
            print(f"  Stale cached file for {role} is not a valid font — re-downloading...")
            os.remove(filepath)

        if not os.path.exists(filepath):
            print(f"  Downloading {font_name} weight {weight} ({role})...")
            try:
                font_resp = requests.get(weight_to_url[weight], timeout=15)
                font_resp.raise_for_status()
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download font file: {e}")

            raw_path = filepath + ".raw"
            with open(raw_path, "wb") as f:
                f.write(font_resp.content)

            # Normalize to plain static TTF (converts WOFF, instantiates variable fonts)
            try:
                _normalize_to_static_ttf(raw_path, int(weight), filepath)
                os.remove(raw_path)
            except RuntimeError as e:
                os.remove(raw_path)
                raise RuntimeError(
                    f"Could not process downloaded font '{font_name}' weight {weight}: {e}"
                )
        else:
            print(f"  Using cached {font_name} weight {weight} ({role})")

        result[role] = filepath

    print(f"✓ Font '{font_name}' ready")
    return result


def generate_output_filename(city, theme_name, fmt="png"):
    """
    Generate unique output filename with city, theme, datetime, and format extension.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(" ", "_")
    filename = f"{city_slug}_{theme_name}_{timestamp}.{fmt}"
    return os.path.join(POSTERS_DIR, filename)


def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []

    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith(".json"):
            themes.append(file[:-5])
    return themes


def load_theme(theme_name="feature_based"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")

    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default feature_based theme.")
        return {
            "name": "Feature-Based Shading",
            "bg": "#FFFFFF",
            "text": "#000000",
            "gradient_color": "#FFFFFF",
            "water": "#C0C0C0",
            "parks": "#F0F0F0",
            "road_motorway": "#0A0A0A",
            "road_primary": "#1A1A1A",
            "road_secondary": "#2A2A2A",
            "road_tertiary": "#3A3A3A",
            "road_residential": "#4A4A4A",
            "road_default": "#3A3A3A",
        }

    with open(theme_file, "r") as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if "description" in theme:
            print(f"  {theme['description']}")
        return theme


# Globals set at runtime after arg parsing
THEME = None
FONTS = None


def create_gradient_fade(ax, color, location="bottom", zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))

    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]

    if location == "bottom":
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end

    ax.imshow(
        gradient,
        extent=[xlim[0], xlim[1], y_bottom, y_top],
        aspect="auto",
        cmap=custom_cmap,
        zorder=zorder,
        origin="lower",
    )


def get_edge_colors_by_type(G):
    """
    Assigns colors to edges based on road type hierarchy.
    """
    edge_colors = []
    for u, v, data in G.edges(data=True):
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0] if highway else "unclassified"

        if highway in ["motorway", "motorway_link"]:
            color = THEME["road_motorway"]
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            color = THEME["road_primary"]
        elif highway in ["secondary", "secondary_link"]:
            color = THEME["road_secondary"]
        elif highway in ["tertiary", "tertiary_link"]:
            color = THEME["road_tertiary"]
        elif highway in ["residential", "living_street", "unclassified"]:
            color = THEME["road_residential"]
        else:
            color = THEME["road_default"]

        edge_colors.append(color)
    return edge_colors


def get_edge_widths_by_type(G):
    """
    Assigns line widths to edges based on road type.
    """
    edge_widths = []
    for u, v, data in G.edges(data=True):
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0] if highway else "unclassified"

        if highway in ["motorway", "motorway_link"]:
            width = 1.2
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            width = 1.0
        elif highway in ["secondary", "secondary_link"]:
            width = 0.8
        elif highway in ["tertiary", "tertiary_link"]:
            width = 0.6
        else:
            width = 0.4

        edge_widths.append(width)
    return edge_widths


# ---------------------------------------------------------------------------
# Elevation / topographic helpers
# ---------------------------------------------------------------------------

def _srtm_tile_name(lat_int, lon_int):
    ns = 'N' if lat_int >= 0 else 'S'
    ew = 'E' if lon_int >= 0 else 'W'
    return f"{ns}{abs(lat_int):02d}{ew}{abs(lon_int):03d}"


def _download_srtm_tile(lat_int, lon_int):
    """Fetch one SRTM tile from the public terrain-tiles S3 bucket."""
    name = _srtm_tile_name(lat_int, lon_int)
    folder = name[:3]
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{folder}/{name}.hgt.gz"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            return None
        raw = gzip.decompress(resp.content)
        n_vals = len(raw) // 2
        side = int(round(n_vals ** 0.5))
        arr = np.frombuffer(raw, dtype=np.dtype('>i2')).reshape(side, side).astype(np.float32)
        arr[arr == -32768] = np.nan
        return arr
    except Exception:
        return None


def fetch_elevation_grid(point, dist):
    """
    Return (lons_1d, lats_1d, elev_grid) covering the bounding box.
    SRTM tiles (1 arc-second, ~30 m) are downloaded once and cached as .npy files.
    Returns (None, None, None) if no data could be fetched.
    """
    os.makedirs(SRTM_CACHE_DIR, exist_ok=True)
    lat, lon = point
    lat_deg = dist / 111000.0
    lon_deg = dist / (111000.0 * math.cos(math.radians(lat)))
    lat_min, lat_max = lat - lat_deg, lat + lat_deg
    lon_min, lon_max = lon - lon_deg, lon + lon_deg

    lat_tiles = list(range(int(math.floor(lat_min)), int(math.floor(lat_max)) + 1))
    lon_tiles = list(range(int(math.floor(lon_min)), int(math.floor(lon_max)) + 1))

    tile_data = {}
    side = None
    for tlat in lat_tiles:
        for tlon in lon_tiles:
            name = _srtm_tile_name(tlat, tlon)
            cache_file = os.path.join(SRTM_CACHE_DIR, f"{name}.npy")
            if os.path.exists(cache_file):
                arr = np.load(cache_file)
            else:
                print(f"  Fetching elevation tile {name}...")
                arr = _download_srtm_tile(tlat, tlon)
                if arr is not None:
                    np.save(cache_file, arr)
            if arr is not None:
                tile_data[(tlat, tlon)] = arr
                side = arr.shape[0]

    if not tile_data or side is None:
        return None, None, None

    # Stitch tiles: sorted N→S in rows, W→E in columns
    n_lat = len(lat_tiles)
    n_lon = len(lon_tiles)
    big = np.full((n_lat * side, n_lon * side), np.nan, dtype=np.float32)
    for row_i, tlat in enumerate(sorted(lat_tiles, reverse=True)):
        for col_j, tlon in enumerate(sorted(lon_tiles)):
            if (tlat, tlon) in tile_data:
                big[row_i * side:(row_i + 1) * side,
                    col_j * side:(col_j + 1) * side] = tile_data[(tlat, tlon)]

    # Full lat/lon extents of stitched grid (row 0 = northernmost)
    full_lat_max = float(max(lat_tiles) + 1)
    full_lat_min = float(min(lat_tiles))
    full_lon_min = float(min(lon_tiles))
    full_lon_max = float(max(lon_tiles) + 1)
    lats_1d = np.linspace(full_lat_max, full_lat_min, n_lat * side, endpoint=False)
    lons_1d = np.linspace(full_lon_min, full_lon_max, n_lon * side, endpoint=False)

    # Crop to bounding box
    lat_mask = (lats_1d >= lat_min) & (lats_1d <= lat_max)
    lon_mask = (lons_1d >= lon_min) & (lons_1d <= lon_max)
    return lons_1d[lon_mask], lats_1d[lat_mask], big[np.ix_(lat_mask, lon_mask)]


def get_coordinates(city, country):
    """
    Fetches coordinates for a given city and country using geopy.
    Includes rate limiting to be respectful to the geocoding service.
    """
    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster")
    time.sleep(1)

    location = geolocator.geocode(f"{city}, {country}")

    if location:
        print(f"✓ Found: {location.address}")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        return (location.latitude, location.longitude)
    else:
        raise ValueError(f"Could not find coordinates for {city}, {country}")


def create_poster(city, country, point, dist, output_file,
                  width_in=DEFAULT_WIDTH_IN, height_in=DEFAULT_HEIGHT_IN, fmt="png",
                  title=None, topo=False):
    """
    Render a map poster and save it.

    Parameters
    ----------
    width_in, height_in : float
        Figure dimensions in inches. Font sizes scale proportionally
        relative to the default 12×16 in canvas.
    fmt : str
        Output format passed to matplotlib: 'png' or 'pdf'.
    """
    print(f"\nGenerating map for {city}, {country}...")
    print(f"  Canvas: {width_in:.2f} × {height_in:.2f} in  |  Format: {fmt.upper()}")

    # Scale font sizes relative to the default 12-inch wide canvas
    font_scale = width_in / DEFAULT_WIDTH_IN

    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
    cache_key = hashlib.md5(f"{point[0]:.6f},{point[1]:.6f},{dist}".encode()).hexdigest()

    if topo:
        # --- Topographic mode: elevation contours ---
        print("Fetching elevation data...")
        elev_lons, elev_lats, elev_grid = fetch_elevation_grid(point, dist)
        if elev_grid is None:
            raise RuntimeError("Could not download elevation data for this location.")

        # Still fetch water/parks for overlaying
        water_pickle = os.path.join(GRAPH_CACHE_DIR, f"{cache_key}_water_parks.pkl")
        if os.path.exists(water_pickle):
            print("Loading water/parks from cache...")
            with open(water_pickle, "rb") as f:
                water, parks = pickle.load(f)
        else:
            print("  Downloading water features...")
            try:
                water = ox.features_from_point(
                    point, tags={"natural": "water", "waterway": "riverbank"}, dist=dist
                )
            except Exception:
                water = None
            print("  Downloading parks/green spaces...")
            try:
                parks = ox.features_from_point(
                    point, tags={"leisure": "park", "landuse": "grass"}, dist=dist
                )
            except Exception:
                parks = None
            with open(water_pickle, "wb") as f:
                pickle.dump((water, parks), f)

        # Setup plot
        print("Rendering topographic map...")
        fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor=THEME["bg"])
        ax.set_facecolor(THEME["bg"])
        ax.set_position([0, 0, 1, 1])

        # Set axes bounds to match the bounding box
        lat_deg = dist / 111000.0
        lon_deg = dist / (111000.0 * math.cos(math.radians(point[0])))
        ax.set_xlim(point[1] - lon_deg, point[1] + lon_deg)
        ax.set_ylim(point[0] - lat_deg, point[0] + lat_deg)
        ax.set_aspect("equal")
        ax.axis("off")

        # Layer 1: water & parks
        if water is not None and not water.empty:
            water.plot(ax=ax, facecolor=THEME["water"], edgecolor="none", zorder=1)
        if parks is not None and not parks.empty:
            parks.plot(ax=ax, facecolor=THEME["parks"], edgecolor="none", zorder=2)

        # Layer 2: elevation contours
        elev_filled = np.where(np.isnan(elev_grid), 0.0, elev_grid)
        elev_min = float(np.nanmin(elev_grid))
        elev_max = float(np.nanmax(elev_grid))
        elev_range = elev_max - elev_min

        # Upsample then Gaussian-blur so contour lines flow continuously
        # rather than looking jagged/fragmented from raw SRTM pixels.
        upsample = 3
        elev_smooth = zoom(elev_filled, upsample, order=3)        # bicubic upsample
        elev_smooth = gaussian_filter(elev_smooth, sigma=upsample * 3)  # smooth
        lons_smooth = np.linspace(elev_lons[0], elev_lons[-1], elev_smooth.shape[1])
        lats_smooth = np.linspace(elev_lats[0], elev_lats[-1], elev_smooth.shape[0])

        # Pick an interval that gives roughly 10–20 minor contours
        for interval in [5, 10, 20, 25, 50, 100, 200, 500]:
            if elev_range / interval <= 20:
                break

        levels = np.arange(
            math.floor(elev_min / interval) * interval,
            elev_max + interval,
            interval,
        )

        ax.contour(
            lons_smooth, lats_smooth, elev_smooth,
            levels=levels,
            colors=[THEME["road_secondary"]], alpha=0.5, linewidths=0.4, zorder=3,
        )
        major_levels = levels[::5]
        ax.contour(
            lons_smooth, lats_smooth, elev_smooth,
            levels=major_levels,
            colors=[THEME["road_primary"]], alpha=0.9, linewidths=0.9, zorder=4,
        )

    else:
        # --- Road map mode (default) ---
        graph_pickle = os.path.join(GRAPH_CACHE_DIR, f"{cache_key}.pkl")
        if os.path.exists(graph_pickle):
            print("Loading map data from local cache...")
            with open(graph_pickle, "rb") as f:
                G, water, parks = pickle.load(f)
            print("✓ Loaded from cache!")
        else:
            print("  [1/3] Downloading street network (this may take a moment)...")
            G = ox.graph_from_point(point, dist=dist, dist_type="bbox", network_type="all")
            print("  [2/3] Downloading water features...")
            try:
                water = ox.features_from_point(
                    point, tags={"natural": "water", "waterway": "riverbank"}, dist=dist
                )
            except Exception:
                water = None
            print("  [3/3] Downloading parks/green spaces...")
            try:
                parks = ox.features_from_point(
                    point, tags={"leisure": "park", "landuse": "grass"}, dist=dist
                )
            except Exception:
                parks = None
            print("✓ All data downloaded — saving to local cache...")
            with open(graph_pickle, "wb") as f:
                pickle.dump((G, water, parks), f)
            print("✓ Cached!")

        # Setup plot
        print("Rendering map...")
        fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor=THEME["bg"])
        ax.set_facecolor(THEME["bg"])
        ax.set_position([0, 0, 1, 1])

        # Layer 1: water & parks polygons
        if water is not None and not water.empty:
            water.plot(ax=ax, facecolor=THEME["water"], edgecolor="none", zorder=1)
        if parks is not None and not parks.empty:
            parks.plot(ax=ax, facecolor=THEME["parks"], edgecolor="none", zorder=2)

        # Layer 2: roads with hierarchy coloring
        print("Applying road hierarchy colors...")
        edge_colors = get_edge_colors_by_type(G)
        edge_widths = get_edge_widths_by_type(G)

        ox.plot_graph(
            G, ax=ax, bgcolor=THEME["bg"],
            node_size=0,
            edge_color=edge_colors,
            edge_linewidth=edge_widths,
            show=False, close=False,
        )

    # Layer 3: gradient fades
    create_gradient_fade(ax, THEME["gradient_color"], location="bottom", zorder=10)
    create_gradient_fade(ax, THEME["gradient_color"], location="top", zorder=10)

    # Layer 4: typography (sizes scale with canvas width)
    # City name: letter-space short names; fit-to-width for longer ones.
    # Bold proportional font: character width ≈ font_size_pt * 0.55
    city_upper = (title if title else city).upper()
    if len(city_upper) <= 12:
        city_display = "  ".join(list(city_upper))
    else:
        city_display = city_upper
    max_city_pts = width_in * 72 * 0.80  # 80% of canvas width — 10% margin each side
    city_size = min(60 * font_scale, max_city_pts / (len(city_display) * 0.55))

    if FONTS:
        font_main = FontProperties(fname=FONTS["bold"], size=city_size)
        font_sub = FontProperties(fname=FONTS["light"], size=22 * font_scale)
        font_coords = FontProperties(fname=FONTS["regular"], size=14 * font_scale)
        font_attr = FontProperties(fname=FONTS["light"], size=8 * font_scale)
    else:
        font_main = FontProperties(family="monospace", weight="bold", size=city_size)
        font_sub = FontProperties(family="monospace", weight="normal", size=22 * font_scale)
        font_coords = FontProperties(family="monospace", size=14 * font_scale)
        font_attr = FontProperties(family="monospace", size=8 * font_scale)

    ax.text(0.5, 0.14, city_display, transform=ax.transAxes,
            color=THEME["text"], ha="center", fontproperties=font_main, zorder=11)

    ax.text(0.5, 0.10, country.upper(), transform=ax.transAxes,
            color=THEME["text"], ha="center", fontproperties=font_sub, zorder=11)

    lat, lon = point
    coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    if lon < 0:
        coords = coords.replace("E", "W")

    ax.text(0.5, 0.07, coords, transform=ax.transAxes,
            color=THEME["text"], alpha=0.7, ha="center", fontproperties=font_coords, zorder=11)

    ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes,
            color=THEME["text"], linewidth=1, zorder=11)

    ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
            color=THEME["text"], alpha=0.5, ha="right", va="bottom",
            fontproperties=font_attr, zorder=11)

    # Save
    print(f"Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, facecolor=THEME["bg"], format=fmt)
    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")


def print_examples():
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Examples:
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000

  # PDF output
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream --format pdf

  # Custom physical size (A3 portrait in mm)
  python create_map_poster.py -c "Venice" -C "Italy" -t blueprint --width 297 --height 420 --unit mm

  # Custom size in inches
  python create_map_poster.py -c "London" -C "UK" -t noir --width 18 --height 24 --unit in

  # Google Font
  python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige --font "Playfair Display"

  # List available themes
  python create_map_poster.py --list-themes

Options:
  --city, -c        City name (required)
  --country, -C     Country name (required)
  --theme, -t       Theme name (default: feature_based)
  --distance, -d    Map radius in meters (default: 29000)
  --format, -f      Output format: png or pdf (default: png)
  --width, -w       Output width in --unit (default: 12 in)
  --height, -H      Output height in --unit (default: 16 in)
  --unit, -u        Unit for width/height: in or mm (default: in)
  --font, -F        Google Font name (default: bundled Roboto)
  --list-themes     List all available themes

Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)
""")


def list_themes():
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return

    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, "r") as f:
                theme_data = json.load(f)
                display_name = theme_data.get("name", theme_name)
                description = theme_data.get("description", "")
        except Exception:
            display_name = theme_name
            description = ""
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --city "New York" --country "USA"
  python create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster.py --city Paris --country France --format pdf
  python create_map_poster.py --city Venice --country Italy --width 297 --height 420 --unit mm
  python create_map_poster.py --city London --country UK --font "Playfair Display"
  python create_map_poster.py --list-themes
        """,
    )

    parser.add_argument("--city", "-c", type=str, help="City name")
    parser.add_argument("--country", "-C", type=str, help="Country name")
    parser.add_argument("--theme", "-t", type=str, default="feature_based",
                        help="Theme name (default: feature_based)")
    parser.add_argument("--distance", "-d", type=int, default=29000,
                        help="Map radius in meters (default: 29000)")
    parser.add_argument("--format", "-f", type=str, default="png",
                        choices=["png", "pdf"],
                        help="Output format (default: png)")
    parser.add_argument("--width", "-w", type=float, default=None,
                        help="Output width in --unit (default: 12 in)")
    parser.add_argument("--height", "-H", type=float, default=None,
                        help="Output height in --unit (default: 16 in)")
    parser.add_argument("--unit", "-u", type=str, default="in",
                        choices=["in", "mm"],
                        help="Unit for --width/--height (default: in)")
    parser.add_argument("--font", "-F", type=str, default=None,
                        help="Google Font name to use instead of bundled Roboto "
                             "(e.g. 'Playfair Display', 'Montserrat', 'Lato')")
    parser.add_argument("--title", "-T", type=str, default=None,
                        help="Custom title printed on the poster instead of the city name")
    parser.add_argument("--topo", action="store_true",
                        help="Render as a topographic contour map instead of a road map")
    parser.add_argument("--list-themes", action="store_true",
                        help="List all available themes")

    args = parser.parse_args()

    if len(os.sys.argv) == 1:
        print_examples()
        os.sys.exit(0)

    if args.list_themes:
        list_themes()
        os.sys.exit(0)

    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        os.sys.exit(1)

    available_themes = get_available_themes()
    if args.theme not in available_themes:
        print(f"Error: Theme '{args.theme}' not found.")
        print(f"Available themes: {', '.join(available_themes)}")
        os.sys.exit(1)

    # Resolve canvas dimensions in inches
    MM_PER_INCH = 25.4
    width_in = args.width if args.width is not None else DEFAULT_WIDTH_IN
    height_in = args.height if args.height is not None else DEFAULT_HEIGHT_IN
    if args.unit == "mm":
        width_in = width_in / MM_PER_INCH
        height_in = height_in / MM_PER_INCH

    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)

    # Load fonts (Google or bundled Roboto)
    if args.font:
        try:
            FONTS = download_google_font(args.font)
        except RuntimeError as e:
            print(f"✗ Font error: {e}")
            os.sys.exit(1)
    else:
        FONTS = load_fonts()

    # Load theme
    THEME = load_theme(args.theme)

    try:
        coords = get_coordinates(args.city, args.country)
        output_file = generate_output_filename(args.city, args.theme, args.format)
        create_poster(
            args.city, args.country, coords, args.distance, output_file,
            width_in=width_in, height_in=height_in, fmt=args.format,
            title=args.title, topo=args.topo,
        )

        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        os.sys.exit(1)

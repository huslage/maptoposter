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
import requests
from datetime import datetime
import argparse

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

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


def download_google_font(font_name):
    """
    Download a font from Google Fonts and cache the TTF files locally.

    Fetches the bold (700), regular (400), and light (300) variants.
    If the 300 weight is unavailable the regular (400) is used for light.

    Returns a dict with keys 'bold', 'regular', 'light' mapping to file paths.
    Raises RuntimeError if the font cannot be found or downloaded.
    """
    cache_dir = os.path.join(FONTS_DIR, "google_fonts", font_name.replace(" ", "_"))
    os.makedirs(cache_dir, exist_ok=True)

    # Use an old User-Agent so the API returns TTF format instead of WOFF2
    headers = {"User-Agent": "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)"}
    font_slug = font_name.replace(" ", "+")
    css_url = f"https://fonts.googleapis.com/css?family={font_slug}:300,400,700"

    print(f"Fetching Google Font '{font_name}'...")
    try:
        resp = requests.get(css_url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Could not fetch '{font_name}' from Google Fonts: {e}")

    css = resp.text

    # Parse @font-face blocks to extract weight → URL mappings
    weight_to_url = {}
    for block in re.findall(r"@font-face\s*\{([^}]+)\}", css, re.DOTALL):
        weight_match = re.search(r"font-weight:\s*(\d+)", block)
        url_match = re.search(r"url\(([^)]+)\)", block)
        if weight_match and url_match:
            weight = weight_match.group(1)
            url = url_match.group(1).strip("'\"")
            weight_to_url[weight] = url

    if not weight_to_url:
        raise RuntimeError(
            f"Font '{font_name}' not found on Google Fonts. "
            "Check the spelling (e.g. 'Roboto', 'Open Sans', 'Montserrat')."
        )

    # Resolve which weight to use for each role
    role_weights = {
        "bold": "700",
        "regular": "400",
        "light": "300" if "300" in weight_to_url else "400",
    }

    result = {}
    for role, weight in role_weights.items():
        if weight not in weight_to_url:
            raise RuntimeError(
                f"Weight {weight} not available for font '{font_name}'. "
                f"Available weights: {sorted(weight_to_url)}"
            )

        filepath = os.path.join(cache_dir, f"{role}_{weight}.ttf")
        if not os.path.exists(filepath):
            print(f"  Downloading {font_name} weight {weight}...")
            try:
                font_resp = requests.get(weight_to_url[weight], timeout=15)
                font_resp.raise_for_status()
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download font file: {e}")
            with open(filepath, "wb") as f:
                f.write(font_resp.content)
        else:
            print(f"  Using cached {font_name} weight {weight}")

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
                  width_in=DEFAULT_WIDTH_IN, height_in=DEFAULT_HEIGHT_IN, fmt="png"):
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

    with tqdm(total=3, desc="Fetching map data", unit="step",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        pbar.set_description("Downloading street network")
        G = ox.graph_from_point(point, dist=dist, dist_type="bbox", network_type="all")
        pbar.update(1)
        time.sleep(0.5)

        pbar.set_description("Downloading water features")
        try:
            water = ox.features_from_point(
                point, tags={"natural": "water", "waterway": "riverbank"}, dist=dist
            )
        except Exception:
            water = None
        pbar.update(1)
        time.sleep(0.3)

        pbar.set_description("Downloading parks/green spaces")
        try:
            parks = ox.features_from_point(
                point, tags={"leisure": "park", "landuse": "grass"}, dist=dist
            )
        except Exception:
            parks = None
        pbar.update(1)

    print("✓ All data downloaded successfully!")

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
    if FONTS:
        font_main = FontProperties(fname=FONTS["bold"], size=60 * font_scale)
        font_sub = FontProperties(fname=FONTS["light"], size=22 * font_scale)
        font_coords = FontProperties(fname=FONTS["regular"], size=14 * font_scale)
        font_attr = FontProperties(fname=FONTS["light"], size=8 * font_scale)
    else:
        font_main = FontProperties(family="monospace", weight="bold", size=60 * font_scale)
        font_sub = FontProperties(family="monospace", weight="normal", size=22 * font_scale)
        font_coords = FontProperties(family="monospace", size=14 * font_scale)
        font_attr = FontProperties(family="monospace", size=8 * font_scale)

    spaced_city = "  ".join(list(city.upper()))

    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
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
        )

        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        os.sys.exit(1)

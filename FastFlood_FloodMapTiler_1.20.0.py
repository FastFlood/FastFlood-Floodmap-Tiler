"""FASTFLOOD USER-FRIENDLY RUNNER
================================

Purpose
-------
Run FastFlood over a user-defined bounding box using two staggered tile grids.
Each grid is blended into a continuous raster, then the two blended rasters are
combined pixel-by-pixel using the maximum valid depth value.

Requirements
------------
* Python 3.10 or newer.
* Packages: numpy and rasterio (install with: python -m pip install numpy rasterio).
* A working FastFlood command-line executable and a valid API key, if required by
  your FastFlood installation.
* A georeferenced DEM covering the requested bounding box.

IMPORTANT
---------
The FastFlood command-line switches are application/version-specific. Before a
production run, verify build_command() against the documentation for the version
of FastFlood installed on your computer. This script does not replace FastFlood;
it prepares tile DEMs, calls FastFlood, and post-processes the output rasters.

USER INPUTS AT A GLANCE
-----------------------
Required for every run:
  1. Processing mode.
  2. Bounding box (west, south, east, north), expressed in the DEM's CRS.
  3. Resolution or named resolution preset.
  4. Output parent folder.
  5. A DEM path, unless DEFAULT_DEM_PATH is configured below.

Optional or defaulted:
  * MULT_MAN and MULT_INF calibration multipliers (both default to 1.0).
  * Keep tile folders and per-tile diagnostic files (defaults to No).
  * API key (defaults to the <KEY> placeholder; replace it or enter a key at run time).

The detailed controls are documented individually in ADVANCED SETTINGS below.
"""
from __future__ import annotations

import csv
import math
import os
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# =============================================================================
# EASY-TO-FIND RUN SETTINGS — EDIT THESE FIRST
# =============================================================================
# RUN_MODE accepts: "FLASH", "HYBRID", "FLUVIAL", "FLASH3H", "FLASH6H",
# "FLUVIAL72H", or "FLUVIAL120H". These names select a sensible mode/duration
# combination automatically. The value is also offered as the interactive
# prompt default; users can choose another valid option when the script starts.
#   FLASH        -> Flash Only, default duration 3 hours, channelmv=300
#   HYBRID       -> Hybrid, default duration 6 hours, no channelmv
#   FLUVIAL      -> Fluvial, default duration 120 hours, no channelmv
#   FLASH3H      -> Flash Only, explicitly 3 hours, channelmv=300
#   FLASH6H      -> Flash Only, explicitly 6 hours, channelmv=900
#   FLUVIAL72H   -> Fluvial, explicitly 72 hours, no channelmv
#   FLUVIAL120H  -> Fluvial, explicitly 120 hours, no channelmv
RUN_MODE = "HYBRID"

# Optional duration override in hours. Leave as None to use the duration implied
# by RUN_MODE (or the interactive choice). Set to 1, 3, or 6 for FLASH/HYBRID;
# set to 72 or 120 for FLUVIAL. An explicit override takes precedence over the
# duration encoded in a mode such as FLASH3H, but must be valid for that mode.
DURATION_HOURS_OVERRIDE = None

# Default resolution shown in the prompt. Named presets are translated using
# RESOLUTION_PRESETS below; literal supported values can also be entered.
DEFAULT_RESOLUTION = "20m"

# Default calibration values. The user can accept these or enter different
# multipliers at the prompts. 1.0 means no scaling relative to the base value.
DEFAULT_MULT_MAN = 1.0
DEFAULT_MULT_INF = 1.0

# Default answer for retaining tile DEMs, per-tile rasters, and diagnostics.
DEFAULT_KEEP_TILES = "no"

# Default API-key prompt value. <KEY> is deliberately a placeholder, not a
# credential; enter a valid key when prompted if your FastFlood setup requires it.
DEFAULT_API_KEY = "<KEY>"

# =============================================================================
# ADVANCED SETTINGS — EDIT ONLY WHEN YOU NEED TO CHANGE THE WORKFLOW
# =============================================================================
# These controls are deliberately kept out of the normal interactive prompts.
# The script uses the values below unless you edit this section.

# --- Required software and coordinate-system assumptions --------------------
# Full path to FastFlood's command-line executable. The environment variable
# FASTFLOOD_EXE takes precedence, which is useful on machines with different
# install locations. Update the fallback path to match your own installation.
FASTFLOOD_EXE = Path(os.environ.get("FASTFLOOD_EXE", r"C:\FastFlood\fastflood.exe"))

# CRS expected for the DEM and for all four bounding-box coordinates. EPSG:32735
# is inherited from the original script; change it if your DEM uses another CRS.
# The DEM itself is not reprojected automatically.
EXPECTED_DEM_CRS = "EPSG:32735"

# Sentinel written to pixels that have no valid data. Keep this consistent
# across tile rasters and mosaics; changing it is generally not necessary.
OUTPUT_NODATA = -9999.0

# --- Tiling and parallel processing -----------------------------------------
# Approximate width/height of each tile in kilometres. Larger tiles mean fewer
# FastFlood jobs but more memory per job. Smaller tiles mean more job overhead.
TILE_SIZE_KM = 50.0

# Fraction of each tile's width/height that overlaps its neighbours. 0.10 means
# 10% overlap. Overlap provides common coverage for smoother blending.
TILE_OVERLAP_PERCENT = 0.10

# Grid B is offset from Grid A by this fraction of one tile stride. 0.50 creates
# the staggered/half-stride second grid used to reduce grid-edge artefacts.
GRID_B_OFFSET_SPACING_FRACTION = 0.50

# Maximum number of tile simulations submitted concurrently. Default requested
# by the user: 8. Lower this if FastFlood, RAM, CPU, or API limits are exceeded.
PROCESSES = 8

# If True, reuse a tile when both depth and discharge outputs already exist and
# pass basic raster-read checks. Set False to rerun every tile from scratch.
SKIP_VALID_EXISTING_TILES = True

# Files smaller than this are rejected as likely incomplete/corrupt raster files.
# This is a sanity check, not a guarantee that the raster is scientifically valid.
MIN_VALID_FILE_BYTES = 1024

# --- Mosaic creation and blending -------------------------------------------
# If True, create a blended depth mosaic for each grid and a final max-combined
# raster. If False, only per-tile FastFlood outputs and run reports are produced.
CREATE_MOSAICS = True

# If False, any failed tile prevents mosaics from being created. Set True only if
# you explicitly accept mosaics with gaps where failed tiles should have contributed.
ALLOW_INCOMPLETE_MOSAIC = False

# Width of the feather/blending edge as a fraction of tile dimensions. By default
# it follows the tile overlap, so edge weights taper across the overlap zone.
BLEND_EDGE_FRACTION = TILE_OVERLAP_PERCENT

# Minimum blending weight assigned to a valid pixel. Prevents edge pixels from
# receiving a mathematically zero weight; must be positive for normal blending.
MIN_BLEND_WEIGHT = 0.01

# Raster block width/height for mosaics and temporary accumulators. GeoTIFF tiled
# block sizes must be multiples of 16; 256 is a practical general-purpose value.
MOSAIC_BLOCK_SIZE = 256

# --- FastFlood model options -------------------------------------------------
# Return period passed to FastFlood (original script used 100). This script keeps
# the original value for all three modes; change only if your scenarios require it.
RETURN_PERIOD = 100

# Value passed to FastFlood's -relmindabcond option. Its exact interpretation is
# FastFlood-version-specific; retain the original value unless you know otherwise.
RELMINDABCOND = 0.01

# Parameters passed after FastFlood's -channel switch. The list is expanded into
# separate command-line arguments. Confirm their meaning/order in your FastFlood
# version before changing them. ChannelMV is controlled automatically by mode.
CHANNEL_ARGS = ["-channel", "1.2", "0.2", "1.2", "0.2", "0.25"]

# Intended interval for progress messages. Progress is also logged per tile.
LOG_EVERY_SECONDS = 15

# Named resolution shortcuts. These are convenience mappings, not a claim that
# one resolution is universally appropriate. Users may instead enter any value
# in VALID_RESOLUTIONS directly. Adjust these mappings to suit your project.
RESOLUTION_PRESETS = {"low": "150m", "medium": "80m", "high": "20m", "veryhigh": "5m"}

# Explicitly accepted literal resolution strings. FastFlood must support the
# chosen resolution; this list only validates the input before calling FastFlood.
VALID_RESOLUTIONS = {"1m", "2m", "5m", "10m", "20m", "40m", "80m", "150m", "300m"}

# Optional preconfigured DEM. Leave as None to be prompted for a DEM path each
# run. If set to a valid file path here, the DEM prompt can be left blank to use it.
DEFAULT_DEM_PATH = None

# =============================================================================
# RUNTIME SETTINGS — populated from prompts in main(); normally do not edit.
# =============================================================================
# DEM_PATH: the DEM actually used by this run (required for tiling/mosaicking).
# OUTPUT_ROOT: unique, parameter-named directory created under the output folder.
DEM_PATH = None
OUTPUT_ROOT = None
SCENARIO = "flashfloodfluvialflood"  # FastFlood scenario identifier; verify for your installation.
RESOLUTION = DEFAULT_RESOLUTION
DURATION_HOURS = 6
MULT_INF = DEFAULT_MULT_INF
MULT_MAN = DEFAULT_MULT_MAN
CHANNELMV = None
BBOX = None
KEEP_TILES = False
API_KEY = "<KEY>"
DEPTH_TILE_NAME = "flooddepth.tif"
DISCHARGE_TILE_NAME = "discharge.tif"
LOG_LOCK = threading.Lock()

@dataclass(frozen=True)
class Tile:
    grid: str
    tile_id: str
    col: int
    row: int
    width: int
    height: int
    west: float
    north: float
    east: float
    south: float

    def key(self):
        return self.grid, self.tile_id


def timestamp():
    """Return a local, timezone-aware timestamp for logs and reports."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message):
    """Print a timestamped message and append it to the run log."""
    line = f"[{timestamp()}] {message}"
    with LOG_LOCK:
        print(line, flush=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with (OUTPUT_ROOT / "run.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def tile_dir(tile):
    """Return the working directory for one tile in one of the two grids."""
    return OUTPUT_ROOT / "tiles" / f"grid_{tile.grid}" / tile.tile_id


def output_paths(tile):
    """Return the expected depth and discharge output paths for a tile."""
    d = tile_dir(tile)
    return d / DEPTH_TILE_NAME, d / DISCHARGE_TILE_NAME


def valid_raster(path):
    """Return True if path looks like a readable, non-empty georeferenced raster."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size < MIN_VALID_FILE_BYTES:
        return False
    try:
        with rasterio.open(path) as src:
            if src.width < 1 or src.height < 1 or src.count < 1 or src.crs is None:
                return False
            # Read a small sample to catch truncated/corrupt rasters.
            src.read(1, window=Window(0, 0, min(16, src.width), min(16, src.height)))
        return True
    except Exception:
        return False


def make_axis_starts(length, tile_length, shift=0):
    # Cover the axis with regular overlapping windows and include both edges.
    stride = max(1, int(round(tile_length * (1.0 - TILE_OVERLAP_PERCENT))))
    starts = []
    x = max(0, int(round(shift)))
    # Keep only full-size regular windows. Add an explicit final window
    # aligned to the far edge instead of creating a narrow sliver beyond it.
    while x + tile_length <= length:
        starts.append(x)
        x += stride
    starts.extend([0, max(0, length - tile_length)])
    return sorted(set(starts))


def build_tiles(src):
    """Build two staggered overlapping tile grids within the requested bbox.

    The requested bbox is converted to pixel coordinates in the DEM. The area is
    then covered by Grid A and a half-stride-shifted Grid B. Tiles are clipped to
    the selected area and DEM edges; the input DEM is not resampled or reprojected.
    """
    if not math.isclose(src.transform.b, 0, abs_tol=1e-10) or not math.isclose(src.transform.d, 0, abs_tol=1e-10):
        raise ValueError("Rotated DEM grids are not supported; rectify the DEM first.")
    from rasterio.windows import from_bounds
    west, south, east, north = BBOX
    raw = from_bounds(west, south, east, north, src.transform)
    col0, row0 = max(0, int(math.floor(raw.col_off))), max(0, int(math.floor(raw.row_off)))
    col1 = min(src.width, int(math.ceil(raw.col_off + raw.width)))
    row1 = min(src.height, int(math.ceil(raw.row_off + raw.height)))
    if col1 <= col0 or row1 <= row0:
        raise ValueError("Bounding box does not overlap the DEM.")
    px, py = abs(src.transform.a), abs(src.transform.e)
    area_width, area_height = col1-col0, row1-row0
    tw, th = max(1, int(round(TILE_SIZE_KM*1000/px))), max(1, int(round(TILE_SIZE_KM*1000/py)))
    stride_x, stride_y = max(1, int(round(tw*(1-TILE_OVERLAP_PERCENT)))), max(1, int(round(th*(1-TILE_OVERLAP_PERCENT))))
    grids = {
        "A": (make_axis_starts(area_width, tw), make_axis_starts(area_height, th)),
        "B": (make_axis_starts(area_width, tw, stride_x*GRID_B_OFFSET_SPACING_FRACTION),
              make_axis_starts(area_height, th, stride_y*GRID_B_OFFSET_SPACING_FRACTION)),
    }
    tiles = []
    for grid, (xs, ys) in grids.items():
        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                x0, y0 = col0+x, row0+y
                x1, y1 = min(col1, col0+x+tw), min(row1, row0+y+th)
                if x1 <= x0 or y1 <= y0:
                    continue
                win = Window(x0, y0, x1-x0, y1-y0)
                west_t, south_t, east_t, north_t = rasterio.windows.bounds(win, src.transform)
                tiles.append(Tile(grid, f"tile_{r:03d}_{c:03d}", x0, y0, int(x1-x0), int(y1-y0),
                                  west_t, north_t, east_t, south_t))
    return tiles



def clip_dem(tile, target):
    """Write the DEM window for a tile to a compressed Float32 GeoTIFF."""
    win = Window(tile.col, tile.row, tile.width, tile.height)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.tif")
    with rasterio.open(DEM_PATH) as src:
        profile = {
            "driver": "GTiff", "width": tile.width, "height": tile.height,
            "count": 1, "dtype": "float32", "crs": src.crs,
            "transform": src.window_transform(win), "nodata": OUTPUT_NODATA,
            "tiled": True, "blockxsize": 256, "blockysize": 256,
            "compress": "DEFLATE", "predictor": 3, "BIGTIFF": "IF_SAFER",
        }
        with rasterio.open(tmp, "w", **profile) as dst:
            # masked=True honors the source nodata/mask regardless of sentinel.
            data = src.read(1, window=win, masked=True).astype("float32")
            arr = np.asarray(data.filled(OUTPUT_NODATA), dtype=np.float32)
            invalid = np.ma.getmaskarray(data) | ~np.isfinite(arr)
            # Extremely large float32 sentinels are not plausible elevations.
            invalid |= np.abs(arr) > 1.0e20
            arr[invalid] = OUTPUT_NODATA
            dst.write(arr, 1)
        os.replace(tmp, target)


def redact_command(command):
    """Render a command for diagnostics while hiding the API key."""
    safe = list(command)
    if "-key" in safe:
        i = safe.index("-key")
        if i + 1 < len(safe):
            safe[i + 1] = "<REDACTED>"
    return subprocess.list2cmdline(safe)

#"-channelmv","100",
#"-maxlrres 2500"
#"-channelmv","500"
def build_command(tile, dem_file, depth_file, discharge_file, api_key):
    """Build one FastFlood invocation; CLI-specific flags live here.

    Keep this function as the single place to review CLI syntax. In particular,
    SCENARIO, RETURN_PERIOD, RESOLUTION, duration, calibration values, CHANNEL_ARGS,
    and optional CHANNELMV are translated into FastFlood command-line arguments.
    """
    command = [
        str(FASTFLOOD_EXE), "-key", api_key, "-autohazard",
        str(tile.west), str(tile.north), str(tile.east), str(tile.south),
        SCENARIO, str(RETURN_PERIOD), RESOLUTION, "-sim", *CHANNEL_ARGS,
        "-designdatadur", str(DURATION_HOURS), "-relmindabcond", str(RELMINDABCOND),
        "-dem", str(dem_file.resolve()), "-d_lu", "-d_inf",
        "-whout", str(depth_file.resolve()), "-qout", str(discharge_file.resolve()),
        "-dur", str(DURATION_HOURS), "-mult_inf", str(MULT_INF), "-mult_man", str(MULT_MAN),
        "-noautorefine", "-nochanwh",
    ]
    if CHANNELMV is not None:
        command.extend(["-channelmv", str(CHANNELMV)])
    return command


def run_tile(tile, api_key):
    """Prepare one tile, run FastFlood, save diagnostics, and report its status."""
    started = time.monotonic()
    directory = tile_dir(tile)
    directory.mkdir(parents=True, exist_ok=True)
    depth, discharge = output_paths(tile)
    result = {"grid": tile.grid, "tile": tile.tile_id}
    if SKIP_VALID_EXISTING_TILES and valid_raster(depth) and valid_raster(discharge):
        return {**result, "status": "skipped", "message": "Existing readable outputs", "elapsed_seconds": 0}
    dem_file = directory / "dem_tile.tif"
    try:
        clip_dem(tile, dem_file)
        command = build_command(tile, dem_file, depth, discharge, api_key)
        (directory / "command.txt").write_text(redact_command(command), encoding="utf-8")
        completed = subprocess.run(command, cwd=directory, capture_output=True, text=True, errors="replace")
        (directory / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
        (directory / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode:
            msg = f"FastFlood exit code {completed.returncode}; see stderr.txt"
            status = "failed"
        elif not (valid_raster(depth) and valid_raster(discharge)):
            msg, status = "FastFlood returned success but one or more output rasters are invalid", "failed"
        else:
            msg, status = "OK", "completed"
    except Exception as exc:
        msg, status = f"{exc}\n{traceback.format_exc()}", "failed"
        (directory / "stderr.txt").write_text(msg, encoding="utf-8")
    return {**result, "status": status, "message": msg, "elapsed_seconds": round(time.monotonic()-started, 2)}


def write_manifest(tiles, results):
    """Write tile coordinates and final status to tile_manifest.csv."""
    by_key = {(r["grid"], r["tile"]): r for r in results}
    path = OUTPUT_ROOT / "tile_manifest.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["grid", "tile_id", "col", "row", "width", "height", "west", "north", "east", "south", "status", "message"])
        for t in tiles:
            r = by_key.get(t.key(), {})
            w.writerow([t.grid, t.tile_id, t.col, t.row, t.width, t.height, t.west, t.north, t.east, t.south, r.get("status", "not_run"), r.get("message", "")])


def append_timings(results):
    """Append per-tile duration/status records to simulation_timings.csv."""
    path = OUTPUT_ROOT / "simulation_timings.csv"
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "grid", "tile_id", "status", "elapsed_seconds", "message"])
        for r in results:
            w.writerow([timestamp(), r["grid"], r["tile"], r["status"], r.get("elapsed_seconds", 0), str(r.get("message", "")).replace("\n", " ")[:500]])


def make_weights(h, w):
    """Build smooth edge weights used to feather overlapping tile boundaries."""
    fy = max(1.0, h * BLEND_EDGE_FRACTION)
    fx = max(1.0, w * BLEND_EDGE_FRACTION)
    y = np.minimum(np.arange(h)+0.5, np.arange(h, 0, -1)-0.5)
    x = np.minimum(np.arange(w)+0.5, np.arange(w, 0, -1)-0.5)
    wy, wx = np.clip(y/fy, 0, 1), np.clip(x/fx, 0, 1)
    wy, wx = wy*wy*(3-2*wy), wx*wx*(3-2*wx)
    return np.maximum(wy[:, None]*wx[None, :], MIN_BLEND_WEIGHT).astype("float32")


def mosaic_grid(tiles, grid, output):
    """Blend valid depth tiles belonging to one grid into a DEM-aligned mosaic."""
    paths = [(t, output_paths(t)[0]) for t in tiles if t.grid == grid and valid_raster(output_paths(t)[0])]
    if not paths:
        log(f"No valid depth rasters for grid {grid}")
        return False
    with rasterio.open(DEM_PATH) as dem:
        crs, transform = dem.crs, dem.transform
        width, height = dem.width, dem.height
    profile = {"driver":"GTiff", "width":width, "height":height, "count":1, "dtype":"float32", "crs":crs, "transform":transform, "nodata":OUTPUT_NODATA, "tiled":True, "blockxsize":MOSAIC_BLOCK_SIZE, "blockysize":MOSAIC_BLOCK_SIZE, "compress":"DEFLATE", "predictor":3, "BIGTIFF":"IF_SAFER"}
    output.parent.mkdir(parents=True, exist_ok=True)
    sum_path, weight_path = output.with_suffix(".sum.tmp.tif"), output.with_suffix(".weight.tmp.tif")
    for p in (output, sum_path, weight_path):
        if p.exists(): p.unlink()
    acc_profile = dict(profile); acc_profile["nodata"] = None
    # Open accumulators in read/write mode: the blending loop reads previous
    # values back from these files for each overlapping tile.
    with rasterio.open(sum_path, "w+", **acc_profile) as sums, rasterio.open(weight_path, "w+", **acc_profile) as weights:
        # Initialize valid zero-valued accumulators in bounded windows.
        for _, win in sums.block_windows(1):
            z = np.zeros((int(win.height), int(win.width)), dtype="float32")
            sums.write(z, 1, window=win); weights.write(z, 1, window=win)
        for tile, path in paths:
            win = Window(tile.col, tile.row, tile.width, tile.height)
            with rasterio.open(path) as src:
                data = src.read(1, masked=True).astype("float32")
                arr = np.asarray(data.filled(0), dtype="float32")
                good = ~np.ma.getmaskarray(data) & np.isfinite(arr) & (arr != OUTPUT_NODATA)
                wt = make_weights(tile.height, tile.width)
                wt[~good] = 0
                old_s = sums.read(1, window=win); old_w = weights.read(1, window=win)
                old_s += arr * wt; old_w += wt
                sums.write(old_s, 1, window=win); weights.write(old_w, 1, window=win)
    with rasterio.open(sum_path) as sums, rasterio.open(weight_path) as weights, rasterio.open(output, "w", **profile) as dst:
        for _, win in dst.block_windows(1):
            total, wt = sums.read(1, window=win), weights.read(1, window=win)
            out = np.full(total.shape, OUTPUT_NODATA, dtype="float32")
            good = wt > 0
            out[good] = total[good] / wt[good]
            dst.write(out, 1, window=win)
    sum_path.unlink(missing_ok=True); weight_path.unlink(missing_ok=True)
    return True


def combine_max(grid_a, grid_b, output):
    """Combine two blended grids using the maximum where both pixels are valid."""
    has_a = valid_raster(grid_a)
    has_b = valid_raster(grid_b)
    if not has_a and not has_b:
        log("No valid grid mosaics available; combined mosaic was not created.")
        return False

    with rasterio.open(DEM_PATH) as dem:
        profile = {
            "driver": "GTiff", "width": dem.width, "height": dem.height,
            "count": 1, "dtype": "float32", "crs": dem.crs,
            "transform": dem.transform, "nodata": OUTPUT_NODATA,
            "tiled": True, "blockxsize": MOSAIC_BLOCK_SIZE,
            "blockysize": MOSAIC_BLOCK_SIZE, "compress": "DEFLATE",
            "predictor": 3, "BIGTIFF": "IF_SAFER",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    from contextlib import ExitStack
    with ExitStack() as stack:
        a = stack.enter_context(rasterio.open(grid_a)) if has_a else _NullRaster()
        b = stack.enter_context(rasterio.open(grid_b)) if has_b else _NullRaster()
        dst = stack.enter_context(rasterio.open(output, "w", **profile))
        for _, win in dst.block_windows(1):
            aa = a.read(1, window=win, masked=True).astype("float32") if has_a else None
            bb = b.read(1, window=win, masked=True).astype("float32") if has_b else None
            va = np.zeros((int(win.height), int(win.width)), bool) if aa is None else (~np.ma.getmaskarray(aa) & np.isfinite(aa.data) & (aa.data != OUTPUT_NODATA))
            vb = np.zeros_like(va) if bb is None else (~np.ma.getmaskarray(bb) & np.isfinite(bb.data) & (bb.data != OUTPUT_NODATA))
            out = np.full(va.shape, OUTPUT_NODATA, dtype="float32")
            if aa is not None: out[va] = aa.data[va]
            if bb is not None:
                both = va & vb; only_b = ~va & vb
                out[both] = np.maximum(aa.data[both], bb.data[both]); out[only_b] = bb.data[only_b]
            dst.write(out, 1, window=win)
    return True


class _NullRaster:
    def __enter__(self): return self
    def __exit__(self, *args): return False


def ask_choice(prompt, choices, default):
    """Prompt until a valid choice is entered; Enter selects the default."""
    lookup = {str(c).lower(): c for c in choices}
    while True:
        value = input(f"{prompt} ({'/'.join(map(str, choices))}; default {default}): ").strip().lower()
        if not value:
            return default
        if value in lookup:
            return lookup[value]
        print("Choose one of: " + ", ".join(map(str, choices)))


def ask_required_float(prompt):
    """Prompt for a numeric input that must be explicitly supplied by the user.

    Unlike ask_float(), this helper has no default: pressing Enter without a
    value repeats the prompt. It also rejects NaN and infinity.
    """
    while True:
        raw = input(f"{prompt}: ").strip()
        if not raw:
            print("This value is required; please enter a number.")
            continue
        try:
            value = float(raw)
            if math.isfinite(value):
                return value
        except ValueError:
            pass
        print("Enter a valid finite number.")


def ask_float(prompt, default):
    """Prompt for a finite numeric value; pressing Enter accepts the default."""
    while True:
        raw = input(f"{prompt} (default {default}): ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
            if math.isfinite(value):
                return value
        except ValueError:
            pass
        print("Enter a valid number.")


def main():
    """Collect run settings, execute tile jobs, blend each grid, then max-combine.

    Exit codes: 0 = success; 1 = one or more tile failures after processing;
    2 = tile failure prevented mosaic creation under the current safety settings.
    """
    global DEM_PATH, OUTPUT_ROOT, SCENARIO, RESOLUTION, DURATION_HOURS
    global MULT_INF, MULT_MAN, CHANNELMV, BBOX, KEEP_TILES, API_KEY
    global DEPTH_TILE_NAME, DISCHARGE_TILE_NAME

    # -------------------------------------------------------------------------
    # REQUIRED RUN INPUTS: mode, bbox, resolution, output folder, and DEM source.
    # OPTIONAL/DEFAULTED INPUTS: duration override, calibration, tile retention,
    # and API key. RUN_MODE at the top sets the default offered by the prompt.
    # -------------------------------------------------------------------------
    print("\nFastFlood — simple flood simulation setup")
    print("\nREQUIRED INPUTS")
    print(f"Bounding-box coordinates must use the DEM CRS ({EXPECTED_DEM_CRS}).")

    # Mode aliases encode the common scenario/duration combinations. Generic
    # FLASH, HYBRID, and FLUVIAL use their standard defaults; explicit aliases
    # such as FLASH6H set a specific duration unless a valid override is supplied.
    mode_aliases = {
        "FLASH": ("flash", 3),        
	"FLASH3H": ("flash", 1),
        "FLASH3H": ("flash", 3),
        "FLASH6H": ("flash", 6),
        "HYBRID": ("hybrid", 6),
        "FLUVIAL": ("fluvial", 120),
        "FLUVIAL72H": ("fluvial", 72),
        "FLUVIAL120H": ("fluvial", 120),
    }
    default_mode = RUN_MODE.strip().upper()
    if default_mode not in mode_aliases:
        raise ValueError(f"RUN_MODE must be one of: {', '.join(mode_aliases)}")
    chosen = ask_choice(
        "Scenario/mode",
        list(mode_aliases),
        default_mode,
    )
    mode, implied_duration = mode_aliases[chosen]

    # An explicit configuration override wins over the alias's implied duration.
    # If no override is configured, the user can accept the inferred duration or
    # choose another valid duration for the selected mode at the prompt.
    allowed_durations = [72, 120] if mode == "fluvial" else [1, 3, 6]
    duration_default = DURATION_HOURS_OVERRIDE if DURATION_HOURS_OVERRIDE is not None else implied_duration
    if duration_default not in allowed_durations:
        raise ValueError(f"Duration override {duration_default!r} is invalid for {chosen}; choose {allowed_durations}.")
    DURATION_HOURS = int(ask_choice("Duration in hours", allowed_durations, duration_default))

    # channelmv is derived from the effective mode and duration. It is only used
    # for Flash Only; Hybrid and Fluvial deliberately omit the CLI option.
    CHANNELMV = ({1: 100, 3: 300, 6: 900}[DURATION_HOURS] if mode == "flash" else None)

    # A named preset is translated through RESOLUTION_PRESETS. Literal values are
    # passed to FastFlood as typed. The default is 20m if the user presses Enter.
    while True:
        raw = input("Resolution (low/medium/high/veryhigh or 1m, 2m, 5m, 10m, 20m, 40m, 80m, 150m, 300m) [20m]: ").strip().lower() or DEFAULT_RESOLUTION
        if raw in RESOLUTION_PRESETS:
            RESOLUTION = RESOLUTION_PRESETS[raw]
            break
        if raw in VALID_RESOLUTIONS:
            RESOLUTION = raw
            break
        print("Unsupported resolution. Please choose a listed preset or resolution.")

    print("Enter bounding box in DEM CRS.")
    # Coordinates are mandatory and intentionally have no defaults, so pressing
    # Enter cannot accidentally run a simulation over a zero-sized/wrong area.
    # The script also checks east > west and north > south below.
    west = ask_required_float("West / minimum X")
    south = ask_required_float("South / minimum Y")
    east = ask_required_float("East / maximum X")
    north = ask_required_float("North / maximum Y")
    if east <= west or north <= south:
        raise ValueError("Bounding box must have east > west and north > south.")
    BBOX = (west, south, east, north)

    print("\nOPTIONAL INPUTS (press Enter to accept each default)")
    # Calibration multipliers modify the corresponding FastFlood model inputs.
    # A value of 1.0 leaves the original model parameter unscaled.
    MULT_MAN = ask_float("MULT_MAN — Manning/roughness multiplier", DEFAULT_MULT_MAN)
    MULT_INF = ask_float("MULT_INF — infiltration multiplier", DEFAULT_MULT_INF)
    out_raw = input("Output folder: ").strip()
    if not out_raw:
        raise ValueError("Output folder is required.")
    output_base = Path(out_raw).expanduser()
    # The DEM path is optional only when DEFAULT_DEM_PATH was set above. The
    # workflow itself requires a DEM because it must clip tiles and align mosaics.
    configured_dem = Path(DEFAULT_DEM_PATH).expanduser() if DEFAULT_DEM_PATH else None
    dem_hint = f" [{configured_dem}]" if configured_dem else ""
    dem_raw = input(f"DEM file path{dem_hint} (Enter uses configured DEM if available): ").strip()
    if dem_raw:
        DEM_PATH = Path(dem_raw).expanduser()
    elif configured_dem:
        DEM_PATH = configured_dem
    else:
        raise ValueError("No DEM supplied and DEFAULT_DEM_PATH is not configured. A DEM is required for this workflow.")
    # Keeping tiles preserves each clipped DEM, tile outputs, command record, and
    # stdout/stderr logs. If No, the tiles directory is deleted after a successful
    # run; final mosaics, manifest, timings, and run.log remain in the run folder.
    KEEP_TILES = ask_choice("Keep tiles and all associated per-tile files?", ["yes", "no"], DEFAULT_KEEP_TILES) == "yes"
    # API keys are redacted from command.txt. Do not share logs that may contain
    # other sensitive information. <KEY> is a placeholder, not a real credential.
    API_KEY = input(f"API key [{DEFAULT_API_KEY}]: ").strip() or DEFAULT_API_KEY

    mode_tag = {"flash": "flash_only", "hybrid": "hybrid", "fluvial": "fluvial"}[mode]
    bbox_tag = f"x{west:g}-{east:g}_y{south:g}-{north:g}".replace(".", "p").replace("-", "m")
    run_name = f"{mode_tag}_{DURATION_HOURS}h_{RESOLUTION}_man{MULT_MAN:g}_inf{MULT_INF:g}_{bbox_tag}"
    OUTPUT_ROOT = output_base / run_name
    DEPTH_TILE_NAME = f"depth_{mode_tag}_{DURATION_HOURS}h_{RESOLUTION}.tif"
    DISCHARGE_TILE_NAME = f"discharge_{mode_tag}_{DURATION_HOURS}h_{RESOLUTION}.tif"

    if not FASTFLOOD_EXE.is_file():
        raise FileNotFoundError(f"FastFlood executable not found: {FASTFLOOD_EXE}. Edit FASTFLOOD_EXE in advanced settings.")
    if not DEM_PATH.is_file():
        raise FileNotFoundError(f"DEM not found: {DEM_PATH}")
    if API_KEY == "<KEY>":
        print("WARNING: API key is the <KEY> placeholder. Enter a valid key if required.")
    if PROCESSES < 1:
        raise ValueError("PROCESSES must be at least 1.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    with rasterio.open(DEM_PATH) as src:
        if src.count < 1:
            raise ValueError("DEM has no bands.")
        if src.crs != rasterio.crs.CRS.from_string(EXPECTED_DEM_CRS):
            raise ValueError(f"Expected {EXPECTED_DEM_CRS}, found {src.crs}. Adjust EXPECTED_DEM_CRS if appropriate.")
        tiles = build_tiles(src)
        log(f"Mode={mode}; duration={DURATION_HOURS}h; resolution={RESOLUTION}; channelmv={CHANNELMV}; generated {len(tiles)} tiles.")

    results = []
    with ThreadPoolExecutor(max_workers=PROCESSES) as pool:
        futures = {pool.submit(run_tile, t, API_KEY): t for t in tiles}
        for future in as_completed(futures):
            t = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"grid": t.grid, "tile": t.tile_id, "status": "failed",
                          "message": str(exc), "elapsed_seconds": 0}
            results.append(result)
            log(f"{len(results)}/{len(tiles)} {t.grid}/{t.tile_id}: {result['status']} ({result.get('elapsed_seconds', 0)}s)")
    write_manifest(tiles, results)
    append_timings(results)
    failed = [r for r in results if r["status"] == "failed"]
    if failed and not ALLOW_INCOMPLETE_MOSAIC:
        log(f"{len(failed)} tile(s) failed; mosaics not created. See manifest and per-tile logs.")
        return 2

    if CREATE_MOSAICS:
        a = OUTPUT_ROOT / f"grid_A_depth_blended_{mode_tag}_{DURATION_HOURS}h_{RESOLUTION}.tif"
        b = OUTPUT_ROOT / f"grid_B_depth_blended_{mode_tag}_{DURATION_HOURS}h_{RESOLUTION}.tif"
        final = OUTPUT_ROOT / f"depth_dualgrid_max_{mode_tag}_{DURATION_HOURS}h_{RESOLUTION}.tif"
        mosaic_grid(tiles, "A", a)
        mosaic_grid(tiles, "B", b)
        combine_max(a, b, final)

    if not KEEP_TILES and not failed:
        import shutil
        shutil.rmtree(OUTPUT_ROOT / "tiles", ignore_errors=True)
    log(f"Finished: {len(tiles)-len(failed)} successful/skipped; {len(failed)} failed.")
    log(f"Output folder: {OUTPUT_ROOT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

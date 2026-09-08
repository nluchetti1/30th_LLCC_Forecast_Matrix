"""
ivt_maps.py — CW3E-style spatial Integrated Vapor Transport maps.

Companion to ivt.py. That module does the column integral for the point/matrix
values; this one does the 2D field: fetch, integrate on the grid, render, and hand
back a {model: {fh_key: png_path}} dict for history.json.

WHAT THE PROBES ESTABLISHED (2026-08-31, verified live -- do not re-derive):

  * NOMADS GRIB filter with `subregion` beats S3 byte-range by ~32x for this job.
    GFS SPFH+U+V over the wide domain is 1.5 MB/step against 48.4 MB/step pulling
    whole global messages off S3. Server-side subsetting wins because IVT needs many
    levels over a small footprint.

  * GFS serves SPFH on all 17 integration levels. Use it directly.

  * RAP and NAM DO NOT serve SPFH through the filter. They return HTTP 200 with
    valid GRIB containing u and v and NO moisture at all -- which would integrate to
    IVT ~= 0 and read as a bone-dry airmass. Both carry RH+TMP on all 17 levels, so
    they go through the RH route. This is why _grid_moisture() refuses to guess:
    a missing moisture field must raise, never silently yield zeros.

  * HRRR is NOT reachable here. filter_hrrr_sub.pl is the SUB-HOURLY filter and
    rejects wrfprs filenames outright ("Filename does not match legal pattern:
    hrrr.t..z.wrfsubhf...grib2$"). HRRR would have to come from HRRR_AWS_ROOT with
    no spatial subsetting. Since IVT is synoptic and HRRR tops out at 48 h, it is
    deliberately excluded rather than paid for.

  * ECMWF open data has q but NO subsetting -- ~16 MB/step global vs GFS's 1.5 MB.
    Left out of the default set; ECMWF_ENABLED below turns it on if wanted.
"""

import datetime
import logging
import math
import os
import tempfile
import time

import numpy as np
import pygrib
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ivt import G, EPSILON, IVT_TOP_HPA, cw3e_ivt_cmap, CW3E_IVT_BOUNDS

IVT_MAPS_ENABLED = True
IVT_MAPS_DIR = "./maps/ivt"

# Wide enough that IVT reads as the synoptic transport field it is. The pipeline's
# CA_DOMAIN (32-37.5N) is far too tight -- at that zoom an atmospheric river fills the frame
# uniformly and the gradient, which is the whole signal, is invisible.
#
# WHY THIS FAR WEST. The eastern-Pacific version of this map has a different job from the
# Atlantic one it replaces. A landfalling AR is a 2000+ km filament whose useful information
# is WHERE ALONG THE COAST the axis intersects and how the plume is oriented upstream, and
# neither is visible if the frame starts at the coastline -- by the time an AR is inside a
# CONUS-width box the forecast is already a nowcast. The left edge at -155 sits just west of
# Hawaii, which is deliberate: it is far enough to hold the subtropical moisture tap that
# feeds a Pineapple Express in the same frame as the landfall, so the connection is visible
# rather than inferred. The top at 57N holds the Gulf of Alaska, which is where the other
# common flavour (a cold, kinematically-driven AR on a deep trough) comes from.
IVT_DOMAIN = {"left": -155.0, "right": -108.0, "top": 57.0, "bottom": 20.0}

# CONUS-model domain. RAP and NAM are Lambert grids that simply do not extend to -155, so
# rendering them on IVT_DOMAIN produces a panel that is ~60% empty basemap with a small
# patch of data in the corner -- technically correct and operationally useless. They get a
# box sized to what they actually cover: the coastal waters, the whole of California, and
# far enough east that an inland-penetrating plume does not run off the edge.
#
# The scales are different on purpose, so read the frame before comparing two panels. A
# GFS map and a RAP map of the same hour are NOT the same zoom, and an AR that looks
# narrow on one will look broad on the other.
IVT_DOMAIN_CONUS = {"left": -132.0, "right": -110.0, "top": 50.0, "bottom": 28.0}


def model_domain(model):
    """Map extent for one model. Per-model override, falling back to the wide Pacific box."""
    return IVT_MODELS.get(model, {}).get("domain") or IVT_DOMAIN

IVT_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 750, 700,
              650, 600, 550, 500, 450, 400, 350, 300]

# Per-model fetch recipe. `moisture` is "spfh" or "rh" and is NOT negotiable per the
# probe results above.
# ECMWF open data publishes only these 8 levels at or below 300 hPa (probed 2026-08-31,
# not assumed). Integrating 8 instead of 17 biases IVT +1.3% high on a smooth column --
# trapezoid over a convex q profile -- which is negligible against model error. A sharp dry
# intrusion inside the 150 hPa gap between 850 and 700 would do worse.
ECMWF_IVT_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300]

IVT_MODELS = {
    "gfs": {
        "enabled": True,
        "source": "nomads",
        "filter": "filter_gfs_0p25.pl",
        "file": lambda c, f: f"gfs.t{c}z.pgrb2.0p25.f{f:03d}",
        "dir": lambda d, c: f"%2Fgfs.{d}%2F{c}%2Fatmos",
        "moisture": "spfh",
        "mslp": "PRMSL",
        "cycles": [0, 6, 12, 18],
        "latency_h": 4,
        # Tapered: 3-hourly through f072 where the field actually evolves, 6-hourly beyond.
        # GFS publishes 1-hourly to f120 so 3-hourly everywhere is available -- but that is 61
        # NOMADS requests instead of 31, and GFS was being refused by NOMADS two runs ago.
        # Tapering buys the short-range detail without doubling throttle exposure.
        "steps": list(range(0, 73, 3)) + list(range(78, 181, 6)),
    },
    "rap": {
        "enabled": True,
        "source": "nomads",
        "filter": "filter_rap.pl",
        "file": lambda c, f: f"rap.t{c}z.awp130pgrbf{f:02d}.grib2",
        "dir": lambda d, c: f"%2Frap.{d}",
        # RAP publishes MSLMA (MAPS MSL pressure), NOT PRMSL. Asking the filter for a
        # variable this file does not contain returns HTTP 500 -- an entire cycle of steps
        # failed on exactly this, while determine_cycle (which requests no MSLP) succeeded.
        "moisture": "rh",                  # SPFH silently absent -- see header
        "mslp": "MSLMA",
        "cycles": list(range(24)),
        "latency_h": 2,
        # Lambert CONUS grid; nowhere near the Pacific box. See IVT_DOMAIN_CONUS.
        "domain": IVT_DOMAIN_CONUS,
        # RAP is an HOURLY model and only runs to f021, so sampling every third hour threw
        # away most of what it offers for 14 extra cheap requests.
        "steps": list(range(0, 22, 1)),
    },
    "nam": {
        "enabled": True,
        "source": "nomads",
        "filter": "filter_nam.pl",
        "file": lambda c, f: f"nam.t{c}z.awphys{f:02d}.tm00.grib2",
        "dir": lambda d, c: f"%2Fnam.{d}",
        "moisture": "rh",                  # SPFH silently absent -- see header
        "mslp": "PRMSL",
        "cycles": [0, 6, 12, 18],
        "latency_h": 3,
        # Lambert CONUS grid; nowhere near the Pacific box. See IVT_DOMAIN_CONUS.
        "domain": IVT_DOMAIN_CONUS,
        # NAM carries 3-hourly all the way to f084, so the full range is available.
        "steps": list(range(0, 85, 3)),
    },
    # ECMWF is NOT a NOMADS model and behaves differently in three ways that matter:
    #   * no spatial subsetting -- every step arrives as a full global 721x1440 grid, so it
    #     costs 16.8 MB/step against GFS's 1.5 MB. Hence the coarser default step list.
    #   * two retrieves per step (pressure levels and surface cannot come in one call).
    #   * cycle discovery via the client's own index, not a filter probe.
    # 6-hourly to f144 is 25 steps, ~420 MB, ~2.5 min of download. Widen or thin this list
    # freely -- it is the single knob that trades coverage against runtime.
    "ecmwf": {
        "enabled": True,
        "source": "ecmwf",
        "moisture": "spfh",            # shortName is 'q', which _grid_moisture reads directly
        "mslp": "msl",
        "levels": ECMWF_IVT_LEVELS,
        # Only 00z and 12z run the full forecast length; 06z and 18z stop near f90, which
        # would silently truncate the map set. determine_cycle snaps back to 00/12.
        "cycles": [0, 12],
        "latency_h": 8,
        # Left at 6-hourly ON PURPOSE. ECMWF has 3-hourly to f144, but at 16.8 MB/step that
        # is 49 steps and ~820 MB per run -- roughly double everything else combined, on a
        # pipeline already running 43 minutes against an hourly cron. For 3-hourly ECMWF over
        # a shorter range, use:
        #     list(range(0, 73, 3)) + list(range(78, 145, 6))
        "steps": list(range(0, 145, 6)),
    },
}

# NOMADS politeness. Matches the main pipeline's posture: one connection, a real
# pause between requests. A 31-step GFS set is ~31 * (1.5 + 4) = ~170 s.
IVT_REQUEST_PAUSE_S = 4.0
IVT_CONNECT_TIMEOUT = 15
IVT_READ_TIMEOUT = 90
IVT_ATTEMPTS = 3
IVT_BUDGET_S = 1500         # whole-job budget; returns what it has when spent
IVT_MODEL_BUDGET_S = 480    # per-model cap, so a slow ECMWF cannot starve GFS
IVT_THROTTLE_BACKOFF_S = 20 # 403/429/503 means NOMADS is declining; retrying in 4s
                            # just burns the budget being refused more quickly
IVT_CYCLE_ATTEMPTS = 2      # probe tries per candidate cycle before walking back
# If every cycle probe fails with a TRANSIENT error (throttle, 5xx, connection), assume
# the newest nominal cycle is fine and let the real fetch decide. NOMADS refusing a probe
# at one moment does not mean the data is absent -- on 2026-08-31 16:08 all three models
# reported "no usable cycle found" in ~5s flat, which is a refusal, not a missing file.
IVT_CYCLE_FALLBACK = True
# Give up on a model after this many consecutive step failures. Without it, a fallback to
# a cycle that really is absent would grind through every step x every retry.
IVT_MAX_CONSECUTIVE_FAILURES = 3
IVT_UA = ("Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
          "Gecko/20100101 Firefox/120.0")

# Rendering
# Transport-vector density, expressed as "arrows across the panel" rather than a grid-cell
# stride, because the panels no longer share a domain or a grid spacing (see render_ivt_map).
# 28 reproduces the density the Cape build got from stride 6 on a 0.25 deg, 40-degree box.
IVT_VECTOR_TARGET_COLS = 28
IVT_MIN_SHADE = 250.0       # below this the field is transparent (CW3E convention)
MSLP_INTERVAL_HPA = 4


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def _build_url(model, date_str, cycle, fh, with_mslp=True):
    cfg = IVT_MODELS[model]
    lev = "".join(f"&lev_{lv}_mb=on" for lv in IVT_LEVELS)
    if cfg["moisture"] == "spfh":
        var = "&var_SPFH=on"
    else:
        var = "&var_RH=on&var_TMP=on"
    var += "&var_UGRD=on&var_VGRD=on&var_PRES=on&lev_surface=on"
    if with_mslp:
        # The MSLP variable NAME differs by model (see "mslp" above) and lives on mean sea
        # level, so it needs its own level flag. Requesting the wrong name is fatal to the
        # whole request, not merely to the contours -- hence the with_mslp=False retry.
        var += f"&var_{cfg['mslp']}=on&lev_mean_sea_level=on"
    # Ask NOMADS only for the box this model is actually plotted on. GFS on the wide Pacific
    # box is ~2.3x the old Atlantic request per step; RAP and NAM stay small because they use
    # the CONUS box. Requesting IVT_DOMAIN for the Lambert models would not fail, it would
    # just return their whole grid and then waste the bandwidth.
    dom = model_domain(model)
    region = (f"&subregion=&leftlon={dom['left']}&rightlon={dom['right']}"
              f"&toplat={dom['top']}&bottomlat={dom['bottom']}")
    return (f"https://nomads.ncep.noaa.gov/cgi-bin/{cfg['filter']}"
            f"?file={cfg['file'](cycle, fh)}{lev}{var}{region}"
            f"&dir={cfg['dir'](date_str, cycle)}")


# Last failure reason per model, so the abandon message can say what actually happened
# rather than only that it happened. The run.log warnings carry the detail, but the
# workflow's post-run grep only surfaces a few lines -- this puts the cause in one of them.
_LAST_FAIL = {}


def _fetch_grib(session, url, tag=""):
    """GET a filter URL, verify it is really GRIB, write to a temp file.

    The filter answers HTTP 200 with an HTML error page on a bad request, so status
    alone proves nothing -- the magic-bytes check is load-bearing, not defensive
    padding. Returns a path the caller must delete, or None.
    """
    for attempt in range(1, IVT_ATTEMPTS + 1):
        try:
            r = session.get(url, timeout=(IVT_CONNECT_TIMEOUT, IVT_READ_TIMEOUT))
        except Exception as e:
            _LAST_FAIL[tag.split()[0]] = f"{type(e).__name__}: {str(e)[:80]}"
            logging.warning(f"[IVT] {tag} attempt {attempt} request error: {e}")
            time.sleep(IVT_REQUEST_PAUSE_S * attempt)
            continue
        if r.status_code != 200:
            _LAST_FAIL[tag.split()[0]] = f"HTTP {r.status_code}"
            # 403/429 is NOMADS declining, not a bad request. Back off much harder than the
            # normal pause -- the standard 4s retry just spends the remaining budget being
            # refused faster.
            if r.status_code in (403, 429, 503):
                logging.warning(f"[IVT] {tag} attempt {attempt} HTTP {r.status_code} "
                                f"(throttled) - backing off {IVT_THROTTLE_BACKOFF_S * attempt}s")
                time.sleep(IVT_THROTTLE_BACKOFF_S * attempt)
            else:
                logging.warning(f"[IVT] {tag} attempt {attempt} HTTP {r.status_code}")
                time.sleep(IVT_REQUEST_PAUSE_S * attempt)
            continue
        if not r.content.startswith(b"GRIB"):
            snippet = r.content[:160].decode("utf-8", "replace").replace("\n", " ")
            logging.warning(f"[IVT] {tag} 200 but not GRIB: {snippet}")
            return None       # a malformed request will not fix itself on retry
        fd, path = tempfile.mkstemp(suffix=".grib2", prefix="ivt_")
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)
        return path
    return None


def _crop_slices(lats, lons, domain, margin=2.0):
    """Row/col slices trimming a REGULAR lat/lon grid to the map domain.

    ECMWF hands back the whole globe on every step. Integrating 8 levels x 3 params of
    721x1440 when the map covers 40x30 degrees wastes most of the work and the memory, so
    each field is trimmed as it is decoded rather than after. Returns None for a grid that
    is not regular (RAP and NAM are Lambert, where a lat/lon box is not a rectangle) --
    those arrive pre-subset from NOMADS anyway.
    """
    try:
        lat1d, lon1d = lats[:, 0], lons[0, :]
        # Regular means every row shares one longitude vector and every column one latitude.
        if not (np.allclose(lats[0, :], lats[0, 0]) and np.allclose(lons[:, 0], lons[0, 0])):
            return None
        lon1d = np.where(lon1d > 180.0, lon1d - 360.0, lon1d)
        rows = np.where((lat1d >= domain["bottom"] - margin) &
                        (lat1d <= domain["top"] + margin))[0]
        cols = np.where((lon1d >= domain["left"] - margin) &
                        (lon1d <= domain["right"] + margin))[0]
        if rows.size < 4 or cols.size < 4:
            return None
        return (slice(int(rows.min()), int(rows.max()) + 1),
                slice(int(cols.min()), int(cols.max()) + 1))
    except Exception:
        return None


def _ecmwf_client():
    """Imported lazily so ivt_maps still loads when ecmwf-opendata is absent."""
    from ecmwf.opendata import Client
    return Client(source="ecmwf")


def _ecmwf_cycle():
    """Newest ECMWF cycle that runs the full forecast length.

    client.latest() reports whatever is newest, including 06z and 18z -- but those stop
    near f90, so a step list reaching f144 would come back half empty with no error. Snap
    back to the preceding 00z or 12z instead.
    """
    try:
        dt_ = _ecmwf_client().latest(type="fc", param="msl", levtype="sfc")
    except Exception as e:
        logging.error(f"[IVT] ecmwf: latest() failed: {type(e).__name__}: {str(e)[:160]}")
        return None, None
    while dt_.hour not in (0, 12):
        dt_ = dt_ - datetime.timedelta(hours=1)
    logging.info(f"[IVT] ecmwf: using cycle {dt_:%Y%m%d} {dt_:%H}z")
    return dt_.strftime("%Y%m%d"), dt_.strftime("%H")


def _fetch_ecmwf(date_str, cycle, fh, tag=""):
    """Two retrieves per step: pressure levels, then surface. Returns [paths] or None.

    They cannot be combined -- levtype is a single value per request. Both files are handed
    to grid_ivt together, which is why it accepts a list.
    """
    cfg = IVT_MODELS["ecmwf"]
    client = _ecmwf_client()
    out = []
    for kwargs, what in (
        (dict(levtype="pl", levelist=cfg["levels"], param=["q", "u", "v"]), "pl"),
        (dict(levtype="sfc", param=["sp", "msl"]), "sfc"),
    ):
        fd, path = tempfile.mkstemp(suffix=".grib2", prefix="ivt_ec_")
        os.close(fd)
        try:
            client.retrieve(date=date_str, time=int(cycle), type="fc",
                            step=[fh], target=path, **kwargs)
            if os.path.getsize(path) == 0:
                raise ValueError("empty file")
            out.append(path)
        except Exception as e:
            logging.warning(f"[IVT] {tag} {what} retrieve failed: "
                            f"{type(e).__name__}: {str(e)[:150]}")
            try:
                os.unlink(path)
            except Exception:
                pass
            # The surface half is optional: without sp the below-ground mask degrades and
            # without msl the contours vanish, but IVT itself still integrates.
            if what == "pl":
                for p in out:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
                return None
    return out


def determine_cycle(session, model):
    """Walk back from the most recent nominal cycle until one answers with real GRIB.
    Probes a single level of a single variable so the check costs almost nothing."""
    cfg = IVT_MODELS[model]
    now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=cfg["latency_h"])
    transient = False          # did anything look like a refusal rather than a 404?
    first_candidate = None     # newest nominal cycle, used by the fallback
    for back in range(0, 5):
        t = now - datetime.timedelta(hours=6 * back)
        cyc = max([c for c in cfg["cycles"] if c <= t.hour], default=None)
        if cyc is None:
            t = t - datetime.timedelta(days=1)
            cyc = cfg["cycles"][-1]
        date_str, cycle = t.strftime("%Y%m%d"), f"{cyc:02d}"
        probe = (f"https://nomads.ncep.noaa.gov/cgi-bin/{cfg['filter']}"
                 f"?file={cfg['file'](cycle, 0)}&lev_500_mb=on&var_UGRD=on"
                 # One-degree box over Vandenberg. This only has to prove the cycle EXISTS,
                 # so it is deliberately tiny -- but it must sit inside every model's grid or
                 # a live cycle reads as missing. The Cape build's -81/-80, 28/29 box is open
                 # Atlantic from here and outside the CONUS models entirely.
                 f"&subregion=&leftlon=-121&rightlon=-120&toplat=35&bottomlat=34"
                 f"&dir={cfg['dir'](date_str, cycle)}")
        if first_candidate is None:
            first_candidate = (date_str, cycle)

        for attempt in range(1, IVT_CYCLE_ATTEMPTS + 1):
            try:
                r = session.get(probe, timeout=(IVT_CONNECT_TIMEOUT, 30))
            except Exception as e:
                # Connection-level failure: transient by definition.
                transient = True
                logging.warning(f"[IVT] {model} cycle probe {date_str} {cycle}z "
                                f"attempt {attempt}: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(2.0 * attempt)
                continue
            if r.status_code == 200 and r.content.startswith(b"GRIB"):
                logging.info(f"[IVT] {model}: using cycle {date_str} {cycle}z")
                return date_str, cycle
            # Log WHAT failed. The old code swallowed this entirely, which made a NOMADS
            # refusal indistinguishable from a cycle that had not posted yet.
            if r.status_code == 200:
                body = r.content[:140].decode("utf-8", "replace").replace("\n", " ")
                logging.warning(f"[IVT] {model} cycle probe {date_str} {cycle}z "
                                f"attempt {attempt}: 200 but not GRIB :: {body}")
            else:
                logging.warning(f"[IVT] {model} cycle probe {date_str} {cycle}z "
                                f"attempt {attempt}: HTTP {r.status_code}")
            # 404 means this cycle genuinely is not there -- walk back rather than retry.
            # Anything else (403/429/5xx) is the server declining, so retry this cycle.
            if r.status_code == 404:
                break
            transient = True
            time.sleep(2.0 * attempt)
        time.sleep(1.0)

    if IVT_CYCLE_FALLBACK and transient and first_candidate:
        logging.warning(f"[IVT] {model}: every cycle probe failed transiently; falling back to "
                        f"{first_candidate[0]} {first_candidate[1]}z and letting the fetch decide")
        return first_candidate
    logging.error(f"[IVT] {model}: no usable cycle found "
                  f"(transient={transient}; see the probe warnings above)")
    return None, None


# ---------------------------------------------------------------------------
# Grid integration
# ---------------------------------------------------------------------------
def _sat_vap_hpa(tc):
    return 6.112 * np.exp((17.67 * tc) / (tc + 243.5))


def _grid_moisture(fields, level, route):
    """Specific humidity (kg/kg) on the grid for one level.

    Raises on a missing field rather than returning zeros. RAP and NAM return
    perfectly valid GRIB with no moisture in it at all when SPFH is requested; a
    silent zero there produces a smooth, plausible, completely wrong dry map.
    """
    if route == "spfh":
        q = fields.get(("q", level))
        if q is None:
            raise KeyError(f"no specific humidity at {level} hPa")
        return q
    rh, t = fields.get(("r", level)), fields.get(("t", level))
    if rh is None or t is None:
        raise KeyError(f"no RH/TMP pair at {level} hPa")
    tc = t - 273.15
    e = _sat_vap_hpa(tc) * np.clip(rh, 0.0, 100.0) / 100.0
    e = np.minimum(e, level * 0.999)          # guard the e >= p singularity
    w = EPSILON * e / (level - e)
    return w / (1.0 + w)


def grid_ivt(filepaths, route, crop_domain=None):
    """Integrate IVT across the whole grid.

    Returns (lons, lats, ivt_mag, ivt_u, ivt_v, mslp_hpa, levels_used).
    Vertical integration is the same trapezoid ivt.column_ivt() uses, vectorised.
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    fields, lats, lons, sfc_pres, mslp = {}, None, None, None, None
    sl = None                      # crop slices, computed once from the first field
    wanted = set(IVT_LEVELS) | set(ECMWF_IVT_LEVELS)
    for filepath in filepaths:
        grbs = pygrib.open(filepath)
        for g in grbs:
            sn = getattr(g, "shortName", "")
            tl = getattr(g, "typeOfLevel", "")
            if lats is None:
                lats, lons = g.latlons()
                if crop_domain is not None:
                    sl = _crop_slices(lats, lons, crop_domain)
                    if sl is not None:
                        lats, lons = lats[sl], lons[sl]
            if tl == "isobaricInhPa" and g.level in wanted:
                if sn in ("q", "r", "t", "u", "v"):
                    v = np.asarray(g.values, dtype=np.float64)
                    fields[(sn, g.level)] = v[sl] if sl is not None else v
            elif sn in ("sp", "pres") and tl == "surface":
                v = np.asarray(g.values, dtype=np.float64) / 100.0      # Pa -> hPa
                sfc_pres = v[sl] if sl is not None else v
            elif sn in ("prmsl", "msl", "mslet", "mslma"):
                v = np.asarray(g.values, dtype=np.float64) / 100.0
                mslp = v[sl] if sl is not None else v
        grbs.close()

    if lats is None:
        raise ValueError("no fields decoded")

    # Levels present with a complete (moisture, u, v) triple. Descending pressure.
    usable = []
    for lv in sorted(wanted, reverse=True):
        if ("u", lv) not in fields or ("v", lv) not in fields:
            continue
        try:
            q = _grid_moisture(fields, lv, route)
        except KeyError:
            continue
        usable.append((lv, q, fields[("u", lv)], fields[("v", lv)]))

    if len(usable) < 6:
        raise ValueError(f"only {len(usable)} complete levels; refusing to integrate")

    shape = usable[0][1].shape
    if sfc_pres is None:
        # No surface pressure: assume everything at or below 1000 hPa is valid.
        # THIS MATTERS MORE ON THIS DOMAIN THAN IT DID OVER FLORIDA. The old box was
        # nearly all ocean and near-sea-level coast, so the assumption cost almost
        # nothing. Here the box holds the Coast Ranges, the Sierra and the Great Basin,
        # where the true surface is hundreds of hPa above 1000 -- integrating those
        # below-ground levels inflates IVT over exactly the terrain an AR is running
        # into. Every model in IVT_MODELS requests var_PRES, so this branch should not
        # fire; if the warning below appears, treat inland values as suspect.
        sfc_pres = np.full(shape, 1013.0)
        logging.warning("[IVT] no surface pressure field; below-ground levels not masked")

    acc_u = np.zeros(shape)
    acc_v = np.zeros(shape)
    for (p1, q1, u1, v1), (p2, q2, u2, v2) in zip(usable[:-1], usable[1:]):
        # Partial bottom slab: clip the lower bound at the surface so a level buried
        # underground contributes nothing and the slab straddling the surface
        # contributes only its above-ground fraction.
        p_low = np.minimum(np.full(shape, float(p1)), sfc_pres)
        dp = np.clip(p_low - p2, 0.0, None) * 100.0        # hPa -> Pa
        acc_u += 0.5 * (q1 * u1 + q2 * u2) * dp
        acc_v += 0.5 * (q1 * v1 + q2 * v2) * dp

    ivt_u, ivt_v = acc_u / G, acc_v / G
    return (lons, lats, np.hypot(ivt_u, ivt_v), ivt_u, ivt_v, mslp,
            [lv for lv, _, _, _ in usable])


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_ivt_map(model, lons, lats, mag, ivt_u, ivt_v, mslp,
                   cycle_label, fh, valid_str, levels_used):
    """One CW3E-style IVT panel: shaded magnitude, transport vectors, MSLP contours."""
    try:
        out_dir = os.path.join(IVT_MAPS_DIR, model)
        os.makedirs(out_dir, exist_ok=True)
        cmap, norm = cw3e_ivt_cmap()
        proj = ccrs.PlateCarree()

        fig = plt.figure(figsize=(8.0, 6.4), dpi=115)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        dom = model_domain(model)
        ax.set_extent([dom["left"], dom["right"],
                       dom["bottom"], dom["top"]], crs=proj)

        # Light basemap, unlike the dark reflectivity maps: the CW3E ramp is a warm
        # yellow-to-purple sequence and needs a pale ground to read against.
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eef2f7", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e6e6e0", zorder=0)

        plot_lons = np.where(lons > 180, lons - 360.0, lons)
        masked = np.ma.masked_less(mag, IVT_MIN_SHADE)
        mesh = ax.pcolormesh(plot_lons, lats, masked, cmap=cmap, norm=norm,
                             shading="auto", transform=proj, zorder=1)

        if mslp is not None:
            lo = math.floor(np.nanmin(mslp) / MSLP_INTERVAL_HPA) * MSLP_INTERVAL_HPA
            hi = math.ceil(np.nanmax(mslp) / MSLP_INTERVAL_HPA) * MSLP_INTERVAL_HPA
            cs = ax.contour(plot_lons, lats, mslp,
                            levels=np.arange(lo, hi + 1, MSLP_INTERVAL_HPA),
                            colors="#111827", linewidths=0.6, transform=proj, zorder=4)
            ax.clabel(cs, inline=True, fontsize=5.5, fmt="%d")

        # Vectors show the DIRECTION OF TRANSPORT, matching the CW3E convention --
        # the way the vapour is moving, not the meteorological FROM direction used
        # everywhere else in the dashboard. Subsampled or the field is unreadable.
        # Stride is in GRID CELLS, which made it a per-model constant only as long as every
        # panel shared one box and one grid spacing. Neither holds now: GFS runs on the wide
        # Pacific box at 0.25 deg while RAP runs on the CONUS box at 13 km, and a stride tuned
        # for one is a solid mat of arrows on the other. Solve for the stride that lands the
        # target number of arrows across the panel instead, so density is a property of the
        # FIGURE rather than of whichever grid happened to produce the field.
        s = max(1, int(round(mag.shape[1] / IVT_VECTOR_TARGET_COLS)))
        vec_mask = mag[::s, ::s] >= IVT_MIN_SHADE
        if np.any(vec_mask):
            ax.quiver(plot_lons[::s, ::s][vec_mask], lats[::s, ::s][vec_mask],
                      ivt_u[::s, ::s][vec_mask], ivt_v[::s, ::s][vec_mask],
                      transform=proj, zorder=5, scale=12000, width=0.0022,
                      color="#1f2937", alpha=0.75)

        ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                       edgecolor="#374151", linewidth=0.7, zorder=6)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                       edgecolor="#6b7280", linewidth=0.5, zorder=6)
        ax.add_feature(cfeature.STATES.with_scale("50m"),
                       edgecolor="#9ca3af", linewidth=0.4, zorder=6)

        span = f"{max(levels_used)}\u2013{min(levels_used)} hPa"
        ax.set_title(f"{model.upper()} IVT (kg m\u207b\u00b9 s\u207b\u00b9), transport vectors, "
                     f"MSLP (hPa)\nInit {cycle_label}   F{fh:03d}   Valid {valid_str}   "
                     f"[{len(levels_used)} lv, {span}]",
                     fontsize=8.5, fontweight="bold", color="#0f172a")

        # Label every boundary, as CW3E does -- with uneven bin widths, unlabelled
        # blocks would read as an evenly spaced ramp and misstate the top of the scale.
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.036, pad=0.02, extend="max",
                            ticks=CW3E_IVT_BOUNDS, spacing="uniform")
        cbar.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        out_name = f"ivt_{cycle_label.replace(' ', '_')}_f{fh:03d}.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, format="png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        return f"maps/ivt/{model}/{out_name}"
    except Exception as e:
        logging.error(f"[IVT] render failed {model} f{fh:03d}: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def fetch_ivt_maps(models=None, session=None):
    """Build the IVT map set.

    Returns {model: {"cycle": "YYYYMMDD HHz", "maps": {"f024": "maps/ivt/..."}}}
    ready to drop into the history.json payload. Never raises: a failure anywhere
    degrades to fewer maps, never to a failed pipeline run.
    """
    if not IVT_MAPS_ENABLED:
        return {}
    models = models or [m for m, c in IVT_MODELS.items() if c["enabled"]]
    own_session = session is None
    if own_session:
        session = requests.Session()
        session.headers.update({"User-Agent": IVT_UA})

    started = time.time()
    out = {}
    try:
        for model in models:
            cfg = IVT_MODELS[model]
            if cfg.get("source") == "ecmwf":
                date_str, cycle = _ecmwf_cycle()
            else:
                date_str, cycle = determine_cycle(session, model)
            if not date_str:
                continue
            cycle_label = f"{date_str} {cycle}z"
            init = datetime.datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H")
            maps, attempted, consecutive_fail = {}, 0, 0

            model_started = time.time()
            for fh in cfg["steps"]:
                if time.time() - started > IVT_BUDGET_S:
                    logging.warning(f"[IVT] budget spent; stopping {model} at f{fh:03d}")
                    break
                if time.time() - model_started > IVT_MODEL_BUDGET_S:
                    logging.warning(f"[IVT] {model} hit its own budget at f{fh:03d}")
                    break
                attempted += 1
                if cfg.get("source") == "ecmwf":
                    path = _fetch_ecmwf(date_str, cycle, fh, tag=f"ecmwf f{fh:03d}")
                else:
                    path = _fetch_grib(session, _build_url(model, date_str, cycle, fh),
                                       tag=f"{model} f{fh:03d}")
                if not path:
                    # Degrade rather than lose the step: retry without MSLP. IVT itself needs
                    # no sea-level pressure, so a wrong variable name should cost the contours
                    # and nothing more.
                    time.sleep(IVT_REQUEST_PAUSE_S)
                    path = _fetch_grib(session,
                                       _build_url(model, date_str, cycle, fh, with_mslp=False),
                                       tag=f"{model} f{fh:03d} (no MSLP)")
                    if path:
                        logging.warning(f"[IVT] {model} f{fh:03d}: rendered without MSLP "
                                        f"contours (check the 'mslp' var name for {model})")
                if not path:
                    consecutive_fail += 1
                    if consecutive_fail >= IVT_MAX_CONSECUTIVE_FAILURES:
                        logging.error(f"[IVT] {model}: {consecutive_fail} consecutive step "
                                      f"failures from {cycle_label}; abandoning this model "
                                      f"(last failure: {_LAST_FAIL.get(model, 'unknown')})")
                        break
                    time.sleep(IVT_REQUEST_PAUSE_S)
                    continue
                consecutive_fail = 0
                try:
                    # Crop only the global model; NOMADS output is already subset, and RAP/NAM
                    # are Lambert grids where a lat/lon box is not a rectangle.
                    crop = model_domain(model) if cfg.get("source") == "ecmwf" else None
                    lons, lats, mag, iu, iv, mslp, lv = grid_ivt(path, cfg["moisture"],
                                                                 crop_domain=crop)
                    valid = (init + datetime.timedelta(hours=fh)).strftime("%d %b %HZ")
                    rel = render_ivt_map(model, lons, lats, mag, iu, iv, mslp,
                                         cycle_label, fh, valid, lv)
                    if rel:
                        maps[f"f{fh:03d}"] = rel
                except Exception as e:
                    logging.error(f"[IVT] {model} f{fh:03d} integrate/render: {e}")
                finally:
                    for _p in ([path] if isinstance(path, str) else (path or [])):
                        try:
                            os.unlink(_p)
                        except Exception:
                            pass
                time.sleep(IVT_REQUEST_PAUSE_S)

            if maps:
                out[model] = {"cycle": cycle_label, "maps": maps}
            logging.info(f"[IVT] {model}: {len(maps)}/{attempted} steps rendered "
                         f"from {cycle_label}")
    except Exception as e:
        logging.error(f"[IVT] driver aborted: {e}")
    finally:
        if own_session:
            session.close()

    total = sum(len(v["maps"]) for v in out.values())
    logging.info(f"[IVT] complete: {total} maps across {len(out)} model(s) "
                 f"in {time.time() - started:.0f}s")
    return out


def prune_ivt_maps(result):
    """Delete IVT PNGs that are not part of the current map set.

    Nothing else prunes maps/ivt: prune_stale_maps() only unlinks top-level files in maps/,
    and unlink on a subdirectory fails silently. Filenames carry the init cycle, so RAP --
    an hourly model -- writes 22 new PNGs every hour and the previous 22 stay forever, all
    of it committed to git. Left alone this is hundreds of MB of history a week.
    """
    keep = set()
    for entry in (result or {}).values():
        for rel in (entry.get("maps") or {}).values():
            keep.add(os.path.basename(rel))
    removed = 0
    for model in (result or {}):
        d = os.path.join(IVT_MAPS_DIR, model)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".png") and f not in keep:
                try:
                    os.unlink(os.path.join(d, f))
                    removed += 1
                except Exception:
                    pass
    if removed:
        logging.info(f"[IVT] pruned {removed} stale PNG(s) from {IVT_MAPS_DIR}")
    return removed


def write_ivt_json(result, path="ivt_maps.json"):
    """Write the map index as a standalone sidecar.

    Deliberately NOT merged into history.json: this runs in its own workflow now, and two
    workflows rewriting the same file is a merge conflict on every overlapping run. The page
    reads this file directly and falls back to history.json's ivt_maps key.
    """
    import json
    payload = {"generated": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%d %H:%M UTC"),
               "models": result or {}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    n = sum(len(v.get("maps", {})) for v in (result or {}).values())
    logging.info(f"[IVT] wrote {path}: {n} maps across {len(result or {})} model(s)")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="one model, three steps -- cheap check that the path still works")
    ap.add_argument("--models", default=None, help="comma-separated subset")
    args, _ = ap.parse_known_args()

    if args.smoke:
        for m in ("rap", "nam", "ecmwf"):
            IVT_MODELS[m]["enabled"] = False
        IVT_MODELS["gfs"]["steps"] = [0, 24, 48]

    sel = args.models.split(",") if args.models else None
    res = fetch_ivt_maps(models=sel)
    prune_ivt_maps(res)
    write_ivt_json(res)

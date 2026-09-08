"""
ivt.py — Integrated Vapor Transport for the Vandenberg (Western Range) aviation dashboard.

Self-contained. Imports nothing from aviation_dashboard.py, so it can be unit-tested
in isolation and dropped in without touching the import graph.

    IVT = (1/g) * | integral_{p_top}^{p_sfc} q * V dp |      [kg m^-1 s^-1]

Two entry points:

    column_ivt(layers, ...)      -> dict, or None if the column can't support the integral
    ivt_from_profile(hour_dict)  -> same, reading the `_layers` that
                                    compute_profile_variables() already returns

Because every profile in the pipeline already carries pres / tmpc / dwpt / rh / u / v,
this runs on GFS, RAP, HRRR (BUFKIT and NOMADS pad columns), ECMWF HRES, RRFS, REFS,
GEFS members and ECMWF ENS members with no additional network fetching.

UNITS, which is where this goes wrong silently:
  * layers["pres"]  hPa
  * layers["u"],["v"]  KNOTS. Both producers agree on this -- parse_time_series_bufkit
    derives them from SKNT, and _grib_levels_to_layers multiplies the GRIB m/s by
    1.943844 on ingest. Feeding knots into the integral unconverted inflates IVT by
    exactly 1.9438x, which lands values in a range that still looks physical.
  * layers["dwpt"],["tmpc"]  degrees C
  * output IVT  kg m^-1 s^-1
"""

import math

G = 9.80665              # m s^-2
KT_TO_MS = 0.514444
EPSILON = 0.622          # Rd/Rv

# Integration bounds. 300 hPa is the standard AR-literature top (Ralph/Neiman, and the
# convention CW3E's operational plots use); above it the moisture contribution is
# under a percent and the extra levels mostly add noise. Worth keeping as-is on this
# domain in particular: CW3E's plots ARE the west-coast AR reference product, so matching
# their integration depth is what makes these panels directly comparable to theirs.
IVT_TOP_HPA = 300.0
IVT_BASE_HPA = 1000.0

# ---- Quality gates -----------------------------------------------------------------
# A column thin on humidity produces an IVT that is ordinary-looking and meaningless --
# the same failure already documented for Thompson/PWAT in PANEL_MIN_RH_LEVELS. The
# gate is on the DATA, not the model name, so any column that later gets fuller
# moisture starts working with no code change.
#
# 6 matches PANEL_MIN_RH_LEVELS deliberately. Keep the two in step: if one moves and
# the other doesn't, the panel will show a PWAT with no IVT beside it (or the reverse)
# and the blank cell will read as a fetch failure rather than a deliberate gate.
IVT_MIN_MOIST_LEVELS = 6

# The integral must actually span the layer that carries the transport. Most IVT lives
# below 500 hPa, so a column that stops at 700 is not "slightly truncated", it is
# missing a large fraction of the flux while still returning a confident number.
IVT_REQUIRE_BOTTOM_HPA = 850.0   # deepest level must be at least this deep
IVT_REQUIRE_TOP_HPA = 400.0      # shallowest level must reach at least this high


def _sat_vap_hpa(tc):
    """Saturation vapour pressure (hPa) over liquid water, Bolton (1980)."""
    return 6.112 * math.exp((17.67 * tc) / (tc + 243.5))


def specific_humidity(layer):
    """Specific humidity (kg/kg) for one layer, or None if it can't be determined.

    Dewpoint is preferred over RH: the GRIB path already derives dwpt from native RH
    via _rh_to_dewpoint_c, so using dwpt keeps one conversion instead of round-tripping
    through a second one. RH is the fallback for any producer that omits dwpt.
    """
    p = layer.get("pres")
    if p is None or p <= 0:
        return None

    td = layer.get("dwpt")
    if td is None:
        rh, tc = layer.get("rh"), layer.get("tmpc")
        if rh is None or tc is None or rh <= 0:
            return None
        e = _sat_vap_hpa(tc) * (max(1.0, min(100.0, rh)) / 100.0)
    else:
        e = _sat_vap_hpa(td)

    # Guard the singularity: e >= p is unphysical and would drive w negative.
    if e >= p:
        return None
    w = EPSILON * e / (p - e)          # mixing ratio, kg/kg
    return w / (1.0 + w)               # specific humidity


def _interp_to_pressure(lo, hi, target_p, key_fn):
    """Linearly interpolate a derived quantity between two layers straddling target_p.
    Interpolation is linear in pressure, matching how the trapezoid treats each slab."""
    p_lo, p_hi = lo["pres"], hi["pres"]
    if p_lo == p_hi:
        return key_fn(lo)
    v_lo, v_hi = key_fn(lo), key_fn(hi)
    if v_lo is None or v_hi is None:
        return None
    f = (target_p - p_lo) / (p_hi - p_lo)
    return v_lo + f * (v_hi - v_lo)


def column_ivt(layers, top_hpa=IVT_TOP_HPA, sfc_pres_hpa=None,
               min_moist_levels=IVT_MIN_MOIST_LEVELS, strict=True):
    """Integrate IVT over one profile.

    Args:
        layers:           profile_layers list (pres/tmpc/dwpt/rh/u/v), any order.
        top_hpa:          integration top. 300 hPa by convention.
        sfc_pres_hpa:     surface pressure, if known. Levels below ground are dropped.
                          The NOMADS pad fetch already requests var_PRES, so this is
                          available on that path; BUFKIT columns start at the surface
                          already and can pass None.
        min_moist_levels: humidity levels required before the result is trusted.
        strict:           when True, apply the depth gates. Set False only for
                          diagnostics -- a non-strict result is not panel-safe.

    Returns dict, or None when the column cannot support the integral:
        ivt        magnitude, kg m^-1 s^-1
        ivt_u      eastward component
        ivt_v      northward component
        ivt_dir_to    compass bearing the vapour is moving TOWARD (CW3E vector sense)
        ivt_dir_from  meteorological FROM direction, matching layer["drct"] convention
        p_bot, p_top  pressure bounds actually integrated (hPa)
        n_levels      levels used
        truncated     True if the column stopped short of top_hpa
    """
    if not layers:
        return None

    usable = []
    for l in layers:
        p = l.get("pres")
        u, v = l.get("u"), l.get("v")
        if p is None or u is None or v is None:
            continue
        if p > (sfc_pres_hpa or IVT_BASE_HPA * 1.1):
            continue                      # below ground
        if p < top_hpa - 1e-6:
            continue                      # above the integration top
        q = specific_humidity(l)
        if q is None:
            continue
        usable.append({
            "pres": p,
            "qu": q * (u * KT_TO_MS),     # kt -> m/s HERE, once
            "qv": q * (v * KT_TO_MS),
        })

    if len(usable) < min_moist_levels:
        return None

    # Descending pressure: bottom of the column first.
    usable.sort(key=lambda d: d["pres"], reverse=True)
    p_bot, p_top = usable[0]["pres"], usable[-1]["pres"]

    if strict:
        if p_bot < IVT_REQUIRE_BOTTOM_HPA:
            return None
        if p_top > IVT_REQUIRE_TOP_HPA:
            return None

    # Trapezoid in pressure. dp in Pa, so the result is kg m^-1 s^-1 directly.
    acc_u = acc_v = 0.0
    for a, b in zip(usable[:-1], usable[1:]):
        dp = (a["pres"] - b["pres"]) * 100.0       # hPa -> Pa, positive
        acc_u += 0.5 * (a["qu"] + b["qu"]) * dp
        acc_v += 0.5 * (a["qv"] + b["qv"]) * dp

    ivt_u, ivt_v = acc_u / G, acc_v / G
    mag = math.hypot(ivt_u, ivt_v)

    # Bearing the transport is moving toward, and the FROM direction in the same
    # convention every other wind field in the matrix uses.
    dir_to = math.degrees(math.atan2(ivt_u, ivt_v)) % 360.0
    dir_from = math.degrees(math.atan2(-ivt_u, -ivt_v)) % 360.0

    return {
        "ivt": round(mag, 1),
        "ivt_u": round(ivt_u, 1),
        "ivt_v": round(ivt_v, 1),
        "ivt_dir_to": round(dir_to),
        "ivt_dir_from": round(dir_from),
        "p_bot": round(p_bot),
        "p_top": round(p_top),
        "n_levels": len(usable),
        "truncated": p_top > top_hpa + 1e-6,
    }


def ivt_from_profile(hour_dict, **kw):
    """Convenience wrapper: read `_layers` straight off a compute_profile_variables()
    result. Returns None when the key is absent, so it is safe on any hour dict."""
    if not hour_dict:
        return None
    return column_ivt(hour_dict.get("_layers") or [], **kw)


# ---- CW3E colour scale --------------------------------------------------------------
# Matched to the operational CW3E IVT plots (cw3e.ucsd.edu/ivt_iwv_namerica): 11 discrete
# blocks, boundaries at 250/300/400/500/600/700/800/1000/1200/1400/1600, running yellow
# through orange and red into purple at the top. Note the UNEVEN spacing -- 100-wide bins
# up to 800, then 200-wide above it. That is theirs, not a mistake: it puts the resolution
# in the range ordinary synoptic transport occupies and lets the AR tail share one bar.
#
# Colours are matched by eye against their published plots, not sampled from their colour
# table. Close enough to read side by side; not an exact reproduction.
#
# This scale was BUILT for West Coast atmospheric rivers, so on this domain it is finally
# being used for the thing it was designed for. Expect the full range to be exercised: an
# ordinary winter day sits in the bottom two or three blocks, a moderate AR runs 500-800,
# and a strong landfall pushes into the red and purple. (On the Cape build the same scale
# spent nearly all year in the bottom blocks, because Florida IVT rarely leaves 200-600.)
CW3E_IVT_BOUNDS = [250, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400, 1600]

# 11 colours against 11 bounds is deliberate, not an off-by-one: BoundaryNorm with
# extend="max" counts 10 interior bins PLUS one over-range bin, so it needs 11 colours
# or it raises "ncolors must equal or exceed the number of bins".
CW3E_IVT_COLORS = [
    "#ffff00",   # 250-300    yellow
    "#ffe600",   # 300-400
    "#ffcc00",   # 400-500
    "#ffa700",   # 500-600
    "#ff8c00",   # 600-700
    "#ff6a00",   # 700-800
    "#ff2b00",   # 800-1000   red
    "#d10000",   # 1000-1200
    "#a80028",   # 1200-1400  dark red
    "#7b1fa2",   # 1400-1600  purple
    "#4a0072",   # >1600      dark purple
]


def cw3e_ivt_cmap():
    """Return (cmap, norm) for pcolormesh. Values under 250 are fully transparent so
    the coastline shows through, matching the existing map renderers in the pipeline."""
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(CW3E_IVT_COLORS)
    cmap.set_under((0, 0, 0, 0))
    cmap.set_over(CW3E_IVT_COLORS[-1])
    norm = mcolors.BoundaryNorm(CW3E_IVT_BOUNDS, cmap.N, extend="max")
    return cmap, norm


# ---- Self-test ----------------------------------------------------------------------
LEVELS_17 = [1000, 975, 950, 925, 900, 850, 800, 750, 700,
             650, 600, 550, 500, 450, 400, 350, 300]

# Realistic RH(p) shapes. An earlier version of this fixture lapsed DEWPOINT linearly
# with height, which left the column ~5 C from saturation at 300 hPa and produced a
# ~980 kg/m/s "typical summer day" -- roughly double reality. Moisture has to fall off
# far faster than temperature aloft, so the fixture specifies RH and derives dewpoint.
RH_PROFILES = {
    "moist": [(1000, 85), (925, 82), (850, 78), (700, 58),
              (500, 38), (400, 25), (300, 18)],
    "dry":   [(1000, 55), (925, 45), (850, 35), (700, 25),
              (500, 18), (400, 12), (300, 10)],
    "tropical": [(1000, 88), (925, 87), (850, 85), (700, 72),
                 (500, 52), (400, 35), (300, 22)],
}


def _rh_at(profile, p):
    pts = RH_PROFILES[profile]
    if p >= pts[0][0]:
        return pts[0][1]
    if p <= pts[-1][0]:
        return pts[-1][1]
    for (p1, r1), (p2, r2) in zip(pts[:-1], pts[1:]):
        if p2 <= p <= p1:
            f = (p - p1) / (p2 - p1)
            return r1 + f * (r2 - r1)
    return pts[-1][1]


def _synthetic_profile(moisture="moist", wind_kt=25.0, drct=200.0, sfc_t=28.0):
    """Subtropical column with uniform wind, for order-of-magnitude checking.
    Temperature on a 6.5 C/km lapse rate; humidity from a realistic RH(p) shape."""
    out = []
    u = -wind_kt * math.sin(math.radians(drct))
    v = -wind_kt * math.cos(math.radians(drct))
    for p in LEVELS_17:
        # 44.3308 * (1 - (p/p0)^0.190284) yields KILOMETRES already. Dividing by
        # 3.28084 on top of that put 850 hPa at 444 m instead of 1457 m, kept the
        # column far too warm aloft, and inflated 850 mb q to ~18 g/kg -- a 3.6 in
        # PW that no XMR sounding has ever produced.
        z_km = 44.3308 * (1.0 - (p / 1013.25) ** 0.190284)
        tmpc = sfc_t - 6.5 * z_km
        out.append({"pres": p, "tmpc": tmpc, "dwpt": None,
                    "rh": _rh_at(moisture, p), "u": u, "v": v})
    return out


def _analytic_check():
    """Validate the integrator against a case with a closed-form answer.

    Constant q and constant wind over a slab reduces the integral to

        IVT = q * V * dp / g

    so any error in the trapezoid, the hPa->Pa conversion, or the knots->m/s
    conversion shows up immediately as a ratio away from 1.000. This is the test that
    actually proves the math; the profiles above only prove it is plausible.
    """
    q_target, v_ms, p_bot, p_top = 0.010, 10.0, 1000.0, 300.0
    expected = q_target * v_ms * (p_bot - p_top) * 100.0 / G

    # Build layers whose specific humidity is exactly q_target at every level, by
    # inverting q -> e -> dewpoint at each pressure.
    layers = []
    for p in LEVELS_17:
        w = q_target / (1.0 - q_target)
        e = w * p / (EPSILON + w)
        td = 243.5 * math.log(e / 6.112) / (17.67 - math.log(e / 6.112))
        layers.append({"pres": p, "tmpc": 30.0, "dwpt": td, "rh": None,
                       "u": 0.0, "v": v_ms / KT_TO_MS})   # due-south wind, in KNOTS

    got = column_ivt(layers)
    ratio = got["ivt"] / expected
    status = "PASS" if abs(ratio - 1.0) < 0.005 else "FAIL"
    print(f"  analytic slab: expected {expected:7.1f}, got {got['ivt']:7.1f}, "
          f"ratio {ratio:.4f}  [{status}]")
    print(f"    (q=10 g/kg, V=10 m/s, 1000-300 hPa; ratio 1.9438 would mean "
          f"knots were never converted)")
    return status == "PASS"


if __name__ == "__main__":
    print("IVT self-test\n")
    print("  1. Integrator correctness (closed form)")
    _analytic_check()

    print("\n  2. Plausibility on realistic columns")
    print("     PW is printed alongside because for a uniform wind IVT == V * PW,")
    print("     so a PW outside XMR climatology (~0.8-2.3 in) means the FIXTURE is")
    print("     wrong, not the integrator. This is the check that caught a bad")
    print("     height formula that had 850 hPa sitting at 444 m.")
    for moist, kt, dr, note in [
        ("moist", 25.0, 200.0, "moist SSW flow, typical FL summer"),
        ("moist", 45.0, 200.0, "same moisture, stronger flow"),
        ("dry", 25.0, 320.0, "dry post-frontal NW flow"),
        ("tropical", 60.0, 180.0, "deep tropical southerly, TC-adjacent"),
    ]:
        prof = _synthetic_profile(moist, kt, dr)
        r = column_ivt(prof)
        if r is None:
            print(f"    {note:38s} -> gated out")
            continue
        pw_mm = 0.0
        for a, b in zip(prof[:-1], prof[1:]):
            qa, qb = specific_humidity(a), specific_humidity(b)
            pw_mm += 0.5 * (qa + qb) * ((a["pres"] - b["pres"]) * 100.0) / G
        print(f"    {note:38s} -> IVT {r['ivt']:6.1f}  toward {r['ivt_dir_to']:03d} deg"
              f"   PW {pw_mm / 25.4:4.2f} in")

    print("\n  3. Gates")
    thin = _synthetic_profile()[:4]
    print(f"    4-level column          -> {column_ivt(thin)}")
    shallow = [l for l in _synthetic_profile() if l["pres"] >= 700]
    print(f"    surface-700 only        -> {column_ivt(shallow)}")
    print(f"    same, strict=False      -> "
          f"{(column_ivt(shallow, strict=False) or {}).get('ivt')}"
          f"   <- truncated at 700 hPa, hence the gate")

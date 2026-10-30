"""The three sphere presets (DESIGN §46.9; sphere series S7a): ``tiny3s``, ``mini3s`` and ``64k3s``, the
planets of the ``tiny3d``, ``mini3d`` and ``64k3d`` scale, with their calibrated population keys, and the
weather overlays made for them (``stormsph`` for ``64k3s``, ``stormsph22`` for ``mini3s``, ``stormsph16`` for
``tiny3s``), calibrated in S4c.

Configuration only (the 3 000-step runs are in ``test_sphere_presets_slow.py``): the values are the ones the
calibration of S7a measured (DESIGN §46.9), pinned so that a change of one is deliberate.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from alife import config, geom, weather_sph
from alife.config import ConfigError, config_hash, dims, load_config

PRESET_DIR = Path(config.__file__).parent / "presets"

# preset: (radius, height, field nside, layers, hash layers, n_max, n_init, regen_max, C, patches, sigma)
TABLE = {
    "tiny3s": (36.0, 16.0, 8, 8, 5, 1024, 256, 320, 32, 4, 12.0),
    "mini3s": (48.0, 24.0, 11, 12, 7, 20480, 4096, 2400, 32, 3, 18.0),
    "64k3s": (90.0, 40.0, 21, 18, 13, 65536, 16384, 1340, 32, 8, 24.0),
}
BOX_ANALOGUE = {"tiny3s": "tiny3d", "mini3s": "mini3d", "64k3s": "64k3d"}
FLOOR_AREA = {name: 4.0 * 3.141592653589793 * v[0] ** 2 for name, v in TABLE.items()}


def test_the_preset_list_is_exact() -> None:
    """Every place that lists the presets compares with this tuple (the CLI choices, the GUI ``Preset`` enum
    in ``test_gui_sync``): a new preset is a deliberate edit here."""
    assert config.PRESETS == (
        "tiny",
        "dev",
        "64k",
        "256k",
        "tiny3d",
        "64k3d",
        "mini3d",
        "tiny3s",
        "mini3s",
        "64k3s",
    )
    assert tuple(TABLE) == config.SPHERE_PRESETS
    spheres = [p for p in config.PRESETS if load_config(preset=p).world.geometry == "sphere"]
    assert tuple(spheres) == config.SPHERE_PRESETS
    for p in config.PRESETS:
        assert (PRESET_DIR / f"{p}.toml").is_file()


@pytest.mark.parametrize("name", list(TABLE))
def test_every_sphere_preset_validates_with_its_table_values(name: str) -> None:
    radius, height, nside, layers, g_z, n_max, n_init, regen, cap, patches, sigma = TABLE[name]
    cfg = load_config(preset=name)
    d = dims(cfg)
    w = cfg.world
    assert (w.geometry, w.dims, w.radius, w.height, w.nside, w.layers) == (
        "sphere",
        3,
        radius,
        height,
        nside,
        layers,
    )
    assert w.size > 4 * (radius + height)  # the image length of min_image (DESIGN §46.0, row 5)
    assert (d.nside_h, d.g_hash_z, d.c, d.q) == (nside, g_z, cap, 125 * cap)  # Nh = Nf in every preset
    assert (cfg.population.n_max, cfg.population.n_init, cfg.fields.regen_max) == (n_max, n_init, regen)
    assert (cfg.fields.n_patches, cfg.fields.patch_sigma) == (patches, sigma)
    assert d.n >= n_max and d.n % 64 == 0 and n_init <= n_max // 2
    # C clears the lid hash cell with its margin (validate's rule) and the candidate bound K <= Q
    assert (
        cap >= geom.min_cell_capacity(radius, height, cfg.sensing.radius, d.nside_h)
        and cfg.sensing.k_neighbors <= d.q
    )
    assert (
        not cfg.weather.enabled and not cfg.structures.enabled
    )  # a dry world until an overlay says otherwise


@pytest.mark.parametrize("name", ["mini3s", "64k3s"])
def test_the_physics_and_ecology_keys_are_the_3d_preset_of_the_same_scale(name: str) -> None:
    """The per-agent physics and economy are the box analogue's, so only the geometry, the sizes and the
    calibrated population keys differ (docs/SPHERE.md 7.3)."""
    sph, box = load_config(preset=name), load_config(preset=BOX_ANALOGUE[name])
    for section, key in [
        ("physics", "gravity"),
        ("physics", "buoyancy_max"),
        ("energy", "brain_cost_per_l1"),
        ("energy", "buoyancy_cost"),
        ("energy", "buoyancy_cap_cost"),
        ("mutation", "buoy_init"),
        ("mutation", "buoy_sigma"),
        ("sensing", "k_neighbors"),
        ("fields", "benthic_lambda"),
        ("fields", "photic_amp"),
        ("fields", "photic_lambda"),
        ("fields", "chem_diffusion"),
    ]:
        assert getattr(getattr(sph, section), key) == getattr(getattr(box, section), key), (section, key)
    assert (
        sph.run.metrics_every == box.run.metrics_every
        and sph.run.heavy_metrics_every == box.run.heavy_metrics_every
    )


def test_the_presets_have_distinct_hashes_and_the_box_hashes_are_unchanged() -> None:
    hashes = {p: config_hash(load_config(preset=p)) for p in config.PRESETS}
    assert len(set(hashes.values())) == len(hashes)
    # tiny, tiny3d, 64k3d and mini3d are pinned in test_config_sphere.BASE_HASHES against main at 5984583


@pytest.mark.parametrize("name", list(TABLE))
def test_all_five_features_validate_on_every_planet(name: str) -> None:
    """The source run of a planet has the features on (build, bonds, comm, niche, arms): the combination the
    launcher's source runs use validates on the three presets, with and without the weather of its size."""
    base = ["arms.enabled=true"]
    feats = ["build", "bonds", "comm", "niche3d"]
    cfg = load_config(preset=name, overlays=feats, overrides=base)
    assert cfg.structures.enabled and cfg.comm.enabled and cfg.arms.enabled and cfg.niche.farm
    assert cfg.bonds.share_mode != "off" or cfg.roles.enabled
    overlay = {"tiny3s": "stormsph16", "mini3s": "stormsph22", "64k3s": "stormsph"}.get(name)
    if overlay:
        both = load_config(preset=name, overlays=[*feats, overlay], overrides=base)
        assert both.weather.enabled and both.structures.enabled and both.arms.enabled


# --- the weather overlays ---------------------------------------------------------------------------------

OVERLAY_OF = {"64k3s": "stormsph", "mini3s": "stormsph22", "tiny3s": "stormsph16"}
# The calibrated weather of each planet (DESIGN §46.6, S4c), pinned so that a change is deliberate: the pools
# (per hemisphere, sx, sy, amplitude, at 40 degrees, the angular drift of `stormsph`) and the couplings. The
# storm statistics they give are the slow gates of `test_sphere_presets_slow.py`.
CALIBRATED = {
    "stormsph": dict(
        pools=(3, 40.0, 20.0, 1.0),
        seed_rate=0.05,
        rain_gain=52000.0,
        storm_p=0.002,
        storm_u0=0.09,
        storm_span=0.03,
    ),
    "stormsph22": dict(
        pools=(3, 24.0, 16.0, 1.0),
        seed_rate=0.04,
        rain_gain=26000.0,
        storm_p=0.002,
        storm_u0=0.095,
        storm_span=0.03,
    ),
    "stormsph16": dict(
        pools=(3, 18.0, 12.0, 1.0),
        seed_rate=0.0032,
        rain_gain=4400.0,
        storm_p=0.002,
        storm_u0=0.06,
        storm_span=0.03,
    ),
}


@pytest.mark.parametrize(("name", "overlay"), list(OVERLAY_OF.items()))
def test_the_overlay_validates_on_its_planet_with_the_courant_rule(name: str, overlay: str) -> None:
    assert overlay in config.WORLD_OVERLAYS and overlay in config.ALL_OVERLAYS
    cfg = load_config(preset=name, overlays=[overlay])
    nf = dims(cfg).nside
    p = weather_sph.params_from_config(cfg)
    assert cfg.weather.enabled and cfg.weather.cells == 2 * nf == p.n
    assert len(cfg.weather.pools) == 6 * 7  # six pools: three per hemisphere
    r = cfg.world.radius
    assert weather_sph.courant(p, r) <= 0.8 * weather_sph.C_LIMIT
    # before and after the derived weather time step (C29): dt = 1 (C_used 0.50) and weather_dt (C_used 0.40)
    assert weather_sph.courant(p, r, dt=1.0) <= 0.8 * weather_sph.C_LIMIT
    dt = weather_sph.weather_dt(p)
    assert 0.75 < dt < 0.85 and p.dt == dt and weather_sph.courant(p, r, dt=dt) == pytest.approx(0.40)
    # f at the pools is 0.0162 in every planet (latitude +-40 degrees, f0 0.025), as 64k3d's
    assert cfg.weather.f0 == 0.025
    # the overlay is a function of the planet alone: it fixes no weather time step, so the Courant number is
    # the rule's own at any dt (S4b's smaller weather dt lowers it, and nothing here depends on it)
    text = tomllib.loads((PRESET_DIR / f"{overlay}.toml").read_text(encoding="utf-8"))
    assert set(text) == {"weather"} and "dt" not in text["weather"]
    assert text["weather"]["pools"] == list(cfg.weather.pools)


@pytest.mark.parametrize(("name", "overlay"), list(OVERLAY_OF.items()))
def test_the_calibrated_pools_and_couplings_of_each_overlay_are_pinned(name: str, overlay: str) -> None:
    """S4c (review M2 of S7a, for all three planets): the pools are the builder's (`w0_sphere.make_pools`: per
    hemisphere, sx, sy, amplitude 1, the angular drift of `stormsph`) at the sizes the study chose, and the
    couplings are the values it set. A change of one is deliberate: the storm statistics of DESIGN §46.6 and
    the slow gates of `test_sphere_presets_slow.py` are of these."""
    from alife.analysis.w0_sphere import make_pools

    cal = CALIBRATED[overlay]
    cfg = load_config(preset=name, overlays=[overlay])
    per, sx, sy, amp = cal["pools"]
    assert cfg.weather.pools == pytest.approx(make_pools(cfg.world.radius, per, sx, sy, amp=amp), abs=1e-5)
    for key in ("seed_rate", "rain_gain", "storm_p", "storm_u0", "storm_span"):
        assert getattr(cfg.weather, key) == cal[key], key
    # what the study left alone is the box's: the seeds' size, the moisture, the damping (S4b M1)
    ref = load_config(preset="64k3d", overlays=["storm"]).weather
    for key in (
        "seed_sigma",
        "seed_v",
        "q_pool",
        "q_bg",
        "tau_rad",
        "r1",
        "r2",
        "c_d",
        "gamma",
        "tau_c",
        "k_E",
    ):
        assert getattr(cfg.weather, key) == getattr(ref, key), key


@pytest.mark.parametrize(("name", "overlay"), list(OVERLAY_OF.items()))
def test_the_pools_sit_at_forty_degrees_three_per_hemisphere_and_fit_the_planet(
    name: str, overlay: str
) -> None:
    import math

    cfg = load_config(preset=name, overlays=[overlay])
    r = cfg.world.radius
    pools = [cfg.weather.pools[i : i + 7] for i in range(0, len(cfg.weather.pools), 7)]
    lats = sorted(round(math.degrees(p[1] / r), 3) for p in pools)
    assert lats == [-40.0] * 3 + [40.0] * 3
    for x, _y, sx, sy, amp, _vx, vy in pools:
        assert 0.0 <= x < 2 * math.pi * r and vy == 0.0 and amp == 1.0
        assert sx < 2 * math.pi * r * math.cos(math.radians(40.0)) / 3 and sy < math.pi * r / 4
    assert {p[1] > 0 for p in pools} == {True, False}
    assert len({round(math.degrees(p[0] / r) % 120, 3) for p in pools if p[1] > 0}) == 1  # 120 degrees apart


@pytest.mark.parametrize("overlay", ["stormsph", "stormsph22", "stormsph16"])
@pytest.mark.parametrize("name", list(TABLE))
def test_an_overlay_is_refused_on_a_planet_it_is_not_made_for(name: str, overlay: str) -> None:
    if OVERLAY_OF.get(name) == overlay:
        pytest.skip("the planet the overlay is made for")
    with pytest.raises(ConfigError, match=r"Courant|latitude|arc length|power of two"):
        load_config(preset=name, overlays=[overlay])

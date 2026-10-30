# Stream protocol v2 (the sphere world): the format text

Status: **normative** (DESIGN §46.7 points here). It was written before the decoders; the independent decoder
`tests/sphere_decode.py` was written from it without reading the encoder, and `tests/test_sphere_stream.py` checks
the encoders against that decoder. A client (the Swift app, S6) needs this file and the protocol 1 text (DESIGN
§33.1) and nothing else. Everything not stated here
is exactly the protocol 1 of `alife play` (DESIGN §33.1, §34.6, the feature extensions of the observer
contract: little endian; a record is `tag(4 ASCII) u32 payload_len payload`; the payload starts with the 16-byte
head `u64 step, u32 seq, u16 version, u16 flags`; every payload length is a multiple of 4; unknown tags are
skipped by length). A **box stream is byte for byte what it was**: nothing below applies to it.

## 1. When v2 applies

A world whose config has `world.geometry = "sphere"` writes **protocol 2**. The first bytes are the same
(`MAGIC = "ALSTRM" 00 01`, `u32 json_len`, the JSON space-padded to a multiple of 4), but the header key
`"protocol"` is `2`. A reader that only knows protocol 1 must refuse it. `alife stream` (the legacy frame stream,
`FRM1` frames) writes the same header; its frames are unchanged (section 3).

## 2. The header (JSON object)

Keys of protocol 1 that **change**:

| key | v2 |
|---|---|
| `protocol` | `2` |
| `L` | **absent** (a sphere has no image length) |
| `FG` | `0` (no lattice side) |
| `FZ` | the number of radial layers (`world.layers`) |
| `height` | the shell thickness `H` (float) |
| `radius` | still the **body radius** of an agent (not the planet's) |
| `dims` | `3` |

Keys that are **new** (present on every v2 header, after `FZ` in the writer's key order, which is not
significant):

| key | type | meaning |
|---|---|---|
| `geometry` | `"sphere"` | |
| `planet_radius` | float > 0 | the planet radius `Rp`; the floor is the sphere of radius `Rp` |
| `shell_edges` | list of `FZ + 1` floats | ascending layer boundary radii from `Rp` to `Rp + H` (equal-volume layers) |
| `nside_field` | int >= 2 | HEALPix Nside of the field cells (`12·Nside²` cells per layer) |
| `nside_weather` | int >= 0 | the weather Nside, 0 when the world has no weather |
| `nside_territory` | int >= 2 | HEALPix Nside of the territory map (`TERR`) |
| `ring_order` | `"ring"` | every cell index below is the HEALPix **RING** pixel index |
| `f0` | float | the reference Coriolis parameter (0.025) |
| `weather_depth` | float >= 0 | `H1 + H2`, the model atmosphere depth the shell is mapped onto; 0 without weather |

The `play` object of the header is as in protocol 1 except: `play.records` may list `MAPH` (never `MAP1`);
`play.weather` (when the weather is on) is `{"enabled": true, "nside": Nw, "coarse": Nc, "f0", "name_zeta",
"refs": {...}}` with `Nc = nside_field` (see `MAPH`); `play.structures.cells` and `play.niche.cells` are
`[FZ, 1, Npix]` with `Npix = 12·nside_field²` (section 5); `play.labels.intent_version` is `4` (the sphere's
rule set); `play.versions` has `"MAPH": 1` when `MAPH` is listed.

## 3. Positions and orientation (`FRMA`, `FOCS`, `FRM1`)

The layout of `FRMA` is unchanged. A position is three float32 `x, y, z` in the **planet frame** (origin at the
planet's centre, `|p|` between `Rp` and `Rp + H`). The "up" at an agent is `û = p / |p|`, its **height** is
`|p| − Rp`.

The orientation pair `(yaw, pitch)` (float16 each) keeps its place. For a sphere:

* `east = normalise(ẑ × û)` (`ẑ = (0, 0, 1)`), and `east = (1, 0, 0)` when `|ẑ × û| < 1e-6` (at a pole);
  `north = û × east`.
* `yaw = ψ` is the **bearing from local north**, in [−π, π): `ψ = atan2(t · east, t · north)` where `t` is the
  tangent (horizontal) part of the heading, normalised.
* `pitch = θ` is the elevation of the heading above the tangent plane, as in a 3D box world.
* A client rebuilds the heading: `t = cos ψ · north + sin ψ · east`, then `forward = cos θ · t + sin θ · û`.
* Both are IEEE half floats: `yaw` in [−π, π) (resolution about 2e-3 rad near ±π), `pitch` in [−π/2, π/2]. The
  rebuilt heading is exact to the half-float rounding (0.06° at worst, measured over the whole sphere).
* **Yaw is ill-conditioned within about 1e-3 rad of a pole.** `east` follows `(−y, x)/|…|` and flips to `x̂` below
  the 1e-6 threshold, so a heading that is constant in space has a yaw that turns by up to π across the polar
  patch. A client that interpolates between two frames must interpolate the **rebuilt tangent vector** `t` (and
  renormalise), never the angle.

`FOCS` positions and velocities are Cartesian planet-frame vectors. A velocity's radial part is `v · û`.
The frames of `alife stream` (`FRM1`, DESIGN §27.11) carry the same position and orientation convention;
their optional field section holds `Npix = 12·nside_field²` `u32` colours (one layer, ring order) instead of
`FG²`.

## 4. `MAPH` (replaces `MAP1`)

Sent where `MAP1` would be (weather worlds), tag `MAPH`, head `version = 1`, head `flags = 3` (bit 0: the
`T_s` plane is present, bit 1: the winds are present; **both are always set**, as in every `MAP1`). Body (all
little endian):

| offset | type | field |
|---|---|---|
| 0 | u16 | `nside` of the planes (`Nc`, equal to `header.nside_field`) |
| 2 | u16 | `k` with `nside_weather = Nc · 2^k` (informational) |
| 4 | f32 | `f0` |
| 8 | f32 | `p_ref` |
| 12 | f32 | `dc_ref` |
| 16 | f32 | `u_ref` |
| 20 | f32 | `zeta_ref` |
| 24 | 8 bytes | reserved, zero |
| 32 | `npix` u8 | `P` |
| | `npix` u8 | `Dc` |
| | `npix` i8 | `zeta` |
| | `npix` u8 | `Ts` |
| | `4·npix` i8 | winds, plane-major: `u1` (all pixels), `v1`, `u2`, `v2` |
| | 0 to 3 | zero padding to a multiple of 4 |

`npix = 12·nside²`; index `p` of a plane is the RING pixel `p`. `u` and `v` are the **eastward and northward**
components of the layer's wind at the pixel centre (`east` and `north` as in section 3; the centre of pixel `p`
is the HEALPix centre). Body length = `32 + 8·npix` rounded up to a multiple of 4 (the payload adds the 16-byte
head). Reference scales: `p_ref = 0.004`, `u_ref = 0.1`, `zeta_ref = f0 = 0.025` (as `play.weather.refs`);
`dc_ref` is the world's cloud-depth scale (`play.weather.refs.dc`).

**Quantisation** (the encoder's, inherited from `MAP1`, DESIGN §33.1.7; `c(x) = clip(x, 0, 1)`), and the decode
each client applies:

| plane | encode | decode |
|---|---|---|
| `P` rain rate | `round(255·sqrt(c(P/p_ref)))` | `(b/255)²·p_ref` |
| `Dc` cloud depth | `round(255·c(Dc/dc_ref))` | `b/255·dc_ref` |
| `zeta` layer-1 vorticity | `round(127·clip(ζ/zeta_ref, −1, 1))` | `b/127·zeta_ref` |
| `Ts` sea temperature | `round(255·c(Ts))` | `b/255` |
| winds `u1 v1 u2 v2` | `round(127·clip(w/u_ref, −1, 1))` | `b/127·u_ref` |

Every plane is the **mean over the weather cells** (`12·Nw²`, `Nw = Nc·2^k`) that are the children of the coarse
cell (the fine pixels whose centres lie in it); the wind is averaged as a 3-vector and then projected on the
east and north of the coarse cell's centre. `Ts` is the warm-pool map **at the tick's step**.

## 5. Sparse cell records (`STRC`, `FSOL`)

The layout of protocol 1 is unchanged: `u16 nx, u16 ny, u16 nz, u16 flags, u32 count`, `count` ascending `u32`
flat indices, `count` `u8` levels, zero padding. On a sphere `nx = Npix = 12·nside_field²`, `ny = 1`,
`nz = FZ`. The flat index of a cell is `layer · Npix + pixel` (layer 0 = the floor layer, pixel = RING index), which
is the z-major index `(z·ny + y)·nx + x` with `y = 0`. `Npix` must fit a `u16` (Nside <= 73): a world beyond it
does not send these records. The header block `play.structures` / `play.niche` carries
`"cells": [FZ, 1, Npix]`.

## 6. `TERR`

Body, little endian: `u16 nx, u16 ny, u32 snapshot_step, u32 json_len, u32 0` (16 bytes); then `json_len` bytes of
compact JSON `{"ids": [...]}` (kin ids by slot, at most 16 entries, `−1` for a slot that holds none) followed by
0 to 3 zero bytes to a multiple of 4; then `owner` (`nx·ny` bytes) and `purity` (`nx·ny` bytes), written back to
back and zero-padded together to a multiple of 4. This is protocol 1's layout; the box has `nx = ny = 32`
(row `y`, column `x`). On a sphere `ny = 1` and `nx = 12·nside_territory²`, and `owner[p]`, `purity[p]` belong to
the RING pixel `p` of the territory map. `owner[p]` is the index in `ids` of the kin group that owns the pixel, or
255 for none; `purity[p] = round(255·share)` where `share` is the owner's share of the pixel's smoothed count
(0 when unowned). The territory map is the same for every client: `nside_territory` is the observer's rules'
Nside (9 by default: 972 pixels).

## 7. Records that carry a place

`EVNT` events have `x, y, z` (numbers or null): on a sphere the planet-frame point (Cartesian), not
longitude/latitude. `ALRT` alerts have `x, y, z` likewise. `STRM` storms (weather worlds) have
`x, y, z` (the centre of the storm on the planet at the floor radius), `r` the radius as an arc length at the
floor, and no other positional keys; `ENVS.patches` entries are `{i, x, y, z, lon, lat, gain, sigma}` with
`x, y, z` the unit vector of the patch centre and `lon, lat` in degrees; `ENVS.pools` is `[]` on a sphere
without weather (the key stays, as in a box) and, with weather, a list of `{x, y, z, lon, lat, amp, sx, sy, vx, vy}`
(the pool centre at step `t` as the planet-frame point at the floor radius and in degrees; `sx, sy` widths and
`vx` the drift, arc lengths). A `STRM` storm's `track` entries are `[t, x, y, z]` and its `pred` entries
`[lead, x, y, z]`; the tracker events (`EVNT` and `storms.jsonl`) carry `x, y, z`. The JSON tables of `IDNT` (`kin`, `colonies`) have `x, y, z` the planet-frame
centroid and, for a kin group, `height` (mean height); nothing in them is a longitude.

Places given **to** the engine by control lines (`power ... x=.. y=..`) are `x` = longitude and `y` = latitude in
degrees, `x` in [−180, 180) and `y` in [−90, 90]; a length (a strike's `radius`) is an arc length at the floor.

## 8. What a decoder checks

Everything protocol 1 checks (alignment, padding, lengths, counts against `N`, slot bytes against their tables,
that the records account for every byte), and for v2: the header keys above (types, `FG == 0`, no `L`,
`len(shell_edges) == FZ + 1`, ascending, first edge `planet_radius`, last edge `planet_radius + height`);
every `FRMA` position has `planet_radius <= |p| <= planet_radius + height` (up to float32 rounding, 1e-3);
`STRC`/`FSOL`/`TERR`/`MAPH` dimensions match the header; levels in 1..255; indices strictly ascending and below
`FZ·Npix`; `MAPH` flags and padding. A sparse cell record may list at most 65536 cells (the
truncation rule of protocol 1: more non-empty cells keep the highest levels and set flag bit 0).

## 9. Records not listed here

`IDNT`, `INTN`, `CLCK`, `FOCS` (and the `FSIG` voice record), `RPRT` and `PLAN` have the protocol 1 layouts; the
places in `FOCS` are Cartesian (section 3) and in `PLAN` cards and `RPRT` facts a lever's place is `(lon, lat)`
degrees (section 7). A world report's facts and entities state a place as `x, y, z` (the planet-frame point; a
reader derives lon and lat from it); only a lever fact's `value.x, value.y` are `(lon, lat)` degrees, as in the lever
form. Unknown tags are skipped by length.

// The globe's part of the viewer's shaders (DESIGN 36.10): the morph of the plane into the sphere in the vertex shaders (the arithmetic of
// `Morph` in AlifeCore, line for line), the map's mesh, the pole caps, the space behind them, and the one-pixel pass that resets the depth.
// Two strings, spliced into `Shaders.source`: `globeCommon` (the struct and the functions the agents', bonds' and points' vertex shaders
// use too) and `globeEntries` (the entry points, which need `mapColorAt`). Do not use a backslash or a triple quote.
extension Shaders {
    static let globeMapVertex = "globeMapVertex"
    static let globeMapFragment = "globeMapFragment"
    static let globeCapVertex = "globeCapVertex"
    static let globeCapFragment = "globeCapFragment"
    static let globeSkyFragment = "globeSkyFragment"
    static let depthResetFragment = "depthResetFragment"
    static let globeEntryPoints = [
        globeMapVertex, globeMapFragment, globeCapVertex, globeCapFragment, globeSkyFragment, depthResetFragment,
    ]

    /// The mesh of the map: this many cells on a side over one world (`L` wide, `L` tall), so 321 x 321 vertices.
    static let meshCells = 320

    static let globeCommon = """
    // The globe's constants (buffer 8 of the world pass; `GlobeUniforms` in GlobeUniforms.swift mirrors it, 160 bytes). Zero on the flat
    // path (the flag `shade.w` is 0 there and no shader reads the rest).
    struct GlobeU {
        float4 a;        // x: s, y: L, z: H, w: R (the sphere's radius)
        float4 b;        // x: the plane's scale M lambda(s), y: zeta(s) (the height scale), z: w2(s) (the Mercator tail), w: kz
        float4 c;        // x: the centre's column x, y: its signed row ybar, z: cos(heading), w: sin(heading)
        float4 right;    // xyz: the centre frame's east axis (world-fixed), w: the centre's latitude
        float4 up;       // xyz: its north axis
        float4 centre;   // xyz: its outward axis (the centre's direction)
        float4 copies;   // x, y, z: the alpha of the copies of the map across the wrap in y (rows -1, 0, 1); w: 1 while rows may use them
        float4 look;     // x: the weight of space against the haze sky, y: the fog's strength, z: the stars' and the atmosphere's weight, w: the graticule's
        float4 shade;    // x: the weight of the lighting, y: of the rim, z: of the pole caps, w: 1 on the globe path
        float4 light;    // xyz: the key light, towards the light, in the camera's world; w: the weight of the haze at the limb
        float4 glow;     // x: the weight of the organism glow, y: of the crowd's floor
    };

    // The plane at the morph's scale rolled onto a sphere of radius R / s, tangent at the centre (the plane itself at s = 0); the height
    // along the rolled normal, scaled by zeta. (Morph.rollPosition.)
    float3 globeRoll(float e, float n, float z, constant GlobeU& g)
    {
        float s = g.a.x;
        float R = g.a.w;
        float ux = g.b.x * (e * g.c.z + n * g.c.w);
        float uy = g.b.x * (-e * g.c.w + n * g.c.z);
        float lift = g.b.y * z;
        if (s <= 0.000001) { return float3(ux, uy, R + lift); }
        float rho = R / s;
        float r = length(float2(ux, uy));
        float theta = min(r / rho, 6.2831853 - 0.2);
        float k = r > 0.000000001 ? sin(theta) / r : 1.0 / rho;
        float3 nhat = float3(k * ux, k * uy, cos(theta));
        float versine = 2.0 * sin(0.5 * theta) * sin(0.5 * theta);
        return float3(rho * k * ux, rho * k * uy, R - rho * versine) + lift * nhat;
    }

    float3 globeRollNormal(float e, float n, constant GlobeU& g)
    {
        float s = g.a.x;
        if (s <= 0.000001) { return float3(0.0, 0.0, 1.0); }
        float ux = g.b.x * (e * g.c.z + n * g.c.w);
        float uy = g.b.x * (-e * g.c.w + n * g.c.z);
        float rho = g.a.w / s;
        float r = length(float2(ux, uy));
        float theta = min(r / rho, 6.2831853 - 0.2);
        float k = r > 0.000000001 ? sin(theta) / r : 1.0 / rho;
        return float3(k * ux, k * uy, cos(theta));
    }

    // The unit direction of the Mercator sphere at the offsets (e, n) from the centre, in the centre's frame: longitude 2 pi x / L, latitude
    // gd(2 pi ybar / L) (sin = tanh, cos = 1 / cosh); n is the unwrapped row offset.
    float3 globeDirection(float e, float n, constant GlobeU& g)
    {
        float L = g.a.y;
        float lam = 6.2831853 * (g.c.x + e) / L;
        float u = 6.2831853 * (g.c.y + n) / L;
        float cl = 1.0 / cosh(u);
        float3 q = float3(cl * cos(lam), cl * sin(lam), tanh(u));
        return float3(dot(q, g.right.xyz), dot(q, g.up.xyz), dot(q, g.centre.xyz));
    }

    // The position of the point at offsets (e, n) and height z in the centre's frame: the roll, mixed into the Mercator sphere by w2.
    float3 globePosition(float e, float n, float z, constant GlobeU& g)
    {
        float3 roll = globeRoll(e, n, z, g);
        float w = g.b.z;
        if (w <= 0.0) { return roll; }
        return mix(roll, (g.a.w + g.b.w * z) * globeDirection(e, n, g), w);
    }

    // The outward unit normal (the lighting normal and the facing test).
    float3 globeNormal(float e, float n, constant GlobeU& g)
    {
        float3 roll = globeRollNormal(e, n, g);
        float w = g.b.z;
        if (w <= 0.0) { return roll; }
        return normalize(mix(roll, globeDirection(e, n, g), w));
    }

    // The offsets of a row's position: x: the east offset (the nearest column), y: the north offset (the nearest row too while the copies
    // exist), z: the alpha of the copy that draws it.
    float3 globeOffsets(float3 raw, constant GlobeU& g)
    {
        float L = g.a.y;
        float e = raw.x - g.c.x;
        e -= L * rint(e / L);
        float n = (raw.y - L * rint(raw.y / L)) - g.c.y;
        float alpha = g.copies.y;
        if (g.copies.w > 0.5) {
            float n0 = n;
            float best = abs(n0);
            if (abs(n0 - L) < best) { n = n0 - L; best = abs(n); alpha = g.copies.x; }
            if (abs(n0 + L) < best) { n = n0 + L; alpha = g.copies.z; }
        }
        return float3(e, n, alpha);
    }

    // Where a row is on the globe: its centre, the surface normal under it, the tangent axes (east, north) of the drawing, the sphere
    // units per torus unit, the copy's alpha, and the foot (at height 0).
    struct GlobePlace {
        float3 p;
        float3 foot;
        float3 nrm;
        float3 east;
        float3 north;
        float scale;
        float alpha;
    };

    GlobePlace globePlace(float3 raw, constant GlobeU& g)
    {
        GlobePlace out;
        float3 o = globeOffsets(raw, g);
        out.alpha = o.z;
        out.p = globePosition(o.x, o.y, raw.z, g);
        out.foot = globePosition(o.x, o.y, 0.0, g);
        out.nrm = globeNormal(o.x, o.y, g);
        const float h = 0.5;
        float3 pe = globePosition(o.x + h, o.y, 0.0, g) - globePosition(o.x - h, o.y, 0.0, g);
        out.scale = max(length(pe) / (2.0 * h), 0.000001);
        float3 east = pe - out.nrm * dot(pe, out.nrm);
        float el = length(east);
        out.east = el > 0.000001 ? east / el : float3(1.0, 0.0, 0.0);
        out.north = cross(out.nrm, out.east);
        return out;
    }
    """

    static let globeEntries = """
    // ---- The globe: the map's mesh, the pole caps, the sky, the depth reset ----

    struct GlobeMapVarying {
        float4 position [[position]];
        float2 uv;
        float2 off;        // the offsets (e, n) from the centre, the copy's included: the haze is a function of them
        float3 normal;
        float3 world;
        float alpha;       // the copy's
    };

    // The mesh: (meshCells + 1)^2 vertices over one world, a triangle list from a shared index buffer; one instance per tile, a tile being the
    // mesh shifted by (ix L, iy L) (buffer 9: ix, iy, alpha). The tiles beside the central one exist while s is under 0.3: the flat close-up,
    // tilted, sees the world beyond the one it is centred on (the same map again, hidden by the haze); the copies across the wrap in y are the
    // true neighbours.
    vertex GlobeMapVarying globeMapVertex(uint vid [[vertex_id]],
                                          uint iid [[instance_id]],
                                          constant Scene& scene [[buffer(0)]],
                                          constant WorldU& world [[buffer(7)]],
                                          constant GlobeU& g [[buffer(8)]],
                                          constant float4* tiles [[buffer(9)]])
    {
        const uint cells = \(meshCells)u;
        float4 tile = tiles[iid];
        float L = g.a.y;
        float fu = float(vid % (cells + 1u)) / float(cells);
        float fv = float(vid / (cells + 1u)) / float(cells);
        float e = (fu - 0.5) * L + tile.x * L;
        float yb = (fv - 0.5) * L;
        float n = yb + tile.y * L - g.c.y;
        GlobeMapVarying out;
        float3 p = globePosition(e, n, 0.0, g);
        out.position = scene.viewProjection * float4(p, 1.0);
        out.uv = float2((g.c.x + e) / L, yb / L);
        out.off = float2(e, n);
        out.normal = globeNormal(e, n, g);
        out.world = p;
        out.alpha = tile.z;
        return out;
    }

    fragment float4 globeMapFragment(GlobeMapVarying in [[stage_in]],
                                     constant Scene& scene [[buffer(0)]],
                                     constant WorldU& world [[buffer(7)]],
                                     constant GlobeU& g [[buffer(8)]],
                                     texture2d<float> rainCur [[texture(0)]],
                                     texture2d<float> cloudCur [[texture(1)]],
                                     texture2d<float> zetaCur [[texture(2)]],
                                     texture2d<float> seaCur [[texture(3)]],
                                     texture2d<float> rainPrev [[texture(4)]],
                                     texture2d<float> cloudPrev [[texture(5)]],
                                     texture2d<float> zetaPrev [[texture(6)]],
                                     texture2d<float> seaPrev [[texture(7)]],
                                     texture2d<float, access::read> territory [[texture(8)]],
                                     texture2d<float> glowTexture [[texture(9)]])
    {
        const float3 background = float3(0.035, 0.04, 0.055);
        float gs = g.look.w;
        // The territory and the graticule fade into the cap over the last rows (latitudes 78 to 85 degrees).
        float poleFade = mix(1.0, 1.0 - smoothstep(0.43, 0.5, abs(in.uv.y)), gs);
        float3 color = mapColorAt(in.uv, world, rainCur, cloudCur, zetaCur, seaCur, rainPrev, cloudPrev, zetaPrev, seaPrev,
                                  territory, poleFade, 1.0 - gs);
        if (gs > 0.0) {
            // The graticule every 30 degrees, the equator gold, the seam rows (the poles) magenta; anti-aliased with fwidth.
            float lonU = in.uv.x * 12.0;
            float lonLine = 1.0 - smoothstep(0.5, 1.5, abs(fract(lonU + 0.5) - 0.5) / max(fwidth(lonU), 1e-6));
            float lat = asin(clamp(tanh(6.2831853 * in.uv.y), -1.0, 1.0)) * (57.29578 / 30.0);
            float latLine = 1.0 - smoothstep(0.5, 1.5, abs(fract(lat + 0.5) - 0.5) / max(fwidth(lat), 1e-6));
            float equator = 1.0 - smoothstep(0.5, 1.5, abs(lat) / max(fwidth(lat), 1e-6));
            float seam = 1.0 - smoothstep(0.5, 1.5, (0.5 - abs(in.uv.y)) / max(fwidth(in.uv.y), 1e-6));
            color = mix(color, float3(1.0), 0.14 * gs * poleFade * max(lonLine, latLine));
            color = mix(color, float3(0.95, 0.78, 0.35), 0.55 * gs * equator);
            color = mix(color, float3(0.90, 0.30, 0.80), 0.50 * gs * seam);
        }
        float3 nrm = normalize(in.normal);
        float3 toEye = normalize(scene.eye.xyz - in.world);
        float nv = clamp(dot(nrm, toEye), 0.0, 1.0);
        color *= mix(1.0, 0.66 + 0.46 * max(dot(nrm, g.light.xyz), 0.0), g.shade.x);
        // The organism glow: where the instances have faded out, their density as warm light (added after the lighting: it is a light).
        if (g.glow.x > 0.0 || g.glow.y > 0.0) {
            float cell = glowTexture.sample(planeSampler, in.uv).r;
            // a cell of 4 rows or more (byte 0.568) keeps a floor of 0.25 whatever the instances' opacity: a crowd that overlaps reads as light
            float weight = max(g.glow.x, 0.25 * g.glow.y * smoothstep(0.50, 0.60, cell));
            color += weight * smoothstep(0.05, 0.9, cell) * float3(1.0, 0.86, 0.55);
        }
        float3 shown = mix(background, color, world.params.y);
        float2 off = abs(in.off);
        float x = max(max(off.x, off.y) - world.fogShape.x, 0.0) * world.fogShape.y;
        // The haze: the flat close-up's (a function of the distance from the centre) and, from the first s on, the limb's: the rolled plane
        // is a ball of radius R / s, whose edge against the sky must be haze to be no edge (the sky below the flat horizon is the haze).
        float fogAmount = max(1.0 - exp(-x * x), g.light.w * (1.0 - smoothstep(0.0, 0.2, nv)));
        shown = mix(shown, world.fog.rgb, fogAmount * g.look.y);
        shown += g.shade.y * 0.55 * pow(1.0 - nv, 3.0) * float3(0.28, 0.52, 1.0);
        return float4(clamp(shown, 0.0, 1.0), in.alpha);
    }

    // The two caps that close the poles: a fan of 64 triangles over the mesh's edge row (ybar = +-L/2, the same row at both poles), its apex
    // the mean of the ring (on the sphere's axis once the tail is in). Instance 0 is the north pole, 1 the south.
    struct GlobeCapVarying {
        float4 position [[position]];
        float rim;
        float3 normal;
        float3 world;
    };

    vertex GlobeCapVarying globeCapVertex(uint vid [[vertex_id]],
                                          uint iid [[instance_id]],
                                          constant Scene& scene [[buffer(0)]],
                                          constant GlobeU& g [[buffer(8)]])
    {
        GlobeCapVarying out;
        out.position = float4(2.0, 2.0, 2.0, 1.0);
        out.rim = 0.0;
        out.normal = float3(0.0, 0.0, 1.0);
        out.world = float3(0.0);
        if (g.shade.z <= 0.0 || iid > 1u) { return out; }
        const uint segments = 64u;
        float L = g.a.y;
        float sgn = iid == 0u ? 1.0 : -1.0;
        float n = sgn * 0.5 * L - g.c.y;
        uint tri = vid / 3u;
        uint corner = vid % 3u;
        float3 p;
        if (corner == 0u) {
            float3 sum = float3(0.0);
            for (uint m = 0u; m < 16u; ++m) { sum += globePosition((float(m) + 0.5) / 16.0 * L - 0.5 * L, n, 0.0, g); }
            float3 mean = sum / 16.0;
            p = mix(mean, normalize(mean) * g.a.w, g.b.z);
        } else {
            uint m = ((corner == 1u) == (sgn > 0.0)) ? tri : tri + 1u;
            p = globePosition((float(m) / float(segments) - 0.5) * L, n, 0.0, g);
            out.rim = 1.0;
        }
        out.position = scene.viewProjection * float4(p, 1.0);
        out.normal = normalize(sgn * float3(g.right.z, g.up.z, g.centre.z));
        out.world = p;
        return out;
    }

    // A flat colour: the mean of the map along the seam row, half way to a neutral, lit like the map, with a thin magenta ring at the rim.
    fragment float4 globeCapFragment(GlobeCapVarying in [[stage_in]],
                                     constant Scene& scene [[buffer(0)]],
                                     constant WorldU& world [[buffer(7)]],
                                     constant GlobeU& g [[buffer(8)]],
                                     texture2d<float> rainCur [[texture(0)]],
                                     texture2d<float> cloudCur [[texture(1)]],
                                     texture2d<float> zetaCur [[texture(2)]],
                                     texture2d<float> seaCur [[texture(3)]],
                                     texture2d<float> rainPrev [[texture(4)]],
                                     texture2d<float> cloudPrev [[texture(5)]],
                                     texture2d<float> zetaPrev [[texture(6)]],
                                     texture2d<float> seaPrev [[texture(7)]],
                                     texture2d<float, access::read> territory [[texture(8)]])
    {
        const float3 background = float3(0.035, 0.04, 0.055);
        float3 sum = float3(0.0);
        for (int m = 0; m < 8; ++m) {
            sum += mapColorAt(float2((float(m) + 0.5) / 8.0, 0.5), world, rainCur, cloudCur, zetaCur, seaCur, rainPrev, cloudPrev,
                              zetaPrev, seaPrev, territory, 0.0, 0.0);
        }
        float3 color = mix(sum / 8.0, float3(0.62, 0.70, 0.80), 0.5);
        float3 nrm = normalize(in.normal);
        float3 toEye = normalize(scene.eye.xyz - in.world);
        float nv = clamp(dot(nrm, toEye), 0.0, 1.0);
        color *= mix(1.0, 0.66 + 0.46 * max(dot(nrm, g.light.xyz), 0.0), g.shade.x);
        float w = max(fwidth(in.rim), 0.0001);
        float ring = smoothstep(1.0 - 2.5 * w, 1.0 - 1.0 * w, in.rim);
        color = mix(color, float3(0.90, 0.30, 0.80), 0.5 * ring);
        float3 shown = mix(background, color, world.params.y);
        shown += g.shade.y * 0.55 * pow(1.0 - nv, 3.0) * float3(0.28, 0.52, 1.0);
        return float4(clamp(shown, 0.0, 1.0), g.shade.z);
    }

    float hash21(float2 p)
    {
        float3 p3 = fract(float3(p.xyx) * 0.1031);
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.x + p3.y) * p3.z);
    }

    // The sky of the globe path: the haze sky of the flat close-up (the horizon of the tangent plane at the centre) and space, one mixed into
    // the other with s. Space is a vignette, stars (a hash of the 3-pixel cell, 0.35% of the cells) and the atmosphere's glow outside the limb
    // from the ray's closest approach to the sphere's centre. Deterministic, no animation.
    fragment float4 globeSkyFragment(SkyVarying in [[stage_in]],
                                     constant Scene& scene [[buffer(0)]],
                                     constant WorldU& world [[buffer(7)]],
                                     constant GlobeU& g [[buffer(8)]])
    {
        const float3 background = float3(0.035, 0.04, 0.055);
        float aspect = scene.viewport.x / max(scene.viewport.y, 1.0);
        float tanHalf = 1.0 / world.refs.w;
        float3 forward = cross(scene.up.xyz, scene.right.xyz);
        float3 ray = normalize(forward + scene.right.xyz * (in.ndc.x * tanHalf * aspect) + scene.up.xyz * (in.ndc.y * tanHalf));
        float elevation = asin(clamp(ray.z, -1.0, 1.0));
        float haze = elevation <= 0.0 ? 1.0 : exp(-elevation / max(world.fogShape.w, 1e-3));
        float3 flatSky = mix(background, world.fog.rgb, haze);
        float3 space = mix(float3(0.040, 0.055, 0.110), float3(0.012, 0.018, 0.045), clamp(0.7071 * length(in.ndc), 0.0, 1.0));
        float2 cell = floor(in.position.xy / 3.0);
        float h = hash21(cell);
        float star = h < 0.0035 ? 0.35 + 0.65 * hash21(cell + 17.0) : 0.0;
        float R = g.a.w;
        float3 o = scene.eye.xyz;
        float t = -dot(o, ray);
        float b = t > 0.0 ? length(o + t * ray) : length(o);
        float d = (b - R) / R;
        float3 glow = b > R ? 1.3 * float3(0.18, 0.40, 0.95) * (0.42 * exp(-d / 0.07) + 0.10 * exp(-d / 0.45)) : float3(0.0);
        space += g.look.z * (star * float3(1.0) + glow);
        return float4(mix(flatSky, space, g.look.x), 1.0);
    }

    // The depth reset between the map and what stands on it: no colour, depth 1.
    fragment float4 depthResetFragment(SkyVarying in [[stage_in]])
    {
        return float4(0.0);
    }
    """
}

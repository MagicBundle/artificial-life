import AlifeCore

// The sphere world's Metal shaders (DESIGN §46.8), one string compiled at run time like `Shaders` (see its header for why not a `.metal`
// file). Nothing here is shared with the box world's shaders: the sphere has its own library, so the box path is untouched.
//
// Do not use a backslash or a triple quote in the source.
//
// How the planet is drawn. The floor (radius Rp) and the lid (radius Rp + H) are not meshes: each is a full-screen pass that casts the
// pixel's ray against the sphere, which gives an exact silhouette and an exact horizon at any zoom, no facets and no seam. (The spec's
// icosphere mesh was a means to the same end; a ray-cast sphere is its limit.) The colour of the hit is looked up by the pixel of the
// HEALPix cell under it, `healpix_ring(n, nside)`, the closed form of the engine's `ang2pix` (`Healpix.pixelFloat` is its line by line
// Swift mirror, tested against the engine's tables), from a per-cell buffer, so an overlay is seamless at every zoom with no texture and no
// cube map. The depth of the hit is written, so organisms, lines and the translucent lid sort against the planet exactly.
//
// The structs `SphereU` and `Seg` mirror `SphereUniforms` and `SphereSegment` in SphereRenderer.swift field by field (a test pins 272 and
// 64 bytes).
enum ShadersSphere {
    static let skyVertex = "sphereFullVertex"
    static let skyFragment = "sphereSkyFragment"
    static let floorFragment = "sphereFloorFragment"
    static let lidFragment = "sphereLidFragment"
    static let segmentVertex = "sphereSegmentVertex"
    static let segmentFragment = "sphereSegmentFragment"
    static let agentVertex = "sphereAgentVertex"
    static let agentFragment = "sphereAgentFragment"
    /// The test kernels: the HEALPix lookup and the heading of an agent, run over arrays so that the Swift mirrors can be compared.
    static let healpixKernel = "sphereHealpixKernel"
    static let headingKernel = "sphereHeadingKernel"

    static let entryPoints = [
        skyVertex, skyFragment, floorFragment, lidFragment, segmentVertex, segmentFragment, agentVertex, agentFragment,
        healpixKernel, headingKernel,
    ]

    /// The shared functions, kept apart so that a test can compile them alone.
    static let common = """
    #include <metal_stdlib>
    using namespace metal;

    struct SphereU {
        float4x4 viewProjection;
        float4 eye;        // xyz: the eye
        float4 right;      // xyz: the screen's +x in the world
        float4 up;         // xyz: the screen's +y
        float4 forward;    // xyz: the view axis
        float4 planet;     // x: Rp, y: the lid's radius, z: the agent radius, w: tan(fovY / 2)
        float4 viewport;   // x, y: the drawable in pixels, z: its aspect, w: the lid's cloud alpha scale
        int4 nsides;       // x: the field Nside, y: the territory Nside, z: agents listed, w: segments
        float4 look;       // x: exposure, y: territory tint alpha, z: territory border alpha, w: graticule alpha
        float4 style;      // x: agent smallest radius in px, y: agent largest radius in px, z: agent alpha, w: cell border alpha
        uint4 flags;       // x: bit 0 floor map, 1 cloud shell, 2 territory tint, 3 territory borders, 4 graticule, 5 height colour
                           // y: the selected field pixel (0xFFFFFFFF none), zw: reserved
        float4 sun;        // xyz: the direction to the sun, w: the ambient light
        float4 colours;    // x: the shell's height H, yzw: reserved
    };

    constant float kPi = 3.14159265358979;

    // HEALPix, RING scheme, any Nside: the closed form of the engine's ang2pix (geom.py), in float. v is a unit vector in the planet frame
    // (z north, x at longitude 0). Healpix.pixelFloat in the Swift is this function line by line.
    int healpix_ring(float3 v, int n)
    {
        float z = clamp(v.z, -1.0, 1.0);
        float phi = atan2(v.y, v.x);
        if (phi < 0.0) { phi += 2.0 * kPi; }
        float tt = phi * (2.0 / kPi);
        float za = fabs(z);
        float nn = float(n);
        int tot = 12 * n * n;
        int ncap = 2 * n * (n - 1);
        if (za <= 2.0 / 3.0) {
            float t1 = nn * (0.5 + tt);
            float t2 = nn * z * 0.75;
            int jp = int(floor(t1 - t2));
            int jm = int(floor(t1 + t2));
            int ir = n + 1 + jp - jm;
            int ks = 1 - (ir & 1);
            int m = 4 * n;
            int ip = (jp + jm - n + ks + 1) >> 1;
            ip = ip % m;
            if (ip < 0) { ip += m; }
            return ncap + (ir - 1) * 4 * n + ip;
        }
        float tp = tt - floor(tt);
        float tmp = nn * sqrt(3.0 * (1.0 - za));
        int jp = int(floor(tp * tmp));
        int jm = int(floor((1.0 - tp) * tmp));
        int ir = clamp(jp + jm + 1, 1, n);
        int m = 4 * ir;
        int ip = int(floor(tt * float(ir)));
        ip = ip % m;
        if (ip < 0) { ip += m; }
        return z > 0.0 ? 2 * ir * (ir - 1) + ip : tot - 2 * ir * (ir + 1) + ip;
    }

    // The heading of an agent from the transmitted bearing: east = normalise(z x up) ((1, 0, 0) at a pole), north = up x east,
    // t = cos(yaw) north + sin(yaw) east, forward = cos(pitch) t + sin(pitch) up (stream protocol 2, section 3).
    float3 agentHeading(float3 position, float yaw, float pitch)
    {
        float3 up = normalize(position);
        float3 ze = float3(-up.y, up.x, 0.0);
        float zn = length(ze);
        float3 east = zn < 1e-6 ? float3(1.0, 0.0, 0.0) : ze / zn;
        float3 north = cross(up, east);
        float3 t = cos(yaw) * north + sin(yaw) * east;
        return cos(pitch) * t + sin(pitch) * up;
    }

    float3 unpackColor(uint c)
    {
        return float3(float((c >> 16) & 255u), float((c >> 8) & 255u), float(c & 255u)) * (1.0 / 255.0);
    }

    // The healpix lookup over an array of unit vectors, for the test that compares it with the Swift mirror and the engine.
    kernel void sphereHealpixKernel(device const packed_float3* directions [[buffer(0)]],
                                    device int* pixels [[buffer(1)]],
                                    constant int& nside [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]])
    {
        pixels[gid] = healpix_ring(float3(directions[gid]), nside);
    }

    kernel void sphereHeadingKernel(device const packed_float3* positions [[buffer(0)]],
                                    device const float2* bearings [[buffer(1)]],
                                    device packed_float3* headings [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]])
    {
        headings[gid] = agentHeading(float3(positions[gid]), bearings[gid].x, bearings[gid].y);
    }
    """

    static let source = common + """


    float3 rayDirection(float2 ndc, constant SphereU& u)
    {
        float2 t = ndc * u.planet.w;
        return normalize(u.forward.xyz + u.right.xyz * (t.x * u.viewport.z) + u.up.xyz * t.y);
    }

    // The parameters of the entry and the exit of a ray through a sphere about the centre; (-1, -1) for a miss.
    float2 raySphere(float3 o, float3 d, float r)
    {
        float b = dot(o, d);
        float c = dot(o, o) - r * r;
        float disc = b * b - c;
        if (disc < 0.0) { return float2(-1.0); }
        float s = sqrt(disc);
        return float2(-b - s, -b + s);
    }

    float depthOf(float3 p, constant SphereU& u)
    {
        float4 c = u.viewProjection * float4(p, 1.0);
        return c.z / c.w;
    }

    float hash31(float3 p)
    {
        p = fract(p * 0.3183099 + float3(0.1, 0.2, 0.3));
        p *= 17.0;
        return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
    }

    // ---- The full-screen passes ----

    struct FullVarying {
        float4 position [[position]];
        float2 ndc;
    };

    vertex FullVarying sphereFullVertex(uint vid [[vertex_id]])
    {
        float2 corner = float2(((vid & 1u) != 0u) ? 1.0 : -1.0, ((vid & 2u) != 0u) ? 1.0 : -1.0);
        FullVarying out;
        out.position = float4(corner, 1.0, 1.0);
        out.ndc = corner;
        return out;
    }

    // The sky: a dark gradient with stars, the atmosphere's glow around the limb from outside, and the haze of the horizon from inside.
    fragment float4 sphereSkyFragment(FullVarying in [[stage_in]], constant SphereU& u [[buffer(0)]])
    {
        float3 o = u.eye.xyz;
        float3 d = rayDirection(in.ndc, u);
        float3 col = mix(float3(0.016, 0.019, 0.032), float3(0.030, 0.038, 0.066), 0.5 + 0.5 * in.ndc.y);
        float lid = u.planet.y;
        float r = length(o);
        float tc = -dot(o, d);
        if (r > lid) {
            // stars: a sparse hash of the direction (cells of about 0.4 degree), faint
            float3 scaled = d * 170.0;
            float3 cell = floor(scaled);
            float s = hash31(cell);
            if (s > 0.992) {
                // a round point, not the cell: the star sits at a hashed place inside it
                float3 centre = float3(hash31(cell + 3.0), hash31(cell + 5.0), hash31(cell + 11.0)) * 0.5 + 0.25;
                float dist = length(fract(scaled) - centre);
                float twinkle = 0.35 + 0.65 * hash31(cell + 7.0);
                col += float3(0.85, 0.9, 1.0) * twinkle * (1.0 - smoothstep(0.0, 0.16, dist)) * 0.7;
            }
            if (tc > 0.0) {
                float h = length(o + d * tc);
                // only outside the lid's silhouette: over the disc the lid pass draws the rim
                float x = max(h - lid, 0.0) / (0.12 * lid);
                float glow = (h >= lid ? exp(-x * 2.2) : 0.0) * smoothstep(0.0, 1.0, tc / max(r, 1.0));
                col += float3(0.16, 0.36, 0.78) * glow * 0.85;
            }
        } else {
            // inside the atmosphere: the sky is bluer towards the horizon of the place
            float3 localUp = o / max(r, 1e-3);
            float elevation = dot(d, localUp);
            float haze = exp(-max(elevation, 0.0) / 0.35);
            col = mix(col, float3(0.10, 0.20, 0.42), 0.8 * haze);
        }
        return float4(col, 1.0);
    }

    struct SurfaceOut {
        float4 color [[color(0)]];
        float depth [[depth(any)]];
    };

    float lightAt(float3 n, constant SphereU& u)
    {
        float l = dot(n, normalize(u.sun.xyz));
        return u.sun.w + (1.0 - u.sun.w) * smoothstep(-0.3, 0.7, l);
    }

    // The floor: the weather map and the built cells by pixel, the territory tint and its borders, the graticule, the cell borders.
    fragment SurfaceOut sphereFloorFragment(FullVarying in [[stage_in]],
                                            constant SphereU& u [[buffer(0)]],
                                            device const float4* floorColours [[buffer(1)]],
                                            device const float4* territory [[buffer(2)]])
    {
        SurfaceOut out;
        out.color = float4(0.0);
        out.depth = 1.0;
        float3 o = u.eye.xyz;
        float3 d = rayDirection(in.ndc, u);
        float rp = u.planet.x;
        float2 t = raySphere(o, d, rp);
        if (t.x <= 0.0) { discard_fragment(); }
        float3 p = o + d * t.x;
        float3 n = p / rp;
        int pix = healpix_ring(n, u.nsides.x);
        float3 color = float3(0.08, 0.10, 0.13);
        if ((u.flags.x & 1u) != 0u) { color = floorColours[pix].rgb; }
        if ((u.flags.x & 64u) != 0u) {   // unlit: the cell's colour as it is (the tests read the lookup back through it)
            out.color = float4(color, 1.0);
            out.depth = depthOf(p, u);
            return out;
        }

        // pixel footprints: the neighbouring samples one pixel to the right and below, for the cell and territory borders
        float3 nx = normalize(n + dfdx(n));
        float3 ny = normalize(n + dfdy(n));

        if ((u.flags.x & 4u) != 0u) {
            int tp = healpix_ring(n, u.nsides.y);
            float4 here = territory[tp];
            color = mix(color, here.rgb, u.look.y * here.a);
        }
        if ((u.flags.x & 8u) != 0u) {
            int tp = healpix_ring(n, u.nsides.y);
            int t1 = healpix_ring(nx, u.nsides.y);
            int t2 = healpix_ring(ny, u.nsides.y);
            float4 here = territory[tp];
            float4 a = territory[t1];
            float4 b = territory[t2];
            bool edge1 = t1 != tp && (any(a.rgb != here.rgb) || (a.a > 0.0) != (here.a > 0.0)) && (a.a > 0.0 || here.a > 0.0);
            bool edge2 = t2 != tp && (any(b.rgb != here.rgb) || (b.a > 0.0) != (here.a > 0.0)) && (b.a > 0.0 || here.a > 0.0);
            if (edge1 || edge2) { color = mix(color, float3(1.0), u.look.z); }
        }
        if ((u.flags.x & 16u) != 0u && u.look.w > 0.0) {
            float lat = asin(clamp(n.z, -1.0, 1.0));
            float lon = atan2(n.y, n.x);
            float gy = lat * (6.0 / kPi);          // every 30 degrees
            float gx = lon * (6.0 / kPi);
            float fy = fract(gy), fx = fract(gx);
            float ly = 1.0 - smoothstep(0.5, 1.5, min(fy, 1.0 - fy) / max(fwidth(gy), 1e-6));
            float lx = 1.0 - smoothstep(0.5, 1.5, min(fx, 1.0 - fx) / max(fwidth(gx), 1e-6));
            lx *= cos(lat) > 0.12 ? 1.0 : 0.0;
            float equator = 1.0 - smoothstep(0.5, 1.5, fabs(lat) / max(fwidth(lat), 1e-6));
            float line = max(max(ly, lx * 0.8), equator);
            color = mix(color, float3(0.75, 0.82, 0.95), u.look.w * line);
        }
        if (u.style.w > 0.0) {
            int p1 = healpix_ring(nx, u.nsides.x);
            int p2 = healpix_ring(ny, u.nsides.x);
            if (p1 != pix || p2 != pix) { color = mix(color, float3(0.9, 0.95, 1.0), u.style.w); }
        }
        if (u.flags.y != 0xFFFFFFFFu) {
            int p1 = healpix_ring(nx, u.nsides.x);
            int p2 = healpix_ring(ny, u.nsides.x);
            if (uint(pix) == u.flags.y) {
                color = mix(color, float3(1.0, 0.95, 0.55), 0.22);
                if (p1 != pix || p2 != pix) { color = float3(1.0, 0.95, 0.55); }
            }
        }
        // the atmosphere over the floor: a faint blue at the limb as seen through the shell
        float ndv = clamp(dot(n, -d), 0.0, 1.0);
        float rim = pow(1.0 - ndv, 3.0);
        color = color * lightAt(n, u) * u.look.x + float3(0.10, 0.22, 0.50) * rim * 0.55;
        out.color = float4(clamp(color, 0.0, 1.0), 1.0);
        out.depth = depthOf(p, u);
        return out;
    }

    // The lid: the cloud shell and the glow of the atmosphere, translucent, over everything drawn before it.
    fragment SurfaceOut sphereLidFragment(FullVarying in [[stage_in]],
                                          constant SphereU& u [[buffer(0)]],
                                          device const float4* cloud [[buffer(1)]])
    {
        SurfaceOut out;
        out.color = float4(0.0);
        out.depth = 1.0;
        float3 o = u.eye.xyz;
        float3 d = rayDirection(in.ndc, u);
        float lid = u.planet.y;
        float2 t = raySphere(o, d, lid);
        bool inside = length(o) < lid;
        float tt = inside ? t.y : t.x;
        if (tt <= 0.0) { discard_fragment(); }
        float3 p = o + d * tt;
        float3 n = p / lid;
        float4 c = float4(0.0);
        if ((u.flags.x & 2u) != 0u) {
            c = cloud[healpix_ring(n, u.nsides.x)];
        }
        float ndv = clamp(fabs(dot(n, d)), 0.0, 1.0);
        float rim = pow(1.0 - ndv, 4.0);
        float3 shade = c.rgb * (0.45 + 0.55 * lightAt(n, u)) * u.look.x;
        float3 glow = float3(0.18, 0.40, 0.85);
        float ca = clamp(c.a * u.viewport.w, 0.0, 1.0);
        float ra = clamp(rim * 0.6, 0.0, 1.0);
        // premultiplied: the cloud over the glow
        out.color = float4(shade * ca + glow * ra * (1.0 - ca), ca + ra * (1.0 - ca));
        out.depth = depthOf(p, u);
        return out;
    }

    // ---- Segments: storm rings and tracks, winds, bonds, pools, the selection ----

    struct Seg {
        packed_float3 a;
        float widthA;
        packed_float3 b;
        float widthB;
        float4 colorA;
        float4 colorB;
    };

    struct SegVarying {
        float4 position [[position]];
        float4 color;
        float across;
    };

    // A segment is a screen-space quad of the width in pixels (widthA at the first end, widthB at the second).
    vertex SegVarying sphereSegmentVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                          constant SphereU& u [[buffer(0)]], device const Seg* segs [[buffer(1)]])
    {
        SegVarying out;
        out.position = float4(2.0, 2.0, 2.0, 1.0);
        out.color = float4(0.0);
        out.across = 0.0;
        Seg s = segs[iid];
        float4 ca = u.viewProjection * float4(float3(s.a), 1.0);
        float4 cb = u.viewProjection * float4(float3(s.b), 1.0);
        if (ca.w <= 0.01 || cb.w <= 0.01) { return out; }
        float2 pa = ca.xy / ca.w * u.viewport.xy * 0.5;
        float2 pb = cb.xy / cb.w * u.viewport.xy * 0.5;
        float2 dir = pb - pa;
        float len = length(dir);
        dir = len > 1e-4 ? dir / len : float2(1.0, 0.0);
        float2 nrm = float2(-dir.y, dir.x);
        bool second = (vid & 2u) != 0u;
        float side = ((vid & 1u) != 0u) ? 1.0 : -1.0;
        float4 c = second ? cb : ca;
        float w = second ? s.widthB : s.widthA;
        float2 offset = (nrm * side + dir * (second ? 1.0 : -1.0) * 0.5) * (0.5 * w + 0.75);
        float2 ndcOffset = offset / (u.viewport.xy * 0.5);
        // a small bias towards the eye keeps a line on the floor from fighting the floor's depth
        float4 clip = float4(c.xy + ndcOffset * c.w, c.z - 0.0004 * c.w, c.w);
        out.position = clip;
        out.color = second ? s.colorB : s.colorA;
        out.across = side;
        return out;
    }

    fragment float4 sphereSegmentFragment(SegVarying in [[stage_in]])
    {
        // a soft edge across the quad; premultiplied
        float a = in.color.a * (1.0 - smoothstep(0.55, 1.0, fabs(in.across)));
        return float4(in.color.rgb * a, a);
    }

    // ---- Agents: one billboard per listed row, shaded as a ray-traced sphere with a nose (an impostor) ----

    constant float noseOffset = 0.75;
    constant float noseRadius = 0.5;
    constant float boundRadius = 1.25;

    struct AgentVarying {
        float4 position [[position]];
        float3 center;
        float3 forward;
        float3 world;
        float3 color;
        float radius;
    };

    // The height colour: the viridis ramp of the altitude over the shell, 17 stops of sRGB (HeightColour.swift writes this table, the same
    // the box world's shaders and the legend use), linear between them.
    constant float3 heightStops[17] = { \(HeightColour.shaderTable) };

    float3 heightRamp(float t)
    {
        float x = clamp(t, 0.0, 1.0) * 16.0;
        int i = min(int(x), 15);
        return mix(heightStops[i], heightStops[i + 1], x - float(i));
    }

    vertex AgentVarying sphereAgentVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                          constant SphereU& u [[buffer(0)]],
                                          device const packed_float3* pos [[buffer(1)]],
                                          device const half2* orient [[buffer(2)]],
                                          device const uint* rgb [[buffer(3)]])
    {
        AgentVarying out;
        out.position = float4(2.0, 2.0, 2.0, 1.0);
        out.center = float3(0.0);
        out.forward = float3(1.0, 0.0, 0.0);
        out.world = float3(0.0);
        out.color = float3(0.0);
        out.radius = 1.0;
        float3 center = float3(pos[iid]);
        float3 toCenter = center - u.eye.xyz;
        float depth = dot(toCenter, u.forward.xyz);
        if (depth <= 0.2) { return out; }
        // pixels per world unit at this depth, and the radius drawn: the physical one, held between the smallest and the largest
        float ppu = u.viewport.y / (2.0 * depth * u.planet.w);
        float physical = u.planet.z * ppu;
        float drawn = clamp(physical, u.style.x, max(u.style.y, u.style.x));
        float radius = drawn / ppu;
        float3 f = normalize(toCenter);
        float3 bx = normalize(cross(f, u.up.xyz));
        float3 by = cross(bx, f);
        float dist = length(toCenter);
        float bound = boundRadius * radius;
        float extent = bound * rsqrt(max(1.0 - (bound * bound) / (dist * dist), 0.05));
        float2 corner = float2(((vid & 1u) != 0u) ? 1.0 : -1.0, ((vid & 2u) != 0u) ? 1.0 : -1.0);
        float3 world = center + (bx * corner.x + by * corner.y) * extent;
        float2 angles = float2(orient[iid]);
        out.forward = agentHeading(center, angles.x, angles.y);
        out.position = u.viewProjection * float4(world, 1.0);
        out.center = center;
        out.world = world;
        out.radius = radius;
        float3 base = unpackColor(rgb[iid]);
        if ((u.flags.x & 32u) != 0u && (rgb[iid] & 0xFFFFFFu) != 0xFF0000u) {
            base = heightRamp((length(center) - u.planet.x) / max(u.colours.x, 1e-3));
        }
        out.color = base;
        return out;
    }

    struct AgentOut {
        float4 color [[color(0)]];
        float depth [[depth(any)]];
    };

    fragment AgentOut sphereAgentFragment(AgentVarying in [[stage_in]], constant SphereU& u [[buffer(0)]])
    {
        AgentOut out;
        out.color = float4(0.0);
        out.depth = 1.0;
        float3 o = u.eye.xyz;
        float3 dir = normalize(in.world - o);
        float r = in.radius;
        float3 noseCenter = in.center + in.forward * (noseOffset * r);
        float noseR = noseRadius * r;

        float3 oc = o - in.center;
        float b1 = dot(oc, dir);
        float d1 = b1 * b1 - (dot(oc, oc) - r * r);
        float t1 = d1 >= 0.0 ? -b1 - sqrt(d1) : -1.0;
        float3 on = o - noseCenter;
        float b2 = dot(on, dir);
        float d2 = b2 * b2 - (dot(on, on) - noseR * noseR);
        float t2 = d2 >= 0.0 ? -b2 - sqrt(d2) : -1.0;
        bool hitBody = t1 > 0.0;
        bool hitNose = t2 > 0.0;
        if (!hitBody && !hitNose) { discard_fragment(); }
        bool nose = hitNose && (!hitBody || t2 < t1);
        float t = nose ? t2 : t1;
        float3 hit = o + dir * t;
        float3 centre = nose ? noseCenter : in.center;
        float3 n = normalize(hit - centre);
        float3 base = nose ? mix(in.color, float3(1.0), 0.45) : in.color;
        float3 sun = normalize(u.sun.xyz);
        float diffuse = max(dot(n, sun), 0.0);
        float head = max(dot(n, -dir), 0.0);
        float spec = pow(max(dot(reflect(-sun, n), -dir), 0.0), 24.0);
        float3 shaded = base * (0.28 + 0.52 * diffuse + 0.30 * head) + float3(0.18) * spec;
        // a darker rim so that a pale organism reads over a cloud and a dark one over the sea
        float rim = pow(1.0 - head, 2.5);
        shaded = mix(shaded, shaded * 0.35, rim * 0.7);
        out.color = float4(clamp(shaded, 0.0, 1.0) * u.style.z, u.style.z);
        out.depth = depthOf(hit, u);
        return out;
    }
    """
}

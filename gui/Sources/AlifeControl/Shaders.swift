import AlifeCore
// The viewer's Metal shaders as one string, compiled at run time with `MTLDevice.makeLibrary(source:options:)`.
//
// They are not a `.metal` file because the installed app is the release binary and nothing else (`make-app.sh` copies
// no resource bundle), and because compiling a `.metal` file at build time needs the Metal Toolchain, which is a separate
// download. The price is that the source is first compiled when the viewer opens; the compiler's diagnostics then reach
// the viewer window (`Renderer.Failure.shaders`). Do not use a backslash or a triple quote in the source.
//
// The structs `Scene`, `Plane` and `WorldU` mirror `SceneUniforms`, `PlaneUniforms` and `WorldUniforms` in Renderer.swift
// field by field (a test pins the Swift side: 160, 32 and 112 bytes).
enum Shaders {
    /// The entry points of `source` that `Renderer` builds pipelines from (a test compares them with the source).
    static let sphereVertex = "sphereVertex"
    static let sphereFragment = "sphereFragment"
    static let bondVertex = "bondVertex"
    static let boxVertex = "boxVertex"
    static let lineFragment = "lineFragment"
    static let planeVertex = "planeVertex"
    static let planeFragment = "planeFragment"
    static let sliceFragment = "sliceFragment"
    // The World window's entry points (DESIGN 33.4.6): alive-only agents at their minimum image, the repeating map.
    static let agentVertexA = "agentVertexA"
    static let agentFragmentA = "agentFragmentA"
    static let agentFragmentFlat = "agentFragmentFlat"
    static let bondVertexA = "bondVertexA"
    static let pointVertexA = "pointVertexA"
    static let pointFragment = "pointFragment"
    static let mapVertex = "mapVertex"
    static let mapFragment = "mapFragment"
    // The minimap's (DESIGN 36.5): the whole world in a viewport of the drawable, the map's fragment shader over a quad of its own,
    // and the organisms as translucent points.
    static let insetVertex = "insetVertex"
    static let insetPointVertex = "insetPointVertex"
    static let entryPoints = [
        sphereVertex, sphereFragment, bondVertex, boxVertex, lineFragment, planeVertex, planeFragment, sliceFragment,
        agentVertexA, agentFragmentA, agentFragmentFlat, bondVertexA, pointVertexA, pointFragment, mapVertex, mapFragment,
        insetVertex, insetPointVertex,
    ]

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Scene {
        float4x4 viewProjection;
        float4 eye;        // xyz: the eye
        float4 right;      // xyz: the screen's +x in the world
        float4 up;         // xyz: the screen's +y in the world
        float4 world;      // x: L (x and y are periodic on [0, L)), y: height, z: the agent radius
        uint4 counts;      // x: rows, y: bonds, z: dims
        float4 viewport;   // x, y: the drawable size in pixels
    };

    struct Plane {
        float4 color;
        float4 info;       // x: the z of the plane
    };

    // The World window's constants (buffer 7).
    struct WorldU {
        float4 target;     // xy: the camera target (everything is drawn at its minimum image), z: L, w: 1 / L
        float4 params;     // x: map blend (previous -> current), y: map opacity, z: territory tint alpha, w: flat-disc threshold in px
        float4 refs;       // x: p_ref, y: dc_ref, z: zeta_ref, w: cot(fovY / 2)
        uint4 flags;       // x: pass (0 spheres, 1 flat discs), y: rows listed, z: bonds, w: bit 0 map, bit 1 territory, bit 2 equators, bit 3 height colour
        float4 style;      // x: point alpha, y: point size in px, z: smallest radius in px, w: outline width in px (0: none)
        float4 size;       // x: the cap on the drawn radius in px (huge: physical size), y, z: the alpha ramp's low and high radius in px, w: depth cue slope
        float4 look;       // x: exposure (the map's brightness), y: territory border alpha, z: depth cue base, w: hemisphere tint weight
        float4 patch;      // x: the index k of the cell the view's centre is in, y, z, w: the alpha of the copies k - 1, k, k + 1 (0: not drawn)
    };

    // The nose of an agent: a small sphere on its forward axis, paler than the body, so the heading shows.
    constant float noseOffset = 0.75;
    constant float noseRadius = 0.5;
    // A sphere about the agent's centre that holds the body and the nose, in units of the agent radius: the farthest point
    // of the nose from the centre is noseOffset + noseRadius.
    constant float boundRadius = 1.25;

    float3 unpackColor(uint c) {
        return float3(float((c >> 16) & 255u), float((c >> 8) & 255u), float(c & 255u)) * (1.0 / 255.0);
    }

    // The height colour mode (DESIGN 36.5): an organism is coloured by its altitude on the viridis ramp (HeightColour.swift writes this
    // table), 17 stops of sRGB, linear between them. The engine's hurt red is kept. Bit 3 of the world flags turns it on.
    constant float3 heightStops[17] = { \(HeightColour.shaderTable) };

    float3 heightColor(float z, float height)
    {
        float t = clamp(z / max(height, 1e-6), 0.0, 1.0) * 16.0;
        int i = min(int(t), 15);
        return mix(heightStops[i], heightStops[i + 1], t - float(i));
    }

    // The colour of a row: the stream's, or in the height mode its altitude's, except the engine's hurt red (FF0000), which every mode keeps.
    float3 organismColor(uint packed, float z, float height, constant WorldU& world)
    {
        if ((world.flags.w & 8u) != 0u && height > 0.0 && (packed & 0xFFFFFFu) != 0xFF0000u) { return heightColor(z, height); }
        return unpackColor(packed);
    }

    // ---- Agents: one billboard per row, shaded as a ray-traced sphere (an impostor) ----

    struct SphereVarying {
        float4 position [[position]];
        float3 center;
        float3 forward;
        float3 world;
        float3 color;
    };

    vertex SphereVarying sphereVertex(uint vid [[vertex_id]],
                                      uint iid [[instance_id]],
                                      constant Scene& scene [[buffer(0)]],
                                      device const uchar* alive [[buffer(1)]],
                                      device const packed_float3* pos [[buffer(2)]],
                                      device const half2* orient [[buffer(3)]],
                                      device const uint* rgb [[buffer(4)]])
    {
        SphereVarying out;
        // A dead row, or one the eye is in or behind, draws nothing: all four corners at one point outside the clip volume.
        out.position = float4(2.0, 2.0, 2.0, 1.0);
        out.center = float3(0.0);
        out.forward = float3(1.0, 0.0, 0.0);
        out.world = float3(0.0);
        out.color = float3(0.0);
        if (alive[iid] == 0) { return out; }

        float radius = scene.world.z;
        float3 center = float3(pos[iid]);
        float3 toCenter = center - scene.eye.xyz;
        float dist = length(toCenter);
        float3 viewAxis = cross(scene.up.xyz, scene.right.xyz);
        if (!(dist > 1.5 * radius) || dot(toCenter, viewAxis) <= 0.0) { return out; }

        // The billboard faces the eye, and is as large as the silhouette of the sphere that holds the body and the nose
        // (seen from outside it, from the eye at least 1.5 radii away: R / sqrt(1 - R^2 / d^2) at the plane of the centre).
        float3 f = toCenter / dist;
        float3 bx = normalize(cross(f, scene.up.xyz));
        float3 by = cross(bx, f);
        float bound = boundRadius * radius;
        float extent = bound * rsqrt(max(1.0 - (bound * bound) / (dist * dist), 0.05));
        float2 corner = float2(((vid & 1u) != 0u) ? 1.0 : -1.0, ((vid & 2u) != 0u) ? 1.0 : -1.0);
        float3 world = center + (bx * corner.x + by * corner.y) * extent;

        // The forward axis is (cos(pitch) cos(yaw), cos(pitch) sin(yaw), sin(pitch)), as in DESIGN 27.2.
        float2 angles = float2(orient[iid]);
        float cp = cos(angles.y);
        out.forward = float3(cp * cos(angles.x), cp * sin(angles.x), sin(angles.y));
        out.position = scene.viewProjection * float4(world, 1.0);
        out.center = center;
        out.world = world;
        out.color = unpackColor(rgb[iid]);
        return out;
    }

    struct SphereOut {
        float4 color [[color(0)]];
        float depth [[depth(any)]];
    };

    // sRGB to linear light, and the outline of an impostor: dark on a light pixel, light on a dark one, chosen from the luminance of
    // the shaded colour of the pixel next to the ring, so that the edge has a contrast of at least 3:1 whatever the colour,
    // the depth cue and the shading (DESIGN 36.7; OutlineContrast.swift is the same arithmetic).
    float srgbToLinear(float c)
    {
        c = clamp(c, 0.0, 1.0);
        return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
    }

    float3 outlineFor(float3 shaded)
    {
        float l = 0.2126 * srgbToLinear(shaded.r) + 0.7152 * srgbToLinear(shaded.g) + 0.0722 * srgbToLinear(shaded.b);
        return l > 0.17 ? float3(0.06) : float3(0.92);
    }

    // The shading of one impostor: the ray against the body and the nose, the headlight, the depth of the hit. `radius` is the
    // body's, `cue` a brightness factor, `ring` the outline's width in pixels (0: none; `pxPerUnit` pixels per world unit) and
    // `alpha` the opacity of the whole impostor. The Lab draws with the scene's radius, cue 1, no ring, alpha 1.
    SphereOut shadeSphere(float3 center, float3 forwardAxis, float3 worldPoint, float3 baseColor, constant Scene& scene,
                          float radius, float pxPerUnit, float cue, float ring, float alpha)
    {
        SphereOut out;
        out.color = float4(0.0);
        out.depth = 1.0;

        float3 origin = scene.eye.xyz;
        float3 dir = normalize(worldPoint - origin);
        float3 noseCenter = center + forwardAxis * (noseOffset * radius);
        float noseR = noseRadius * radius;

        // Ray against the body and against the nose. The squared miss distance is taken from the component of the
        // offset perpendicular to the ray, which keeps its precision when the eye is far away.
        float tBody = -1.0;
        float3 oc = origin - center;
        float b = dot(oc, dir);
        float3 q = oc - dir * b;
        float h = radius * radius - dot(q, q);
        if (h >= 0.0) { tBody = -b - sqrt(h); }

        float tNose = -1.0;
        float3 ocn = origin - noseCenter;
        float bn = dot(ocn, dir);
        float3 qn = ocn - dir * bn;
        float hn = noseR * noseR - dot(qn, qn);
        if (hn >= 0.0) { tNose = -bn - sqrt(hn); }

        bool hitBody = tBody > 0.0;
        bool hitNose = tNose > 0.0;
        if (!hitBody && !hitNose) {
            discard_fragment();
            return out;
        }
        bool useNose = hitNose && (!hitBody || tNose < tBody);
        float t = useNose ? tNose : tBody;
        float3 hit = origin + dir * t;
        float3 normal = useNose ? (hit - noseCenter) / noseR : (hit - center) / radius;
        float3 base = (useNose ? mix(baseColor, float3(1.0), 0.6) : baseColor) * cue;
        float lambert = max(dot(normal, -dir), 0.0);
        out.color = float4(base * (0.3 + 0.7 * lambert), alpha);
        // The outline is the silhouette of the organism: the body's edge, and the nose's edge where the nose sticks out past the body
        // (a ray that misses the body). A nose in front of the body has its edge inside the organism: no ring there (the "eye" look).
        if (ring > 0.0 && !(useNose && hitBody)) {
            // The distance of the pixel to the silhouette of the sphere it shows, in pixels: the sphere's radius less its miss distance.
            float rr = useNose ? noseR : radius;
            float miss = sqrt(max(rr * rr - (useNose ? hn : h), 0.0));
            if ((rr - miss) * pxPerUnit < ring) {
                // The pixel next to the ring is the one the outline is chosen against: n.v there is the sphere's at (ring + 0.5) px inside.
                float rrPx = rr * pxPerUnit;
                float k = clamp((rrPx - ring - 0.5) / max(rrPx, 1e-3), 0.0, 1.0);
                out.color = float4(outlineFor(base * (0.3 + 0.7 * sqrt(1.0 - k * k))), alpha);
            }
        }
        float4 clip = scene.viewProjection * float4(hit, 1.0);
        out.depth = clip.z / clip.w;
        return out;
    }

    fragment SphereOut sphereFragment(SphereVarying in [[stage_in]],
                                      constant Scene& scene [[buffer(0)]])
    {
        return shadeSphere(in.center, in.forward, in.world, in.color, scene, scene.world.z, 1.0, 1.0, 0.0, 1.0);
    }

    // ---- Lines: bonds and the edges of the box ----

    struct LineVarying {
        float4 position [[position]];
        float4 color;
    };

    // A bond is two half segments, one from each agent to the midpoint of the minimum-image offset between them, so a
    // bond across the periodic seam in x or y is drawn as two stubs towards the seam, not as a line across the world.
    // Four vertices per bond: (a, a + d/2) and (b, b - d/2), drawn as a line list. z is not periodic.
    vertex LineVarying bondVertex(uint vid [[vertex_id]],
                                  uint iid [[instance_id]],
                                  constant Scene& scene [[buffer(0)]],
                                  device const packed_float3* pos [[buffer(2)]],
                                  device const uint* rgb [[buffer(4)]],
                                  device const uint2* bonds [[buffer(5)]])
    {
        uint2 ij = bonds[iid];
        float3 pa = float3(pos[ij.x]);
        float3 pb = float3(pos[ij.y]);
        float3 d = pb - pa;
        float size = scene.world.x;
        d.xy -= size * rint(d.xy / size);
        bool fromB = vid >= 2u;
        bool atMiddle = (vid & 1u) != 0u;
        float3 p = fromB ? pb : pa;
        if (atMiddle) { p += (fromB ? -0.5 : 0.5) * d; }
        LineVarying out;
        out.position = scene.viewProjection * float4(p, 1.0);
        out.color = float4(mix(unpackColor(rgb[fromB ? ij.y : ij.x]), float3(1.0), 0.35), 1.0);
        return out;
    }

    // The twelve edges of the box [0, L] x [0, L] x [0, height] as a line list (24 vertices). In 2D, height is 0 and the
    // box is the square of the world.
    vertex LineVarying boxVertex(uint vid [[vertex_id]],
                                 constant Scene& scene [[buffer(0)]])
    {
        uint edge = vid / 2u;
        uint axis = edge / 4u;
        uint combo = edge % 4u;
        float u = float(combo & 1u);
        float v = float((combo >> 1u) & 1u);
        float w = ((vid & 1u) != 0u) ? 1.0 : 0.0;
        float3 unit = float3(u, v, w);
        if (axis == 0u) { unit = float3(w, u, v); }
        if (axis == 1u) { unit = float3(u, w, v); }
        LineVarying out;
        out.position = scene.viewProjection * float4(unit * float3(scene.world.x, scene.world.x, scene.world.y), 1.0);
        out.color = float4(0.5, 0.58, 0.7, 1.0);
        return out;
    }

    fragment float4 lineFragment(LineVarying in [[stage_in]])
    {
        return in.color;
    }

    // ---- Planes: the floor, the surface and the field slice, each a quad of [0, L] x [0, L] at some z ----

    struct PlaneVarying {
        float4 position [[position]];
        float2 uv;
    };

    vertex PlaneVarying planeVertex(uint vid [[vertex_id]],
                                    constant Scene& scene [[buffer(0)]],
                                    constant Plane& plane [[buffer(6)]])
    {
        float2 uv = float2(((vid & 1u) != 0u) ? 1.0 : 0.0, ((vid & 2u) != 0u) ? 1.0 : 0.0);
        PlaneVarying out;
        out.position = scene.viewProjection * float4(uv * scene.world.x, plane.info.x, 1.0);
        out.uv = uv;
        return out;
    }

    fragment float4 planeFragment(PlaneVarying in [[stage_in]],
                                  constant Plane& plane [[buffer(6)]])
    {
        return plane.color;
    }

    // The field slice: the texture is the stream's FG x FG colours, row [cy] and column [cx], so u = x / L and v = y / L.
    fragment float4 sliceFragment(PlaneVarying in [[stage_in]],
                                  constant Plane& plane [[buffer(6)]],
                                  texture2d<float> field [[texture(0)]])
    {
        constexpr sampler cells(coord::normalized, address::clamp_to_edge, filter::nearest);
        return float4(field.sample(cells, in.uv).rgb, plane.color.a);
    }

    // ---- The World window: alive-only agents at their minimum image, bonds, points and the repeating map ----

    // The image of a point nearest the camera target: x and y are periodic on [0, L), z is not.
    float3 minimumImage(float3 p, float2 target, float L)
    {
        float2 d = p.xy - target;
        d -= L * rint(d / L);
        return float3(target + d, p.z);
    }

    struct AgentVaryingA {
        float4 position [[position]];
        float3 center;
        float3 forward;
        float3 world;
        float3 color;
        float2 corner;                 // the billboard's corner in [-1, 1]^2 (the flat discs)
        float radius [[flat]];         // the drawn radius in world units
        float pxPerUnit [[flat]];      // pixels per world unit at the centre's depth
        float alpha [[flat]];          // the instance's opacity: a ramp of the drawn radius
        float cue [[flat]];            // the depth cue: a brightness factor by height
    };

    // One billboard per listed row (the stream's alive-only arrays: the instance id is the row). Two passes over the same
    // rows: pass 0 draws the ones whose drawn radius is at least the threshold as ray-traced spheres, pass 1 the rest as
    // flat discs, which a ray test would turn into nothing. The drawn radius is the physical one clamped by the size policy
    // (DESIGN 36.7): at least the smallest radius, at most the larger of that and the cap; the opacity is a ramp of it, so an
    // organism too small to see is not drawn and there is no level at which they all appear.
    vertex AgentVaryingA agentVertexA(uint vid [[vertex_id]],
                                      uint iid [[instance_id]],
                                      constant Scene& scene [[buffer(0)]],
                                      device const packed_float3* pos [[buffer(2)]],
                                      device const half2* orient [[buffer(3)]],
                                      device const uint* rgb [[buffer(4)]],
                                      constant WorldU& world [[buffer(7)]])
    {
        AgentVaryingA out;
        out.position = float4(2.0, 2.0, 2.0, 1.0);
        out.center = float3(0.0);
        out.forward = float3(1.0, 0.0, 0.0);
        out.world = float3(0.0);
        out.color = float3(0.0);
        out.corner = float2(0.0);
        out.radius = scene.world.z;
        out.pxPerUnit = 1.0;
        out.alpha = 1.0;
        out.cue = 1.0;

        float3 raw = float3(pos[iid]);
        float3 center = minimumImage(raw, world.target.xy, world.target.z);
        float3 toCenter = center - scene.eye.xyz;
        float dist = length(toCenter);
        float3 viewAxis = cross(scene.up.xyz, scene.right.xyz);
        float depth = dot(toCenter, viewAxis);
        if (depth <= 0.0) { return out; }

        // Pixels per world unit at the centre, the body's physical radius in pixels, and the radius it is drawn with.
        float perPixel = 0.5 * scene.viewport.y * world.refs.w / depth;
        float smallest = world.style.z;
        float drawnPx = clamp(scene.world.z * perPixel, smallest, max(smallest, world.size.x));
        float radius = drawnPx / perPixel;
        if (!(dist > 1.5 * radius)) { return out; }
        float alpha = smoothstep(world.size.y, world.size.z, drawnPx);
        if (!(alpha > 0.0)) { return out; }
        bool isFlat = drawnPx < world.params.w;
        if (isFlat != (world.flags.x == 1u)) { return out; }

        float2 corner = float2(((vid & 1u) != 0u) ? 1.0 : -1.0, ((vid & 2u) != 0u) ? 1.0 : -1.0);
        float3 worldPoint;
        if (isFlat) {
            worldPoint = center + (scene.right.xyz * corner.x + scene.up.xyz * corner.y) * radius;
        } else {
            float3 f = toCenter / dist;
            float3 bx = normalize(cross(f, scene.up.xyz));
            float3 by = cross(bx, f);
            float bound = boundRadius * radius;
            float extent = bound * rsqrt(max(1.0 - (bound * bound) / (dist * dist), 0.05));
            worldPoint = center + (bx * corner.x + by * corner.y) * extent;
        }
        float2 angles = float2(orient[iid]);
        float cp = cos(angles.y);
        out.forward = float3(cp * cos(angles.x), cp * sin(angles.x), sin(angles.y));
        out.position = scene.viewProjection * float4(worldPoint, 1.0);
        out.center = center;
        out.world = worldPoint;
        out.color = organismColor(rgb[iid], raw.z, scene.world.y, world);
        out.corner = corner;
        out.radius = radius;
        out.pxPerUnit = perPixel;
        out.alpha = alpha;
        // (the height colour is the altitude: no second brightness cue on top of it)
        bool byHeight = (world.flags.w & 8u) != 0u;
        out.cue = (scene.world.y > 0.0 && !byHeight) ? world.look.z + world.size.w * clamp(raw.z / scene.world.y, 0.0, 1.0) : 1.0;
        return out;
    }

    fragment SphereOut agentFragmentA(AgentVaryingA in [[stage_in]],
                                      constant Scene& scene [[buffer(0)]],
                                      constant WorldU& world [[buffer(7)]])
    {
        return shadeSphere(in.center, in.forward, in.world, in.color, scene, in.radius, in.pxPerUnit, in.cue, world.style.w, in.alpha);
    }

    // A flat disc: no ray test, a little shading towards the rim, the depth of the billboard, and the same outline as a sphere.
    fragment SphereOut agentFragmentFlat(AgentVaryingA in [[stage_in]],
                                         constant WorldU& world [[buffer(7)]])
    {
        SphereOut out;
        out.color = float4(0.0);
        out.depth = 1.0;
        float r = length(in.corner);
        if (r > 1.0) {
            discard_fragment();
            return out;
        }
        float3 base = in.color * in.cue;
        float ring = world.style.w;
        float discPx = in.radius * in.pxPerUnit;
        float shade = 1.0 - 0.3 * r * r;
        if (ring > 0.0 && (1.0 - r) * discPx < ring) {
            float k = clamp((discPx - ring - 0.5) / max(discPx, 1e-3), 0.0, 1.0);
            out.color = float4(outlineFor(base * (1.0 - 0.3 * k * k)), in.alpha);
        } else {
            out.color = float4(base * shade, in.alpha);
        }
        out.depth = in.position.z;
        return out;
    }

    // A bond between listed rows: one segment between the images of its endpoints nearest the target (the first against
    // the target, the second against the first), drawn as two half segments, each in its agent's colour.
    vertex LineVarying bondVertexA(uint vid [[vertex_id]],
                                   uint iid [[instance_id]],
                                   constant Scene& scene [[buffer(0)]],
                                   device const packed_float3* pos [[buffer(2)]],
                                   device const uint* rgb [[buffer(4)]],
                                   device const uint2* bonds [[buffer(5)]],
                                   constant WorldU& world [[buffer(7)]])
    {
        uint2 ij = bonds[iid];
        float L = world.target.z;
        float3 pa = minimumImage(float3(pos[ij.x]), world.target.xy, L);
        float3 pb = minimumImage(float3(pos[ij.y]), pa.xy, L);
        bool fromB = vid >= 2u;
        bool atMiddle = (vid & 1u) != 0u;
        float3 p = atMiddle ? 0.5 * (pa + pb) : (fromB ? pb : pa);
        LineVarying out;
        out.position = scene.viewProjection * float4(p, 1.0);
        out.color = float4(mix(organismColor(rgb[fromB ? ij.y : ij.x], fromB ? pb.z : pa.z, scene.world.y, world), float3(1.0), 0.35), 1.0);
        return out;
    }

    // The planet level's optional layer: one translucent point per listed row.
    struct PointVarying {
        float4 position [[position]];
        float pointSize [[point_size]];
        float4 color;
    };

    vertex PointVarying pointVertexA(uint vid [[vertex_id]],
                                     constant Scene& scene [[buffer(0)]],
                                     device const packed_float3* pos [[buffer(2)]],
                                     device const uint* rgb [[buffer(4)]],
                                     constant WorldU& world [[buffer(7)]])
    {
        PointVarying out;
        float3 p = minimumImage(float3(pos[vid]), world.target.xy, world.target.z);
        out.position = scene.viewProjection * float4(p, 1.0);
        out.pointSize = world.style.y;
        out.color = float4(organismColor(rgb[vid], p.z, scene.world.y, world), world.style.x);
        return out;
    }

    fragment float4 pointFragment(PointVarying in [[stage_in]])
    {
        return in.color;
    }

    // The minimap's organisms: the same points in the box of the whole world, east to the right and north up (x / L across, the signed
    // distance from the equator over L up), pale and translucent, so that a crowd shows as a brighter haze (the average of 40,000
    // colours is mud: the minimap says where the life is, the colour mode is the main view's).
    vertex PointVarying insetPointVertex(uint vid [[vertex_id]],
                                         constant Scene& scene [[buffer(0)]],
                                         device const packed_float3* pos [[buffer(2)]],
                                         device const uint* rgb [[buffer(4)]],
                                         constant WorldU& world [[buffer(7)]])
    {
        PointVarying out;
        float L = world.target.z;
        float3 p = float3(pos[vid]);
        float ybar = p.y - L * rint(p.y / L);
        out.position = float4(2.0 * p.x / L - 1.0, 2.0 * ybar / L, 0.0, 1.0);
        out.pointSize = world.style.y;
        out.color = float4(float3(0.96, 0.97, 1.0), world.style.x);
        return out;
    }

    // The map: one patch of one world, L wide around the camera target in x (where the texture repeats, so the wrap is seamless) and L
    // tall in the cell of the wrap in y that the target is in, with its two neighbours across the wrap as copies (instance 0, 1, 2 are the
    // cells k - 1, k, k + 1, `world.patch`); a copy with no alpha collapses. u = x / L and v = y / L; the planes are row-major [cy][cx],
    // row 0 at v = 0. At most one world is ever in view, so no storm is drawn twice.
    struct MapVarying {
        float4 position [[position]];
        float2 uv;
    };

    vertex MapVarying mapVertex(uint vid [[vertex_id]],
                                uint iid [[instance_id]],
                                constant Scene& scene [[buffer(0)]],
                                constant WorldU& world [[buffer(7)]])
    {
        float L = world.target.z;
        MapVarying out;
        out.uv = float2(0.0);
        out.position = float4(2.0, 2.0, 2.0, 1.0);
        float alpha = world.patch[1u + min(iid, 2u)];
        if (iid > 2u || alpha <= 0.0) { return out; }
        float k = world.patch.x - 1.0 + float(iid);
        float2 corner = float2(((vid & 1u) != 0u) ? 1.0 : -1.0, ((vid & 2u) != 0u) ? 1.0 : -1.0);
        float2 xy = float2(world.target.x, k * L) + corner * (0.5 * L);
        out.position = scene.viewProjection * float4(xy, 0.0, 1.0);
        out.uv = xy * world.target.w;
        return out;
    }

    // The minimap: a quad over the viewport it is drawn into. u = x / L runs left to right; v = y / L is the signed distance from the
    // equator over L, +1/2 at the top edge and -1/2 at the bottom (the sampler repeats, so the wrap in y is the box's top and bottom).
    vertex MapVarying insetVertex(uint vid [[vertex_id]])
    {
        float2 corner = float2(((vid & 1u) != 0u) ? 1.0 : -1.0, ((vid & 2u) != 0u) ? 1.0 : -1.0);
        MapVarying out;
        out.position = float4(corner, 0.0, 1.0);
        out.uv = float2(0.5 + 0.5 * corner.x, 0.5 * corner.y);
        return out;
    }

    constexpr sampler planeSampler(coord::normalized, address::repeat, filter::linear);

    // Sea from the warm-pool map, cloud from the depth of the cloud layer, rain as a blue-grey wash, a signed tint where the
    // vorticity has a sign (cyclonic warm, anticyclonic cool), a faint tint of the sea by hemisphere, the two equators, and the
    // territory of the kin groups: borders, and a weak tint of each cell in its owner's colour. Every plane is the mix of the
    // previous and the current delivery by the wall-clock fade. The palette is MapPalette.swift's (a test compares the numbers).
    fragment float4 mapFragment(MapVarying in [[stage_in]],
                                constant WorldU& world [[buffer(7)]],
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
        float2 uv = in.uv;
        float3 color = float3(0.045, 0.055, 0.075);
        if ((world.flags.w & 1u) != 0u) {
            float k = world.params.x;
            float rainDepth = mix(rainPrev.sample(planeSampler, uv).r, rainCur.sample(planeSampler, uv).r, k);
            float cloudDepth = mix(cloudPrev.sample(planeSampler, uv).r, cloudCur.sample(planeSampler, uv).r, k);
            float zeta = mix(zetaPrev.sample(planeSampler, uv).r, zetaCur.sample(planeSampler, uv).r, k);
            float temperature = mix(seaPrev.sample(planeSampler, uv).r, seaCur.sample(planeSampler, uv).r, k);
            float3 sea = mix(float3(0.08, 0.23, 0.42), float3(0.10, 0.46, 0.55), smoothstep(0.1, 0.9, temperature));
            // The hemisphere with f > 0 is 0 < y < L/2: v = y / L, and sin(2 pi v) is positive there.
            float north = smoothstep(-0.2, 0.2, sin(6.2831853 * uv.y));
            sea = mix(sea, mix(float3(0.10, 0.72, 0.66), float3(0.42, 0.36, 0.92), north), 0.03 * world.look.w);
            sea = mix(sea, float3(0.92, 0.62, 0.30), 0.30 * smoothstep(0.0, 0.8, zeta));
            sea = mix(sea, float3(0.32, 0.38, 0.92), 0.30 * smoothstep(0.0, 0.8, -zeta));
            color = mix(sea, float3(0.40, 0.55, 0.82), clamp(1.4 * rainDepth, 0.0, 0.75));
            color = mix(color, float3(0.95, 0.97, 1.00), 0.88 * smoothstep(0.03, 0.75, cloudDepth));
            color = clamp(color * world.look.x, 0.0, 1.0);
        }
        if ((world.flags.w & 4u) != 0u) {
            float g = uv.y * 2.0;
            float f = fract(g);
            float line = 1.0 - smoothstep(0.5, 1.5, min(f, 1.0 - f) / max(fwidth(g), 1e-6));
            color = mix(color, float3(0.75, 0.80, 0.90), 0.35 * line);
        }
        if ((world.flags.w & 2u) != 0u) {
            float2 size = float2(territory.get_width(), territory.get_height());
            float2 g = uv * size;
            g -= size * floor(g / size);
            float2 cellF = floor(g);
            float4 here = territory.read(uint2(min(cellF, size - 1.0)));
            color = mix(color, here.rgb, world.params.z * here.a);
            // A border where the owners of neighbouring cells differ, a pixel wide, on the side the pixel is on.
            float2 f = g - cellF;
            float2 w = fwidth(g);
            float border = 0.0;
            for (int axis = 0; axis < 2; ++axis) {
                float fa = axis == 0 ? f.x : f.y;
                float wa = axis == 0 ? w.x : w.y;
                float side = fa < wa ? -1.0 : (fa > 1.0 - wa ? 1.0 : 0.0);
                if (side != 0.0) {
                    float2 other = cellF + (axis == 0 ? float2(side, 0.0) : float2(0.0, side));
                    other -= size * floor(other / size);
                    float4 there = territory.read(uint2(min(other, size - 1.0)));
                    bool same = all(there.rgb == here.rgb) && ((there.a > 0.0) == (here.a > 0.0));
                    if (!same && (there.a > 0.0 || here.a > 0.0)) { border = 1.0; }
                }
            }
            color = mix(color, float3(1.0), world.look.y * border);
        }
        return float4(mix(background, color, world.params.y), 1.0);
    }
    """
}

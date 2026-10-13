// The viewer's Metal shaders as one string, compiled at run time with `MTLDevice.makeLibrary(source:options:)`.
//
// They are not a `.metal` file because the installed app is the release binary and nothing else (`make-app.sh` copies
// no resource bundle), and because compiling a `.metal` file at build time needs the Metal Toolchain, which is a separate
// download. The price is that the source is first compiled when the viewer opens; the compiler's diagnostics then reach
// the viewer window (`Renderer.Failure.shaders`). Do not use a backslash or a triple quote in the source.
//
// The structs `Scene` and `Plane` mirror `SceneUniforms` and `PlaneUniforms` in Renderer.swift field by field (a test
// pins the Swift side: 160 and 32 bytes).
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
    static let entryPoints = [
        sphereVertex, sphereFragment, bondVertex, boxVertex, lineFragment, planeVertex, planeFragment, sliceFragment,
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

    // The nose of an agent: a small sphere on its forward axis, paler than the body, so the heading shows.
    constant float noseOffset = 0.75;
    constant float noseRadius = 0.5;
    // A sphere about the agent's centre that holds the body and the nose, in units of the agent radius: the farthest point
    // of the nose from the centre is noseOffset + noseRadius.
    constant float boundRadius = 1.25;

    float3 unpackColor(uint c) {
        return float3(float((c >> 16) & 255u), float((c >> 8) & 255u), float(c & 255u)) * (1.0 / 255.0);
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

    fragment SphereOut sphereFragment(SphereVarying in [[stage_in]],
                                      constant Scene& scene [[buffer(0)]])
    {
        SphereOut out;
        out.color = float4(0.0);
        out.depth = 1.0;

        float radius = scene.world.z;
        float3 origin = scene.eye.xyz;
        float3 dir = normalize(in.world - origin);
        float3 noseCenter = in.center + in.forward * (noseOffset * radius);
        float noseR = noseRadius * radius;

        // Ray against the body and against the nose. The squared miss distance is taken from the component of the
        // offset perpendicular to the ray, which keeps its precision when the eye is far away.
        float tBody = -1.0;
        float3 oc = origin - in.center;
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
        float3 normal = useNose ? (hit - noseCenter) / noseR : (hit - in.center) / radius;
        float3 base = useNose ? mix(in.color, float3(1.0), 0.6) : in.color;
        float lambert = max(dot(normal, -dir), 0.0);
        out.color = float4(base * (0.3 + 0.7 * lambert), 1.0);
        float4 clip = scene.viewProjection * float4(hit, 1.0);
        out.depth = clip.z / clip.w;
        return out;
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
    """
}

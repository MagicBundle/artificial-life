import AlifeCore
import simd

@testable import AlifeControl

// A CPU port of the viewer's sphere fragment shader (`sphereFragment` in Shaders.swift), for the GPU test to compare a
// rendered picture with. With the other layers off, the colour of a pixel follows from the agents' positions, headings
// and colours alone, so a picture can be checked pixel by pixel: the order of the colour channels, which half-precision
// value is the yaw and which the pitch, the forward axis, the nose and the headlight all show. Nothing here needs a
// device.

/// One agent as the shader sees it.
struct ReferenceSphere {
    var center: SIMD3<Float>
    /// The unit forward axis `(cos φ cos θ, cos φ sin θ, sin φ)` of its yaw θ and pitch φ.
    var forward: SIMD3<Float>
    /// 0 to 1 per channel.
    var color: SIMD3<Float>
    var radius: Float

    /// The same constants as in the shader source (a test compares them).
    static let noseOffset: Float = 0.75
    static let noseRadius: Float = 0.5

    init(center: SIMD3<Float>, forward: SIMD3<Float>, color: SIMD3<Float>, radius: Float) {
        self.center = center
        self.forward = forward
        self.color = color
        self.radius = radius
    }

    init(frame: StreamFrame, row: Int, radius: Float) {
        center = frame.position(row)
        let (yaw, pitch) = (frame.yaw(row), frame.pitch(row))
        forward = SIMD3(cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), sin(pitch))
        let rgb = frame.rgb[row]
        color = SIMD3(Float((rgb >> 16) & 255), Float((rgb >> 8) & 255), Float(rgb & 255)) / 255
        self.radius = radius
    }

    /// What the pixel whose ray leaves `eye` along the unit vector `dir` shows of this agent: its colour (nil where the ray
    /// misses the agent) and the distance along the ray to the surface seen (infinite for a miss), and whether the pixel
    /// is well inside a region, not within a hair of a silhouette or of the curve where the nose meets the body, where a
    /// different rounding on the GPU could pick the other side.
    func shade(eye: SIMD3<Float>, dir: SIMD3<Float>) -> (color: SIMD3<Float>?, t: Float, steady: Bool) {
        let noseCenter = center + forward * (Self.noseOffset * radius)
        let noseR = Self.noseRadius * radius

        func hit(_ middle: SIMD3<Float>, _ r: Float) -> (t: Float, edge: Float) {
            let oc = eye - middle
            let b = dot(oc, dir)
            let q = oc - dir * b
            let h = r * r - dot(q, q)
            let edge = abs(h) / (r * r)  // 0 on the silhouette, 1 through the middle
            return h >= 0 ? (-b - h.squareRoot(), edge) : (-1, edge)
        }
        let body = hit(center, radius)
        let nose = hit(noseCenter, noseR)
        let hitBody = body.t > 0, hitNose = nose.t > 0
        var steady = body.edge > 0.06 && nose.edge > 0.06
        guard hitBody || hitNose else { return (nil, .infinity, steady) }
        let useNose = hitNose && (!hitBody || nose.t < body.t)
        let t = useNose ? nose.t : body.t
        let point = eye + dir * t
        let normal = useNose ? (point - noseCenter) / noseR : (point - center) / radius
        let base = useNose ? color + (SIMD3<Float>(repeating: 1) - color) * 0.6 : color
        let lambert = max(dot(normal, -dir), 0)
        if hitBody && hitNose, abs(nose.t - body.t) < 0.05 * radius { steady = false }
        return (base * (0.3 + 0.7 * lambert), t, steady)
    }
}

/// The rays of the pixels of a `width` x `height` picture (row 0 at the top) through `camera`, each from the centre of its
/// pixel: where the shader's interpolated billboard position, and so its ray, is.
struct PixelRays {
    let eye: SIMD3<Float>
    let width: Int
    let height: Int
    private let inverse: simd_float4x4

    init(camera: Camera, width: Int, height: Int) {
        eye = camera.position
        self.width = width
        self.height = height
        inverse = camera.viewProjectionMatrix(aspect: Float(width) / Float(height)).inverse
    }

    /// The unit direction of the ray of pixel `(x, y)`.
    func direction(x: Int, y: Int) -> SIMD3<Float> {
        let ndcX = (Float(x) + 0.5) / Float(width) * 2 - 1
        let ndcY = 1 - (Float(y) + 0.5) / Float(height) * 2
        func unproject(_ depth: Float) -> SIMD3<Float> {
            let p = inverse * SIMD4(ndcX, ndcY, depth, 1)
            return SIMD3(p.x, p.y, p.z) / p.w
        }
        return simd_normalize(unproject(1) - unproject(0))
    }
}

/// The pixels to compare around the alive agents: for each, the square of `half` pixels around its centre's projection,
/// cut to the picture, as `(x, y)` pairs once each.
func pixelsAround(_ frame: StreamFrame, camera: Camera, size: Int, half: Int) -> [(x: Int, y: Int)] {
    var wanted = Set<Int>()
    for row in 0..<frame.rows where frame.isAlive(row) {
        guard let ndc = camera.ndc(of: frame.position(row), aspect: 1) else { continue }
        let (cx, cy) = (Int((ndc.x * 0.5 + 0.5) * Float(size)), Int((0.5 - ndc.y * 0.5) * Float(size)))
        let (x0, x1) = (max(cx - half, 0), min(cx + half, size - 1))
        let (y0, y1) = (max(cy - half, 0), min(cy + half, size - 1))
        guard x0 <= x1, y0 <= y1 else { continue }
        for y in y0...y1 {
            for x in x0...x1 { wanted.insert(y * size + x) }
        }
    }
    return wanted.sorted().map { (x: $0 % size, y: $0 / size) }
}

/// What the picture should show at one pixel, from all the agents: the colour of the nearest surface (nil: the
/// background), whether every agent's contribution is steady there, and which agent is nearest.
func referencePixel(_ spheres: [ReferenceSphere], rays: PixelRays, x: Int, y: Int)
    -> (color: SIMD3<Float>?, steady: Bool, nearest: Int?) {
    let dir = rays.direction(x: x, y: y)
    var best: (color: SIMD3<Float>?, t: Float, index: Int?) = (nil, .infinity, nil)
    var second = Float.infinity
    var steady = true
    for (index, sphere) in spheres.enumerated() {
        let shaded = sphere.shade(eye: rays.eye, dir: dir)
        steady = steady && shaded.steady
        guard let color = shaded.color else { continue }
        if shaded.t < best.t {
            second = best.t
            best = (color, shaded.t, index)
        } else {
            second = min(second, shaded.t)
        }
    }
    // Two surfaces at nearly the same distance (bodies that overlap, where they cross): either may win on the GPU.
    if second.isFinite, abs(second - best.t) < 0.05 * (spheres.first?.radius ?? 1) { steady = false }
    return (best.color, steady, best.index)
}

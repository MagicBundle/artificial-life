import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import simd

@testable import AlifeCore

/// A pure rasterisation of the morph's mesh through `Morph.position` (the vertex shader's mirror), with the pinhole camera of the flat
/// close-up (top down, the eye at `(0, 0, R + M d)` looking down the frame's -z), a depth buffer and back-face culling (the mesh's
/// outward winding is counter-clockwise seen from the eye at every `s`; the folded far part of the roll lies behind the sphere and is
/// culled, as the pipeline's `cullMode = .back` will). It measures the three criteria of the spec's 7.1 and draws the strip of 3.2.
/// Geometry only: a checker for the texture, magenta for the clear colour.
struct MorphRaster {
    let L: Float = 320
    let H: Float = 80
    let fovY = Camera.defaultFovY
    let width = 1100
    let height = 800
    /// The mesh step in torus units (320 / 54 = 5.93, "6 units"), `count` vertices per side including both edges.
    let count = 55

    var aspect: Float { Float(width) / Float(height) }
    var R: Float { GlobeMap.radius(L: L) }
    var step: Float { L / Float(count - 1) }

    /// One rendered frame: coverage, depth, the screen position of every vertex (full resolution, y up from the frame centre).
    struct Frame {
        var scale: Float
        var w: Int
        var h: Int
        var depth: [Float]  // +inf where nothing was drawn
        var colour: [UInt32]
        var screen: [SIMD2<Float>]
        var eyeDepth: [Float]
        var s: Float
        var coverage: Float { Float(depth.reduce(0) { $0 + ($1.isFinite ? 1 : 0) }) / Float(max(depth.count, 1)) }
    }

    func vertexOffsets() -> [(e: Float, ybar: Float)] {
        var out: [(Float, Float)] = []
        for i in 0..<count { for j in 0..<count { out.append((-L / 2 + Float(j) * step, -L / 2 + Float(i) * step)) } }
        return out
    }

    func project(_ p: SIMD3<Float>, eye: Float) -> (screen: SIMD2<Float>, depth: Float) {
        let depth = (R + eye) - p.z
        let t = tan(fovY / 2)
        return (SIMD2(p.x / (depth * t * aspect) * Float(width) / 2, p.y / (depth * t) * Float(height) / 2), depth)
    }

    /// Renders the mesh of a view centred on `(x, ybar)` at the distance `d` (torus units), or at the morph scalar `s` when given.
    func render(distance d: Float, x: Float = 0, ybar: Float, heading: Float = 0, scale: Float = 0.25, sOverride: Float? = nil,
                colour: Bool = false) -> Frame {
        let s = sOverride ?? Morph.s(distance: d, L: L, fovY: fovY, aspect: aspect)
        let frame = Morph.Frame(x: x, ybar: ybar, heading: heading, L: L)
        let eye = Morph.eyeDistance(distance: d)
        let offsets = vertexOffsets()
        var screen = [SIMD2<Float>](), depths = [Float]()
        screen.reserveCapacity(offsets.count)
        for (e, yb) in offsets {
            let p = Morph.position(e: e, n: yb - ybar, z: 0, frame: frame, s: s, L: L, height: H)
            let q = project(p, eye: eye)
            screen.append(q.screen)
            depths.append(q.depth)
        }
        let w = Int(Float(width) * scale), h = Int(Float(height) * scale)
        var out = Frame(scale: scale, w: w, h: h, depth: [Float](repeating: .infinity, count: w * h),
                        colour: [UInt32](repeating: 0xFF_FF_00_FF, count: colour ? w * h : 0), screen: screen, eyeDepth: depths, s: s)
        func toImage(_ v: SIMD2<Float>) -> SIMD2<Float> { SIMD2((v.x + Float(width) / 2) * scale, (Float(height) / 2 - v.y) * scale) }
        for i in 0..<(count - 1) {
            for j in 0..<(count - 1) {
                let a = i * count + j, b = a + 1, c = a + count, d2 = c + 1
                for tri in [(a, b, c), (b, d2, c)] {
                    drawTriangle(tri, offsets: offsets, x: x, toImage: toImage, into: &out, colour: colour)
                }
            }
        }
        return out
    }

    private func drawTriangle(_ t: (Int, Int, Int), offsets: [(e: Float, ybar: Float)], x: Float, toImage: (SIMD2<Float>) -> SIMD2<Float>,
                              into f: inout Frame, colour: Bool) {
        let ids = [t.0, t.1, t.2]
        for k in ids where f.eyeDepth[k] <= 0.01 { return }
        let p = ids.map { f.screen[$0] }
        let area = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[1].y - p[0].y) * (p[2].x - p[0].x)
        guard area > 0 else { return }  // back face (y up: counter-clockwise is front)
        let q = p.map(toImage)
        let minX = max(Int(min(q[0].x, q[1].x, q[2].x).rounded(.down)), 0), maxX = min(Int(max(q[0].x, q[1].x, q[2].x).rounded(.up)), f.w - 1)
        let minY = max(Int(min(q[0].y, q[1].y, q[2].y).rounded(.down)), 0), maxY = min(Int(max(q[0].y, q[1].y, q[2].y).rounded(.up)), f.h - 1)
        guard minX <= maxX, minY <= maxY else { return }
        let denom = (q[1].x - q[0].x) * (q[2].y - q[0].y) - (q[1].y - q[0].y) * (q[2].x - q[0].x)
        guard abs(denom) > 1e-9 else { return }
        for py in minY...maxY {
            for px in minX...maxX {
                let c = SIMD2(Float(px) + 0.5, Float(py) + 0.5)
                let l1 = ((c.x - q[0].x) * (q[2].y - q[0].y) - (c.y - q[0].y) * (q[2].x - q[0].x)) / denom
                let l2 = ((q[1].x - q[0].x) * (c.y - q[0].y) - (q[1].y - q[0].y) * (c.x - q[0].x)) / denom
                let l0 = 1 - l1 - l2
                guard l0 >= 0, l1 >= 0, l2 >= 0 else { continue }
                let z = l0 * f.eyeDepth[ids[0]] + l1 * f.eyeDepth[ids[1]] + l2 * f.eyeDepth[ids[2]]
                let at = py * f.w + px
                guard z < f.depth[at] else { continue }
                f.depth[at] = z
                if colour {
                    let u = l0 * (x + offsets[ids[0]].e) + l1 * (x + offsets[ids[1]].e) + l2 * (x + offsets[ids[2]].e)
                    let v = l0 * offsets[ids[0]].ybar + l1 * offsets[ids[1]].ybar + l2 * offsets[ids[2]].ybar
                    let odd = (Int((u / 20).rounded(.down)) + Int(((v + L / 2) / 20).rounded(.down))) & 1 == 1
                    f.colour[at] = odd ? 0xFF_46_8C_AA : 0xFF_28_5A_82  // 0xAARRGGBB
                }
            }
        }
    }

    // MARK: The three criteria of the spec's 7.1

    /// Pixels per torus unit at the centre: the screen distance between `e = -0.5` and `e = +0.5` around it.
    func centreScale(distance d: Float, ybar: Float, heading: Float = 0) -> Float {
        let s = Morph.s(distance: d, L: L, fovY: fovY, aspect: aspect)
        let frame = Morph.Frame(x: 0, ybar: ybar, heading: heading, L: L)
        let eye = Morph.eyeDistance(distance: d)
        let a = project(Morph.position(e: -0.5, n: 0, z: 0, frame: frame, s: s, L: L, height: H), eye: eye).screen
        let b = project(Morph.position(e: 0.5, n: 0, z: 0, frame: frame, s: s, L: L, height: H), eye: eye).screen
        return simd_length(b - a)
    }

    struct Criteria {
        var worstScaleStep: Float  // the largest relative increase of the centre's scale per step (<= 0: non-increasing)
        var maxCoverageRise: Float  // points
        var maxCoverageDrop: Float  // points per step
        var lines98to60: Float  // wheel lines (0.1 of ln d)
        var minCoverageUpTo03: Float  // fraction
        var p99Displacement: Float  // px per step, the largest over the steps
        var steps: Int
    }

    /// Steps of 0.5% of `d`: the centre scale over `[4, 0.8 L]`; the coverage and the displacement over the morph.
    func criteria(ybar: Float, rate: Float = 1.005, scale: Float = 0.25) -> Criteria {
        var worst: Float = -.infinity
        var d: Float = 4
        var last = centreScale(distance: d, ybar: ybar)
        while d < 0.8 * L {
            d *= rate
            let now = centreScale(distance: d, ybar: ybar)
            worst = max(worst, now / last - 1)
            last = now
        }
        let df = Morph.flatLimit * L, dg = Morph.globeDistance(L: L, fovY: fovY, aspect: aspect)
        d = df * 0.98
        var covs: [(d: Float, cov: Float, s: Float)] = []
        var disp: [Float] = []
        var previous: Frame?
        while d < dg * 1.1 {
            let f = render(distance: d, ybar: ybar, scale: scale)
            covs.append((d, f.coverage, f.s))
            if let p = previous { disp.append(p99(displacements(from: p, to: f))) }
            previous = f
            d *= rate
        }
        var rise: Float = 0, drop: Float = 0
        for i in 1..<covs.count {
            rise = max(rise, covs[i].cov - covs[i - 1].cov)
            drop = max(drop, covs[i - 1].cov - covs[i].cov)
        }
        let i98 = covs.firstIndex { $0.cov < 0.98 }, i60 = covs.firstIndex { $0.cov < 0.60 }
        let lines = (i98 != nil && i60 != nil) ? log(covs[i60!].d / covs[i98!].d) / 0.1 : 0
        let upTo = covs.filter { $0.s <= 0.3 }.map(\.cov).min() ?? 1
        return Criteria(worstScaleStep: worst, maxCoverageRise: rise * 100, maxCoverageDrop: drop * 100, lines98to60: lines,
                        minCoverageUpTo03: upTo, p99Displacement: disp.max() ?? 0, steps: covs.count)
    }

    /// The screen displacement of the vertices drawn in both frames (inside the frame, in front of the eye, on the front-most sheet).
    private func displacements(from a: Frame, to b: Frame) -> [Float] {
        func visible(_ f: Frame, _ k: Int) -> Bool {
            let v = f.screen[k]
            guard f.eyeDepth[k] > 0.01, abs(v.x) < Float(width) / 2, abs(v.y) < Float(height) / 2 else { return false }
            let px = Int((v.x + Float(width) / 2) * f.scale), py = Int((Float(height) / 2 - v.y) * f.scale)
            guard px >= 0, px < f.w, py >= 0, py < f.h else { return false }
            let z = f.depth[py * f.w + px]
            return z.isFinite && f.eyeDepth[k] <= z + 4
        }
        var out: [Float] = []
        for k in 0..<a.screen.count where visible(a, k) && visible(b, k) { out.append(simd_length(b.screen[k] - a.screen[k])) }
        return out
    }

    private func p99(_ v: [Float]) -> Float {
        guard v.count > 50 else { return 0 }
        let s = v.sorted()
        return s[Int(Float(s.count - 1) * 0.99)]
    }

    // MARK: Output

    static func writePNG(_ f: Frame, to url: URL, label: String? = nil) throws {
        var bytes = [UInt8](repeating: 0, count: f.w * f.h * 4)
        for i in 0..<(f.w * f.h) {
            let c = f.colour[i]  // 0xAARRGGBB
            bytes[4 * i] = UInt8((c >> 16) & 0xFF)
            bytes[4 * i + 1] = UInt8((c >> 8) & 0xFF)
            bytes[4 * i + 2] = UInt8(c & 0xFF)
            bytes[4 * i + 3] = 255
        }
        try write(bytes: bytes, w: f.w, h: f.h, to: url)
    }

    /// A sheet of tiles in rows of `columns`.
    static func writeSheet(_ frames: [Frame], columns: Int, to url: URL) throws {
        guard let first = frames.first else { return }
        let rows = (frames.count + columns - 1) / columns
        let w = first.w * columns, h = first.h * rows
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        for (n, f) in frames.enumerated() {
            let ox = (n % columns) * f.w, oy = (n / columns) * f.h
            for y in 0..<f.h {
                for x in 0..<f.w {
                    let c = f.colour[y * f.w + x]
                    let o = ((oy + y) * w + ox + x) * 4
                    bytes[o] = UInt8((c >> 16) & 0xFF)
                    bytes[o + 1] = UInt8((c >> 8) & 0xFF)
                    bytes[o + 2] = UInt8(c & 0xFF)
                    bytes[o + 3] = 255
                }
            }
        }
        try write(bytes: bytes, w: w, h: h, to: url)
    }

    private static func write(bytes: [UInt8], w: Int, h: Int, to url: URL) throws {
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }
}

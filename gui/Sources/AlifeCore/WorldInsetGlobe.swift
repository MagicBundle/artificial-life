import CoreGraphics
import Foundation
import simd

/// The minimap's footprint at the globe and in the morph (DESIGN §36.10, the spec's 4.1): the part of the torus the view shows.
///
/// * At `s >= WorldVantage.capFrom` it is the **visible cap**, the circle of angular radius `acos(R / |eye|)` about the direction
///   under the eye (`E = M d` above the surface, 69.5 degrees at Home), 96 samples mapped by `GlobeMap`. A cap that holds a pole is not a
///   closed curve of the torus: it spans every column, and is the polygon between the boundary and the box's top (or bottom) edge.
/// * Between the flat view and the cap it is the border of the picture marched in onto the surface (a border ray that misses the
///   planet is shortened until it lands), as points of the box unwrapped about the centre, the way the flat footprint is.
extension WorldInset {
    /// Samples of the cap's circle.
    package static let capSamples = 96
    /// Border samples and bisection steps of the morph's footprint.
    package static let marchSamples = 32
    package static let marchSteps = 7

    /// The pieces to draw (each a closed polygon in the box's points, the cut across the box's edges done by the caller's clip),
    /// nil when the view is the flat map (`s = 0`, no globe): the caller then draws `footprint(of:)` as before.
    package func footprintPieces(of vantage: WorldVantage) -> [[CGPoint]]? {
        guard vantage.isGlobe, vantage.s > 0, size > 0 else { return nil }
        if vantage.s >= WorldVantage.capFrom { return cap(of: vantage) }
        return march(of: vantage)
    }

    /// The point of the box for the centre of the view at the globe (the looked-at point), unwrapped like the footprint.
    package func centre(of vantage: WorldVantage) -> CGPoint {
        point(x: vantage.frame.x, ybar: vantage.frame.ybar)
    }

    // MARK: The cap

    /// The unit direction under the eye and the angular radius of the visible cap, in world-fixed axes.
    package func visibleCap(of vantage: WorldVantage) -> (direction: SIMD3<Float>, radius: Float) {
        let frame = vantage.frame
        let eye = vantage.camera.position
        let global = eye.x * frame.right + eye.y * frame.up + eye.z * frame.centre
        let length = simd_length(global)
        let R = GlobeMap.radius(L: size)
        guard length > R * 1.0001 else { return (frame.centre, 0) }
        return (global / length, acos(min(R / length, 1)))
    }

    private func cap(of vantage: WorldVantage) -> [[CGPoint]] {
        let (g, psi) = visibleCap(of: vantage)
        guard psi > 1e-4 else { return [] }
        let helper: SIMD3<Float> = abs(g.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)
        let a = simd_normalize(simd_cross(g, helper)), b = simd_cross(g, a)
        let L = size
        // (longitude, signed row) of each sample of the circle; a row inside a pole cap is held at the seam row.
        var samples: [(x: Float, ybar: Float)] = []
        for i in 0..<Self.capSamples {
            let t = 2 * Float.pi * Float(i) / Float(Self.capSamples)
            let d = cos(psi) * g + sin(psi) * (cos(t) * a + sin(t) * b)
            let lat = asin(min(max(d.z, -1), 1))
            let ybar = abs(lat) < GlobeMap.phiMax ? GlobeMap.ybar(latitude: lat, L: L) : (lat < 0 ? -L / 2 : L / 2)
            samples.append((Torus.wrap(L * atan2(d.y, d.x) / (2 * .pi), size: L), ybar))
        }
        let north = g.z >= cos(psi), south = -g.z >= cos(psi)
        if north || south {
            let edge: Float = north ? L / 2 : -L / 2
            let sorted = samples.sorted { $0.x < $1.x }
            // the boundary, three worlds wide so that the polygon covers the box with nothing to join at the wrap
            var polygon: [CGPoint] = []
            for k: Float in [-1, 0, 1] {
                for p in sorted { polygon.append(point(x: p.x + k * L, ybar: p.ybar)) }
            }
            polygon.append(point(x: 2 * L, ybar: edge))
            polygon.append(point(x: -L, ybar: edge))
            return [polygon]
        }
        let xc = Torus.wrap(L * atan2(g.y, g.x) / (2 * .pi), size: L)
        let polygon = samples.map { point(x: xc + Torus.wrapDelta($0.x - xc, size: L), ybar: $0.ybar) }
        return pieces(of: polygon)
    }

    // MARK: The morph

    private func march(of vantage: WorldVantage) -> [[CGPoint]] {
        let perSide = Self.marchSamples / 4
        var polygon: [CGPoint] = []
        for side in 0..<4 {
            for i in 0..<perSide {
                let t = -1 + 2 * Float(i) / Float(perSide)
                let ndc: SIMD2<Float>
                switch side {
                case 0: ndc = SIMD2(t, 1)
                case 1: ndc = SIMD2(1, -t)
                case 2: ndc = SIMD2(-t, -1)
                default: ndc = SIMD2(-1, t)
                }
                var hit = vantage.pick(at: ndc)
                if hit == nil {
                    var lo: Float = 0, hi: Float = 1
                    for _ in 0..<Self.marchSteps {
                        let mid = (lo + hi) / 2
                        if let h = vantage.pick(at: ndc * mid) {
                            hit = h
                            lo = mid
                        } else {
                            hi = mid
                        }
                    }
                }
                guard let hit else { continue }
                polygon.append(point(x: vantage.frame.x + hit.e, ybar: vantage.frame.ybar + hit.n))
            }
        }
        guard polygon.count >= 3 else { return [] }
        return pieces(of: polygon)
    }
}

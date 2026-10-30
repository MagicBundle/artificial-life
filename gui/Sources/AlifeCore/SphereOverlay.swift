import Foundation
import simd

/// A line of the sphere's overlay: two planet-frame points, a width in pixels and a colour at each end (premultiplied by the shader).
package struct SphereLine: Equatable, Sendable {
    package var a: SIMD3<Float>
    package var b: SIMD3<Float>
    package var widthA: Float
    package var widthB: Float
    package var colorA: SIMD4<Float>
    package var colorB: SIMD4<Float>

    package init(a: SIMD3<Float>, b: SIMD3<Float>, width: Float, color: SIMD4<Float>) {
        self.init(a: a, b: b, widthA: width, widthB: width, colorA: color, colorB: color)
    }

    package init(a: SIMD3<Float>, b: SIMD3<Float>, widthA: Float, widthB: Float, colorA: SIMD4<Float>, colorB: SIMD4<Float>) {
        self.a = a
        self.b = b
        self.widthA = widthA
        self.widthB = widthB
        self.colorA = colorA
        self.colorB = colorB
    }
}

/// What is drawn over the planet as lines (DESIGN §46.8): the storms (a ring of the storm's radius on the floor, the track behind it and the
/// predicted path), the warm pools, the wind as streamlines, the bonds, the selection. Pure geometry, so that the shapes are tested; the
/// renderer only turns `SphereLine`s into quads.
package enum SphereOverlay {
    // MARK: Colours

    package static let stormNamed = SIMD4<Float>(1.0, 0.93, 0.70, 0.95)
    package static let stormDisturbance = SIMD4<Float>(0.80, 0.86, 0.95, 0.65)
    package static let stormSelected = SIMD4<Float>(1.0, 0.86, 0.25, 1.0)
    package static let pool = SIMD4<Float>(0.95, 0.60, 0.30, 0.60)
    package static let wind = SIMD4<Float>(0.86, 0.94, 1.0, 0.42)
    package static let selection = SIMD4<Float>(1.0, 0.95, 0.55, 1.0)

    // MARK: Circles and arcs on the sphere

    /// `count` points of the small circle of angular radius `angle` (radians) about the unit vector `centre`, at radius `radius`, closed
    /// (the last point is the first).
    package static func circle(centre: SIMD3<Float>, angle: Float, radius: Float, count: Int = 48) -> [SIMD3<Float>] {
        let b = SphereFrame.basis(at: centre)
        let c = cos(angle), s = sin(angle)
        return (0...count).map { k in
            let phi: Float = 2 * Float.pi * Float(k % count) / Float(count)
            let ring: SIMD3<Float> = cos(phi) * b.east + sin(phi) * b.north
            let on: SIMD3<Float> = c * b.up + s * ring
            return simd_normalize(on) * radius
        }
    }

    /// The segments joining consecutive points of a polyline, widths and colours interpolated from `first` to `last`.
    package static func strip(_ points: [SIMD3<Float>], widthFrom: Float, widthTo: Float, colorFrom: SIMD4<Float>, colorTo: SIMD4<Float>)
        -> [SphereLine] {
        guard points.count >= 2 else { return [] }
        let n = Float(points.count - 1)
        return (0..<(points.count - 1)).map { k in
            let t0 = Float(k) / n, t1 = Float(k + 1) / n
            return SphereLine(a: points[k], b: points[k + 1], widthA: widthFrom + (widthTo - widthFrom) * t0,
                              widthB: widthFrom + (widthTo - widthFrom) * t1, colorA: colorFrom + (colorTo - colorFrom) * t0,
                              colorB: colorFrom + (colorTo - colorFrom) * t1)
        }
    }

    // MARK: Storms and pools

    /// The direction of a storm's centre (`x, y, z` of `STRM`); nil for a record without `z`.
    package static func direction(of storm: PlayStorms.Storm) -> SIMD3<Float>? {
        guard let z = storm.z else { return nil }
        let v = SIMD3<Float>(Float(storm.x), Float(storm.y), Float(z))
        return simd_length(v) > 1e-6 ? simd_normalize(v) : nil
    }

    /// The ring of a storm at the floor (lifted a little so that it sits on the planet), its track and its predicted path. A named storm is
    /// warm white, a disturbance paler and thinner; the selected one is gold and thicker.
    package static func storms(_ storms: [PlayStorms.Storm], sphere: SphereHeader, selected: Int?) -> [SphereLine] {
        let rp = Float(sphere.planetRadius)
        let lift = rp + 0.35
        var lines: [SphereLine] = []
        for storm in storms {
            guard let centre = direction(of: storm) else { continue }
            let isSelected = storm.id == selected
            let base = isSelected ? stormSelected : (storm.named ? stormNamed : stormDisturbance)
            let colour = storm.state == "coasting" ? SIMD4(base.x, base.y, base.z, base.w * 0.6) : base
            let width: Float = isSelected ? 3 : (storm.named ? 2 : 1.4)
            let angle = min(max(Float(storm.r) / rp, 0.02), 1.2)
            lines += strip(circle(centre: centre, angle: angle, radius: lift), widthFrom: width, widthTo: width, colorFrom: colour, colorTo: colour)
            // the track: oldest first, fading in towards the storm
            let track = storm.track.compactMap { p -> SIMD3<Float>? in
                p.count >= 4 ? simd_normalize(SIMD3(Float(p[1]), Float(p[2]), Float(p[3]))) * lift : nil
            }
            lines += strip(track, widthFrom: 0.8, widthTo: width, colorFrom: SIMD4(colour.x, colour.y, colour.z, 0.05), colorTo: colour)
            // the predicted path from the storm on, dimmer
            let ahead = storm.pred.compactMap { p -> SIMD3<Float>? in
                p.count >= 4 ? simd_normalize(SIMD3(Float(p[1]), Float(p[2]), Float(p[3]))) * lift : nil
            }
            if !ahead.isEmpty {
                lines += strip([centre * lift] + ahead, widthFrom: width, widthTo: 1, colorFrom: SIMD4(colour.x, colour.y, colour.z, colour.w * 0.7),
                               colorTo: SIMD4(colour.x, colour.y, colour.z, 0.1))
            }
        }
        return lines
    }

    /// The warm pools of the weather: an ellipse of the pool's widths at the floor (a ring of the mean width).
    package static func pools(_ pools: [PlayEnvironment.Pool], sphere: SphereHeader, selected: Int?) -> [SphereLine] {
        let rp = Float(sphere.planetRadius)
        var lines: [SphereLine] = []
        for (i, pool) in pools.enumerated() {
            guard let z = pool.z else { continue }
            let v = SIMD3<Float>(Float(pool.x), Float(pool.y), Float(z))
            guard simd_length(v) > 1e-6 else { continue }
            let colour = i == selected ? stormSelected : Self.pool
            let angle = min(max(Float(pool.sx + pool.sy) * 0.5 / rp, 0.02), 1.0)
            lines += strip(circle(centre: simd_normalize(v), angle: angle, radius: rp + 0.25, count: 32), widthFrom: 1.2, widthTo: 1.2,
                           colorFrom: colour, colorTo: colour)
        }
        return lines
    }

    // MARK: The wind

    /// Streamlines of the wind of `layer` (0 or 1): `seeds` starting points spread evenly over the sphere (a Fibonacci spiral), each
    /// followed `steps` steps along the wind of the cell it is in. The step is the wind's speed in units of `uRef`, `stepAngle` radians at
    /// full speed; a seed in a calm cell (under 6% of full speed) draws nothing. The lines sit at `lift` above the floor.
    package static func streamlines(_ map: PlaySphereWeather, sphere: SphereHeader, layer: Int, seeds: Int = 700, steps: Int = 7,
                                    stepAngle: Float = 0.03, lift: Float) -> [SphereLine] {
        let rp = Float(sphere.planetRadius)
        let radius = rp + lift
        let golden: Float = Float.pi * (3 - Float(5).squareRoot())
        var lines: [SphereLine] = []
        for k in 0..<seeds {
            let z: Float = 1 - 2 * (Float(k) + 0.5) / Float(seeds)
            let r: Float = (1 - z * z).squareRoot()
            let phi: Float = golden * Float(k)
            var p = SIMD3<Float>(r * cos(phi), r * sin(phi), z)
            var points: [SIMD3<Float>] = []
            var strengths: [Float] = []
            for _ in 0...steps {
                let pix = Healpix.pixelFloat(of: p, nside: map.nside)
                let w = map.wind(layer: layer, pixel: pix)
                let speed = simd_length(w) / max(map.uRef, 1e-9)
                points.append(p * radius)
                strengths.append(speed)
                if points.count == 1, speed < 0.06 { break }
                let b = SphereFrame.basis(at: p)
                var direction = SIMD3<Float>.zero
                if speed > 1e-6 {
                    let along: SIMD3<Float> = b.east * w.x + b.north * w.y
                    direction = along / simd_length(w)
                }
                let advance: Float = speed * stepAngle
                p = simd_normalize(p + direction * advance)
            }
            guard points.count >= 3 else { continue }
            let strength = min(strengths.max() ?? 0, 1)
            let head = SIMD4(wind.x, wind.y, wind.z, wind.w * (0.35 + 0.65 * strength))
            lines += strip(points, widthFrom: 0.5, widthTo: 1.2, colorFrom: SIMD4(head.x, head.y, head.z, 0), colorTo: head)
        }
        return lines
    }

    // MARK: The selection

    /// A ring about a point, facing the eye: `radius` world units, `count` segments (the marker round a followed or selected organism).
    package static func marker(at point: SIMD3<Float>, eye: SIMD3<Float>, radius: Float, count: Int = 28) -> [SphereLine] {
        let view = simd_normalize(eye - point)
        let helper: SIMD3<Float> = abs(view.z) > 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 0, 1)
        let u = simd_normalize(simd_cross(view, helper)), v = simd_cross(view, u)
        let points = (0...count).map { k -> SIMD3<Float> in
            let phi: Float = 2 * Float.pi * Float(k % count) / Float(count)
            let offset: SIMD3<Float> = cos(phi) * u + sin(phi) * v
            return point + radius * offset
        }
        return strip(points, widthFrom: 2, widthTo: 2, colorFrom: selection, colorTo: selection)
    }

    // MARK: Height cue

    /// A thin line from each organism aloft (above `minHeight` over the floor) straight down to the floor under it: the depth cue that the
    /// floor alone does not give (an organism 10 units up reads as hovering beside the limb, not as being above it). At most `cap` lines,
    /// the highest first. Pale at the organism, fainter at the floor.
    package static func drops(_ frame: PlayFrame, planetRadius: Float, minHeight: Float = 2, cap: Int = 2500) -> [SphereLine] {
        var aloft: [(height: Float, row: Int)] = []
        for i in 0..<frame.n {
            let p = frame.position(i)
            let h = simd_length(p) - planetRadius
            if h > minHeight { aloft.append((h, i)) }
        }
        if aloft.count > cap { aloft = Array(aloft.sorted { $0.height > $1.height }.prefix(cap)) }
        return aloft.map { entry in
            let p = frame.position(entry.row)
            let foot = simd_normalize(p) * planetRadius
            return SphereLine(a: p, b: foot, widthA: 0.8, widthB: 0.6, colorA: SIMD4(1, 1, 1, 0.30), colorB: SIMD4(1, 1, 1, 0.08))
        }
    }

    // MARK: Bonds

    /// The chords of the bonds of a frame; with the bond flow, each half takes the colour of what it does (`BondFlowLook`).
    package static func bonds(_ frame: PlayFrame, flow: Bool) -> [SphereLine] {
        (0..<frame.nBonds).map { k in
            let (i, j) = frame.bond(k)
            let pi = frame.position(i), pj = frame.position(j)
            var ca = rgba(frame.rgb[i], alpha: 0.85), cb = rgba(frame.rgb[j], alpha: 0.85)
            if flow, let f = frame.flow, k < f.count, f[k] != 0 {
                let s = BondFlowLook.strength(f[k])
                // a positive flow went from the lower listed row to the higher: the lower is the donor
                let donor = f[k] > 0 ? BondFlowLook.donor : BondFlowLook.receiver
                let receiver = f[k] > 0 ? BondFlowLook.receiver : BondFlowLook.donor
                ca = SIMD4(simd_mix(SIMD3(ca.x, ca.y, ca.z), donor, SIMD3(repeating: s)), 0.95)
                cb = SIMD4(simd_mix(SIMD3(cb.x, cb.y, cb.z), receiver, SIMD3(repeating: s)), 0.95)
            }
            return SphereLine(a: pi, b: pj, widthA: 1.3, widthB: 1.3, colorA: ca, colorB: cb)
        }
    }

    private static func rgba(_ packed: UInt32, alpha: Float) -> SIMD4<Float> {
        SIMD4(Float((packed >> 16) & 255) / 255, Float((packed >> 8) & 255) / 255, Float(packed & 255) / 255, alpha)
    }
}

import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// The grab, the zoom and the pan through the regime solver (DESIGN §36.10; spec 5.1 and 7.1 `WorldViewGestureTests`), and the camera
/// controls on the globe.
@Suite struct WorldVantageGestureTests {
    private typealias F = VantageFixture
    private let L = F.L
    /// The pointer's radius from the middle of the view, in half heights (the disc of Home is 0.8): 75% of the disc.
    static let pointerRadius: Float = 0.6

    /// One drag of the pointer from `ndc0` by `delta` points: the pixels by which the anchor misses the pointer afterwards (nil when the
    /// drag does not apply: the pointer is off the planet before or after, or the anchor is a cap, or hidden).
    private func drag(_ v: WorldVantage, from ndc0: SIMD2<Float>, by delta: SIMD2<Float>) -> (miss: Float, after: WorldVantage)? {
        guard let first = v.pick(at: ndc0), !first.seam, let anchor = v.anchor(at: ndc0) else { return nil }
        let ndc1 = ndc0 + SIMD2(delta.x * 2 / Float(F.view.width), -delta.y * 2 / Float(F.view.height))
        // The domain of the claim: pointers within 75% of the disc's radius (a pointer at the limb may have no surface once the centre moves)
        guard hypot(ndc1.x * F.aspect, ndc1.y) < Self.pointerRadius, let there = v.pick(at: ndc1), !there.seam else { return nil }
        var after = v
        guard after.grab(anchor: anchor, at: ndc1) else { return nil }
        guard let at = after.screen(SIMD3(anchor.x, anchor.y, anchor.z), in: F.view) else { return nil }
        let want = F.point(ndc1)
        return (Float(hypot(at.x - want.x, at.y - want.y)), after)
    }

    private func misses(s: Float, ybar: Float, count: Int, seed: UInt64, longest: Float = 400, x: Float = 118, tilt: Float = 0,
                        heading: Float = 0, stalled: UnsafeMutablePointer<[Float]>? = nil) -> [Float] {
        var rng = VantageRng(state: seed)
        var out: [Float] = []
        let v = F.make(s: s, ybar: ybar, x: x, heading: heading, tilt: tilt)
        var attempts = 0
        while out.count < count, attempts < count * 12 {
            attempts += 1
            let p0 = SIMD2(rng.range(-0.45, 0.45), rng.range(-0.55, 0.55))
            let angle = rng.range(0, 2 * .pi), length = rng.range(20, longest)
            if let result = drag(v, from: p0, by: SIMD2(length * cos(angle), length * sin(angle))) {
                out.append(result.miss)
                // the centre of a drag whose answer lies beyond the pole row stops at the pole row (no cap under the centre before s = 0.9)
                if abs(result.after.frame.ybar) >= L / 2 - 1 { stalled?.pointee.append(result.miss) }
            }
        }
        return out
    }

    // MARK: Exactness where it exists

    @Test func atSZeroTheGrabIsTheExactTranslationAcrossTheSeamsAndOverAPole() {
        for ybar in [Float(0), 38, 80, 120, 150] {
            for x in [Float(118), 4, 316] {
                let m = misses(s: 0, ybar: ybar, count: 25, seed: UInt64(ybar) + 1, x: x)
                #expect(m.count >= 20)
                #expect((m.max() ?? 0) < 0.5, "ybar \(ybar), x \(x): worst \(m.max() ?? 0) px")
            }
        }
    }

    @Test func atSOneTheGrabIsTheExactRotationAtEveryLatitude() {
        for ybar in [Float(0), 38, 80, 120, 150] {
            let m = misses(s: 1, ybar: ybar, count: 40, seed: UInt64(ybar) + 17)
            #expect(m.count >= 30, "ybar \(ybar): \(m.count) drags")
            #expect((m.max() ?? 0) < 0.5, "ybar \(ybar): worst \(m.max() ?? 0) px")
        }
    }

    @Test func theRotationIsExactOnATiltedAndTurnedView() {
        for (tilt, heading) in [(Float(0.6), Float(0.9)), (1.2, -2.0)] {
            let m = misses(s: 1, ybar: 38, count: 30, seed: 5, longest: 250, tilt: tilt, heading: heading)
            #expect(m.count >= 15 && (m.max() ?? 0) < 0.5, "tilt \(tilt): \(m.count) drags, worst \(m.max() ?? 0) px")
        }
    }

    @Test func aGrabOverThePoleKeepsTurningAndTheCentreContinuesOnTheAntipodalMeridian() {
        var v = F.make(s: 1, ybar: 150, x: 40)  // latitude 83 degrees
        let before = v.frame
        // Take the point above the centre (toward the pole) and pull it down past the middle: the pole passes the centre.
        let anchor = v.anchor(at: SIMD2(0, 0.45))!
        let ok = v.grab(anchor: anchor, at: SIMD2(0, -0.55))
        #expect(ok)
        let after = v.frame
        let jump = abs(Torus.wrapDelta(after.x - before.x, size: L))
        #expect(jump > L / 4, "the longitude moved by \(jump) of \(L / 2)")
        #expect(simd_dot(after.centre, before.centre) > 0.2, "still near the pole")
        // exact: the anchor is under the pointer, by direction
        let hit = v.pick(at: SIMD2(0, -0.55))!
        #expect(simd_length(hit.direction - anchor.direction) < 1e-4, "\(hit.direction) \(anchor.direction)")
    }

    @Test func theHeadingIsCarriedByTheGrabAsParallelTransport() {
        // Along the equator the heading does not drift; at high latitude it does (the first draft's north up has no solution there).
        var v = F.make(s: 1, ybar: 0)
        let a = v.anchor(at: SIMD2(-0.3, 0))!
        let okA = v.grab(anchor: a, at: SIMD2(0.3, 0))
        #expect(okA)
        #expect(abs(v.pose.heading) < 1e-4 && abs(v.pose.ybar(size: L)) < 0.01, "heading \(v.pose.heading)")
        var high = F.make(s: 1, ybar: 120)
        let b = high.anchor(at: SIMD2(-0.5, -0.1))!
        let okB = high.grab(anchor: b, at: SIMD2(0.5, -0.1))
        #expect(okB)
        #expect(abs(high.pose.heading) > 0.05, "a drag along a parallel at 80 degrees turns the map: \(high.pose.heading)")
    }

    @Test func theZoomKeepsThePointUnderThePointerAtSZeroAndSOne() {
        var rng = VantageRng(state: 23)
        for (s, factors) in [(Float(0), [Float(0.5), 2]), (1, [Float(0.7), 1.6])] {
            for ybar in [Float(0), 38, 80] {
                for factor in factors {
                    var worst: Float = 0, n = 0
                    for _ in 0..<20 {
                        var v = F.make(s: s, ybar: ybar, distance: s == 0 ? 15 : Morph.globeDistance(L: L, fovY: F.fov, aspect: F.aspect) * 1.5)
                        // (a pointer must stay inside the smaller disc a zoom out leaves: 0.4 of a half height at 1.6 x the Home distance)
                        let ndc = SIMD2(rng.range(-0.15, 0.15), rng.range(-0.15, 0.15))
                        guard let hit = v.pick(at: ndc), !hit.seam, let anchor = v.anchor(at: ndc) else { continue }
                        v.zoom(by: factor, at: ndc)
                        guard let at = v.screen(SIMD3(anchor.x, anchor.y, anchor.z), in: F.view) else { continue }
                        let want = F.point(ndc)
                        let err = Float(hypot(at.x - want.x, at.y - want.y))
                        worst = max(worst, err)
                        n += 1
                        #expect((s == 0) == (v.s == 0) && (s == 1) == (v.s == 1), "stays in its regime: s \(v.s)")
                    }
                    #expect(n >= 12 && worst < 0.5, "s \(s) ybar \(ybar) factor \(factor): \(n) zooms, worst \(worst) px")
                }
            }
        }
    }

    @Test func theZoomIsClampedToTheGlobesLimitAndTheNearest() {
        var v = F.make(s: 1, ybar: 0)
        v.zoom(by: 100, at: nil)
        #expect(abs(v.pose.distance - Morph.maxDistanceFraction * L) < 1e-3)
        v.zoom(by: 1e-6, at: SIMD2(0.1, 0.1))
        #expect(v.pose.distance == WorldLevels.minDistance)
    }

    // MARK: The gate: the band 0.5 < s < 0.9

    @Test func theBlendedBandResidualIsMeasuredAndBounded() {
        // The gate of PR B step 6 (the spec's 5.1): the residual of the blend of the translation and the rotation was not measured by the
        // first draft. 40 drags of 20 to 400 points per cell at ybar_c = 0, 38, 80, 120, 150 (latitudes 0, 39, 70, 79, 84 degrees), with
        // the pointer inside 75% of the disc, in the pixels the anchor misses the pointer after the grab (up to 5 steps of the solver).
        var table: [String] = []
        var low: [Float] = []  // ybar_c <= 38
        var mid: [Float] = []  // ybar_c = 80
        var high: [Float] = []  // 120, 150
        var offPole: [Float] = []  // the residuals above 1 px that did not stop at the pole row
        for s in [Float(0.5), 0.6, 0.7, 0.8, 0.9] {
            var cells: [String] = []
            for ybar in [Float(0), 38, 80, 120, 150] {
                var stalled: [Float] = []
                let m = misses(s: s, ybar: ybar, count: 40, seed: UInt64(ybar * 3 + s * 100) + 1, stalled: &stalled).sorted()
                guard !m.isEmpty else { continue }
                let p95 = m[min(m.count - 1, Int(Float(m.count) * 0.95))]
                cells.append(String(format: "%.1f/%.1f/%.1f", m[m.count / 2], p95, m.last!))
                if ybar <= 38 { low += m } else if ybar == 80 { mid += m } else { high += m }
                offPole += m.filter { $0 > 1 }.filter { miss in !stalled.contains(miss) }
            }
            table.append("s=\(s): " + cells.joined(separator: "  "))
        }
        print("GATE grab residual px p50/p95/max at ybar_c 0 38 80 120 150\n" + table.joined(separator: "\n"))
        func percentile(_ values: [Float], _ q: Float) -> Float {
            let sorted = values.sorted()
            return sorted.isEmpty ? 0 : sorted[min(sorted.count - 1, Int(Float(sorted.count) * q))]
        }
        print("GATE summary: lat<=39: p95 \(percentile(low, 0.95)) max \(low.max() ?? 0); lat 70: p95 \(percentile(mid, 0.95)) max \(mid.max() ?? 0); lat 79-84: p50 \(percentile(high, 0.5)) p95 \(percentile(high, 0.95)) max \(high.max() ?? 0); off-pole misses over 1 px: \(offPole.count)")
        // exact (the pixel) up to 39 degrees; 70 degrees within a few pixels at the 95th percentile; the median exact everywhere; the large
        // misses are the drags that the pole row stops
        #expect((low.max() ?? 0) < Self.bandBounds.low, "lat <= 39: \(low.max() ?? 0) px")
        #expect(percentile(mid, 0.95) < Self.bandBounds.midP95, "lat 70: p95 \(percentile(mid, 0.95)) px")
        #expect(percentile(high, 0.5) < 0.5 && percentile(high, 0.95) < Self.bandBounds.highP95, "lat 79-84: p95 \(percentile(high, 0.95)) px")
        #expect(offPole.count <= Self.bandBounds.offPole, "misses over a pixel that the pole did not cause: \(offPole.count)")
    }

    /// The test's bounds for the band, from the measurement of the gate (the handover and DESIGN §36.10), 200 drags per latitude: exact
    /// (under 0.5 px, measured 0.004) up to 39 degrees of latitude; at 70 degrees p95 under 2 px (measured 0.002); at 79 to 84 degrees p50
    /// exact, p95 under 15 px (measured 9.5). The worst drags (46 px at 70 degrees, 272 px at 84) are pointers near the edge of the
    /// morphing surface, or answers beyond the pole row, where no pose holds the anchor; 10 of 1000 drags leave more than a pixel.
    static let bandBounds = (low: Float(0.5), midP95: Float(2), highP95: Float(15), offPole: 15)

    // MARK: Pan

    @Test func aPanFollowsTheFingersAtTheGlobeAndTheMorph() {
        for s in [Float(0.3), 0.7, 1] {
            var v = F.make(s: s, ybar: 30)
            guard let middle = v.anchor(at: .zero) else { Issue.record("no surface at the middle"); continue }
            v.pan(dx: 60, dy: -40, view: F.view)
            // the point that was in the middle is now 60 points to the right and 40 up
            guard let at = v.screen(SIMD3(middle.x, middle.y, middle.z), in: F.view) else { Issue.record("hidden"); continue }
            #expect(abs(Float(at.x) - Float(F.view.width / 2) - 60) < 0.6 && abs(Float(at.y) - Float(F.view.height / 2) + 40) < 0.6,
                    "s \(s): \(at)")
        }
    }

    @Test func aPanOfAViewThatLooksAlongTheSurfaceMovesTheCentreAtAnySAndStaysFinite() {
        for s in [Float(0.3), 1] {
            var v = F.make(s: s, ybar: 30, tilt: Float.pi / 2)
            let before = v.centreDirection
            v.pan(dx: 80, dy: 50, view: F.view)
            #expect(simd_length(v.centreDirection - before) > 1e-3 && v.pose.x.isFinite && v.pose.y.isFinite)
        }
    }

    // MARK: The camera controls on the globe

    @Test func theTiltOrbitsTheLookedAtPointOfThePlanetAtEverySAndTheTopLooksStraightAtIt() {
        for s in [Float(0), 0.4, 0.8, 1] {
            for tilt in [Float(0), 0.5, 1.0, Float.pi / 2] {
                let v = F.make(s: s, ybar: 20, tilt: tilt)
                let cam = v.camera
                // the eye is E from the point looked at, the view axis is tilted by tilt from straight down
                #expect(abs(simd_length(cam.position - cam.target) - Morph.eyeDistance(distance: v.pose.distance)) < 1e-2)
                #expect(abs(acos(min(1, -cam.forward.z)) - tilt) < 1e-4, "s \(s) tilt \(tilt)")
                // the point looked at stays at the middle of the screen
                let middle = v.screen(e: 0, n: 0, z: v.pose.lookZ, in: F.view)
                #expect(middle != nil && abs(Float(middle!.point.x) - 550) < 0.5 && abs(Float(middle!.point.y) - 400) < 0.5, "s \(s) tilt \(tilt)")
            }
        }
        // Top at the globe: the view axis goes through the sphere's centre.
        let top = F.make(s: 1, ybar: 20, tilt: 0)
        let toCentre = simd_normalize(-top.camera.position)
        #expect(simd_length(toCentre - top.camera.forward) < 1e-5)
        #expect(top.pick(at: .zero) != nil)
    }

    @Test func theSideViewOfTheGlobeShowsTheLimbWithTheSkyAboveIt() {
        let v = F.make(s: 1, ybar: 0, tilt: Float.pi / 2)
        #expect(v.pick(at: SIMD2(0, -0.8)) != nil, "the planet fills the lower half")
        #expect(v.pick(at: SIMD2(0, 0.6)) == nil, "the sky above the limb")
        // the limb: scanning up the middle column the planet ends where the view axis grazes it (E = D - R puts the tangent through the axis)
        var top: Float = -1
        for i in 0..<400 {
            let y = -1 + Float(i) / 199
            if v.pick(at: SIMD2(0, y)) != nil { top = max(top, y) }
        }
        #expect(top > -0.2 && top < 0.3, "the limb is near the middle of the screen: ndc \(top)")
        // The point looked at is still the middle of the screen, and a pick at the horizon of a morphing side view is finite.
        let morph = F.make(s: 0.5, ybar: 0, tilt: Float.pi / 2)
        for y in stride(from: Float(-0.95), through: 0.95, by: 0.05) {
            if let hit = morph.pick(at: SIMD2(0.1, y)) { #expect(hit.x.isFinite && hit.y.isFinite && hit.depth > 0) }
        }
    }

    @Test func theHeadingTurnsTheMapAboutTheViewAxisAtEveryS() {
        for s in [Float(0), 0.5, 1] {
            let v0 = F.make(s: s, ybar: 20)
            let v1 = F.make(s: s, ybar: 20, heading: 1.1)
            let probe = SIMD3<Float>(118 + 20, 28, 0)
            let a = v0.screen(probe, in: F.view)!, b = v1.screen(probe, in: F.view)!
            // same distance from the middle of the screen at tilt 0 (a rotation about the view axis, in physical pixels)
            #expect(abs(hypot(a.x - 550, a.y - 400) - hypot(b.x - 550, b.y - 400)) < 0.05, "s \(s)")
            #expect(hypot(a.x - b.x, a.y - b.y) > 5)
            #expect(abs(v1.frame.heading - 1.1) < 1e-6)
        }
    }

    @Test func theStandardViewsAreAnglesAboutThePlanetAndSideKeepsTheGlobesDistance() {
        let globeD = Morph.globeDistance(L: L, fovY: F.fov, aspect: F.aspect)
        let pose = WorldPose(x: 100, y: 0, heading: 0.5, tilt: 0, distance: globeD, fovY: F.fov)
        for (preset, tilt) in [(ViewPreset.top, Float(0)), (.oblique, Float.pi / 4), (.low, 70 * Float.pi / 180), (.side, Float.pi / 2)] {
            let ease = ViewEase(preset: preset, from: pose, wish: nil, aspect: F.aspect, size: L, height: F.H, keepHeight: false, globe: true)
            #expect(ease.toTilt == tilt)
            #expect(ease.toDistance == globeD, "\(preset): the globe keeps its distance, \(ease.toDistance)")
            #expect(ease.toHeading == (preset == .top ? 0 : 0.5))
        }
        // The flat close-up keeps the Side fit to the height.
        let close = WorldPose(x: 100, y: 0, tilt: 0, distance: 20, fovY: F.fov)
        let side = ViewEase(preset: .side, from: close, wish: nil, aspect: F.aspect, size: L, height: F.H, keepHeight: false, globe: true)
        #expect(abs(side.toDistance - 20) > 1, "Side at the close-up fits the height: \(side.toDistance)")
        // The 3D button at the globe is not held by the flat map's eye reach.
        let far = WorldPose(x: 100, y: 0, tilt: 0, distance: 0.5 * L, fovY: F.fov)
        let tilted = ViewEase(tiltTo: 70 * .pi / 180, from: far, wish: nil, aspect: F.aspect, size: L, globe: true)
        let flat = ViewEase(tiltTo: 70 * .pi / 180, from: far, wish: nil, aspect: F.aspect, size: L, globe: false)
        #expect(tilted.toDistance == 0.5 * L && flat.toDistance < 0.5 * L)
    }
}

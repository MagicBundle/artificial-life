import Foundation
import Testing
import simd

@testable import AlifeCore

/// The size rule of DESIGN §36.7, with the numbers of the spec's diagnosis (a recorded 44,162-organism planet at 2280 x 1800
/// drawable pixels: 3,993 rows in view at the region pose, 163 at the organism fly-to).
@Suite struct OrganismSizePolicyTests {
    private let frameArea: Float = 2280 * 1800
    private let L: Float = 320
    private let fov = Camera.defaultFovY

    /// A deterministic generator (splitmix64), so the synthetic cases are the same on every run.
    private struct Generator: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    // MARK: Coverage

    @Test func theCoverageNeverExceedsTheLimitWhenTheMinimumDoesNotBind() {
        var rng = Generator(state: 7)
        var checked = 0
        for _ in 0..<1000 {
            let area = Float.random(in: 2e5...6e6, using: &rng)
            let rows = Int.random(in: 1...60_000, using: &rng)
            let physical = Float.random(in: 0.5...120, using: &rng)
            let cover = OrganismSizePolicy.coverRadius(area: area, rows: rows)
            guard cover >= OrganismSizePolicy.minimumRadius else { continue }  // the minimum binds: no promise
            let r = OrganismSizePolicy.radius(physical: physical, cap: cover, minimum: OrganismSizePolicy.minimumRadius)
            let covered = Float(rows) * Float.pi * r * r / area
            #expect(covered <= OrganismSizePolicy.coverage * 1.0001, "A \(area) N \(rows) r_phys \(physical): \(covered)")
            checked += 1
        }
        #expect(checked > 300, "the sample is not all minimum-bound: \(checked)")
    }

    @Test func theRadiusIsAtLeastTheMinimumAndCoverShrinksAsRowsGrow() {
        var previous = Float.infinity
        for rows in stride(from: 10, through: 80_000, by: 790) {
            let cover = OrganismSizePolicy.coverRadius(area: frameArea, rows: rows)
            #expect(cover <= previous, "monotone non-increasing in N at \(rows)")
            previous = cover
            let capped = OrganismSizePolicy.radius(physical: 0.2, cap: cover, minimum: OrganismSizePolicy.minimumRadius)
            #expect(capped >= OrganismSizePolicy.minimumRadius)
            // Even a view with a vast crowd keeps the minimum, and one organism may be as large as its physical size.
            let crowded = OrganismSizePolicy.radius(physical: 40, cap: cover, minimum: OrganismSizePolicy.minimumRadius)
            #expect(crowded >= OrganismSizePolicy.minimumRadius && crowded <= max(cover, OrganismSizePolicy.minimumRadius) + 1e-4)
        }
        #expect(OrganismSizePolicy.coverRadius(area: frameArea, rows: 0) == .infinity, "an empty view has no cover radius")
        let drawn = OrganismSizePolicy.radius(physical: 40, cap: .infinity, minimum: 2.5)
        #expect(drawn == 40, "no cap: the physical radius")
    }

    @Test func theFootprintIsTheAreaAndTheRowsOfWhatIsVisible() {
        // A globe whose disc is 32% of the frame, with the 14,701 rows of its visible cap: 3.16 px, not the 3.4 of the whole frame
        // with every row of the planet (39,848 rows in the flat view).
        let disc = 0.32 * frameArea
        #expect(abs(OrganismSizePolicy.coverRadius(area: disc, rows: 14_701) - 3.16) < 0.03)
        #expect(abs(OrganismSizePolicy.coverRadius(area: frameArea, rows: 39_848) - 3.40) < 0.03)
    }

    @Test func theRegionProbeIsDrawnAtNinePixelsAndTheOrganismProbeIsMinOfPhysicalAndCover() {
        // Region pose: d = 108.4 (0.339 L), 3,993 rows, a physical median of 19.3 px: the coverage cap is 10.7 px, the absolute 9.
        let cover = OrganismSizePolicy.coverRadius(area: frameArea, rows: 3_993)
        #expect(abs(cover - 10.7) < 0.1)
        let region = OrganismSizePolicy.limits(mode: .capped, cover: cover, distance: 108.4, size: L, agentRadius: 1,
                                                viewHeight: 1800, fovY: fov, pixelsPerPoint: 2)
        #expect(region.cap == 9 && region.minimum == 2.5)
        let drawn = OrganismSizePolicy.radius(physical: 19.3, cap: region.cap, minimum: region.minimum)
        #expect(drawn == 9)
        let coverage = 3_993 * Float.pi * drawn * drawn / frameArea
        #expect(abs(coverage - 0.248) < 0.005, "24.8% of the view, not 83.6%: \(coverage)")

        // Organism fly-to (d = 0.08 L): no absolute cap, so r_draw = min(r_phys, r_cover): 53 px against 56 at 163 rows ...
        let near = OrganismSizePolicy.limits(
            mode: .capped, cover: OrganismSizePolicy.coverRadius(area: frameArea, rows: 163), distance: 0.08 * L, size: L,
            agentRadius: 1, viewHeight: 1800, fovY: fov, pixelsPerPoint: 2)
        #expect(abs(near.cap - 53) < 0.5 && near.cap.isFinite)
        #expect(abs(OrganismSizePolicy.radius(physical: 56.3, cap: near.cap, minimum: near.minimum) - near.cap) < 1e-4)
        #expect(OrganismSizePolicy.radius(physical: 40, cap: near.cap, minimum: near.minimum) == 40, "a small one stays physical")
        // ... and about 30 against 75 in a dense spot of 500 rows.
        let dense = OrganismSizePolicy.limits(
            mode: .capped, cover: OrganismSizePolicy.coverRadius(area: frameArea, rows: 500), distance: 0.08 * L, size: L,
            agentRadius: 1, viewHeight: 1800, fovY: fov, pixelsPerPoint: 2)
        #expect(abs(OrganismSizePolicy.radius(physical: 75.4, cap: dense.cap, minimum: dense.minimum) - 30.2) < 0.5)
        // Closer than 0.08 L the absolute cap is gone for good.
        let closer = OrganismSizePolicy.limits(mode: .capped, cover: .infinity, distance: 0.04 * L, size: L, agentRadius: 1,
                                                viewHeight: 1800, fovY: fov, pixelsPerPoint: 2)
        #expect(closer.cap == .infinity)
    }

    // MARK: The absolute cap in the distance

    @Test func theAbsoluteCapIsContinuousFromTheFarLimitToTheOrganismLevel() {
        func absolute(_ fraction: Float) -> Float {
            OrganismSizePolicy.absoluteRadius(distance: fraction * L, size: L, agentRadius: 1, viewHeight: 1800, fovY: fov,
                                              pixelsPerPoint: 2)
        }
        #expect(absolute(0.2) == 9 && absolute(0.5) == 9 && absolute(2) == 9, "4.5 pt from 0.2 L outwards")
        let atNear = absolute(0.0801)
        let physicalAtNear = 1 * OrganismSizePolicy.pixelsPerUnit(depth: 0.08 * L, viewHeight: 1800, fovY: fov)
        #expect(abs(atNear - physicalAtNear) < 0.01 * physicalAtNear, "the organism's physical radius at 0.08 L: \(atNear) vs \(physicalAtNear)")
        #expect(abs(physicalAtNear - 75.4) < 0.5)
        var previous = absolute(0.2)
        var largestStep: Float = 0
        for i in 1...600 {
            let f = 0.2 - 0.12 * Float(i) / 600 + 0.0001
            let v = absolute(f)
            #expect(v >= previous - 1e-4, "non-increasing in the distance (a cap that grows as the camera comes in) at \(f)")
            largestStep = max(largestStep, (v - previous) / previous)
            previous = v
        }
        #expect(largestStep < 0.01, "no step: the largest is \(largestStep) of the value per 0.0002 L")
        #expect(absolute(0.08) == .infinity && absolute(0.05) == .infinity)
        // The blend is a power of the distance: from 9 px to 75 px over a fall of 2.5 in d, the 2.3rd power of the scene scale.
        let exponent = log(physicalAtNear / 9) / log(Float(0.2) / 0.08)
        #expect(abs(exponent - 2.3) < 0.05, "organisms inflate as the \(exponent)th power of the zoom")
    }

    @Test func thePhysicalModeHasNoCapAndV1sMinimum() {
        let limits = OrganismSizePolicy.limits(mode: .physical, cover: 3, distance: 400, size: L, agentRadius: 1, viewHeight: 1800,
                                                fovY: fov, pixelsPerPoint: 2)
        #expect(limits.cap == .infinity && limits.minimum == WorldLevels.minDiscPixels)
    }

    // MARK: The instance alpha

    @Test func theInstanceAlphaIsARampOfTheDrawnRadius() {
        #expect(OrganismSizePolicy.instanceAlpha(radius: 0) == 0 && OrganismSizePolicy.instanceAlpha(radius: 1.5) == 0)
        #expect(OrganismSizePolicy.instanceAlpha(radius: 3.5) == 1 && OrganismSizePolicy.instanceAlpha(radius: 60) == 1)
        #expect(abs(OrganismSizePolicy.instanceAlpha(radius: 2.5) - 0.5) < 1e-6)
        var previous: Float = 0
        for i in 0...400 {
            let a = OrganismSizePolicy.instanceAlpha(radius: Float(i) * 0.01)
            #expect(a >= previous && a <= 1)
            previous = a
        }
        // The capped size is at least 2.5 px, so a capped organism is at least half drawn; the physical minimum is not.
        #expect(OrganismSizePolicy.instanceAlpha(radius: OrganismSizePolicy.minimumRadius) >= 0.5)
        #expect(OrganismSizePolicy.instanceAlpha(radius: OrganismSizePolicy.physicalMinimumRadius) == 0)
    }

    // MARK: Rows in view

    @Test func theRowsInViewAreCountedAtTheirMinimumImage() throws {
        let size: Float = 48
        var camera = Camera.planet(size: size, height: 24, aspect: 1)
        camera.target = SIMD3(1, 1, 0)
        camera.distance = 20  // a window of about 18 units: the corner of the world, seen across the seam
        let inside: [SIMD3<Float>] = [SIMD3(1, 1, 0), SIMD3(5, 3, 2), SIMD3(47, 46, 1), SIMD3(-3, 4, 0).wrappedInto(size)]
        let outside: [SIMD3<Float>] = [SIMD3(24, 24, 0), SIMD3(14, 1, 0), SIMD3(1, 30, 0)]
        let frame = WorldFixtures.frame(inside + outside)
        #expect(OrganismSizePolicy.rowsInView(frame: frame, camera: camera, aspect: 1, size: size) == inside.count)
        // Behind the eye is not in view: a row above the camera.
        let above = WorldFixtures.frame(inside + [SIMD3(1, 1, 60)])
        #expect(OrganismSizePolicy.rowsInView(frame: above, camera: camera, aspect: 1, size: size) == inside.count)
        // Zoomed out far enough that the whole world is in view, every row is (at its minimum image).
        camera.distance = 200
        #expect(OrganismSizePolicy.rowsInView(frame: frame, camera: camera, aspect: 1, size: size) == frame.n)
    }

    @Test func theCoverRadiusFollowsTheTargetWithAOnePoleFilter() {
        var value: Float = 12
        let target: Float = 4
        // Not finite on either side: the target at once.
        #expect(OrganismSizePolicy.smoothed(.infinity, toward: target, dt: 0.016) == target)
        #expect(OrganismSizePolicy.smoothed(value, toward: .infinity, dt: 0.016) == .infinity)
        #expect(OrganismSizePolicy.smoothed(value, toward: target, dt: 0) == value)
        var steps = 0
        while abs(value - target) > 0.05 * (12 - target), steps < 1000 {
            value = OrganismSizePolicy.smoothed(value, toward: target, dt: 1.0 / 60)
            steps += 1
        }
        let seconds = Float(steps) / 60
        #expect(seconds > 0.6 && seconds < 0.9, "95% of the way in about three time constants of 0.25 s: \(seconds)")
        // The step does not depend on how the time is cut up.
        let once = OrganismSizePolicy.smoothed(12, toward: 4, dt: 0.5)
        var cut: Float = 12
        for _ in 0..<5 { cut = OrganismSizePolicy.smoothed(cut, toward: 4, dt: 0.1) }
        #expect(abs(once - cut) < 1e-4)
    }

    // MARK: The metric: organisms that keep half their disc

    /// A disc as it is drawn: a centre, a radius and a height (nearer discs are drawn over farther ones).
    private struct Disc {
        var x: Float, y: Float, r: Float, z: Float
    }

    /// The share of the discs with at least `fraction` of their area visible after the nearer ones are drawn over them, rasterised
    /// on a grid of `cell`-pixel cells (the pure form of the GPU test's metric).
    private func visibleShare(_ discs: [Disc], width: Int, height: Int, fraction: Float = 0.5, cell: Float = 1) -> Float {
        let w = Int(Float(width) / cell), h = Int(Float(height) / cell)
        var ids = [Int32](repeating: -1, count: w * h)
        let order = discs.indices.sorted { discs[$0].z < discs[$1].z }  // far (low z) first, the nearest last = on top
        for i in order {
            let d = discs[i]
            let x0 = max(Int(((d.x - d.r) / cell).rounded(.down)), 0), x1 = min(Int(((d.x + d.r) / cell).rounded(.up)), w - 1)
            let y0 = max(Int(((d.y - d.r) / cell).rounded(.down)), 0), y1 = min(Int(((d.y + d.r) / cell).rounded(.up)), h - 1)
            guard x0 <= x1, y0 <= y1 else { continue }
            for y in y0...y1 {
                for x in x0...x1 {
                    let dx = (Float(x) + 0.5) * cell - d.x, dy = (Float(y) + 0.5) * cell - d.y
                    if dx * dx + dy * dy <= d.r * d.r { ids[y * w + x] = Int32(i) }
                }
            }
        }
        var visible = [Int](repeating: 0, count: discs.count)
        for id in ids where id >= 0 { visible[Int(id)] += 1 }
        var counted = 0, kept = 0
        for (i, d) in discs.enumerated() {
            guard d.x >= d.r, d.y >= d.r, d.x <= Float(width) - d.r, d.y <= Float(height) - d.r else { continue }  // whole discs only
            counted += 1
            let area = Float.pi * d.r * d.r / (cell * cell)
            if Float(visible[i]) >= fraction * area { kept += 1 }
        }
        return counted == 0 ? 1 : Float(kept) / Float(counted)
    }

    @Test func singleOrganismsStayVisibleUnderTheRuleAndHideUnderPhysicalSizes() {
        // The region probe's numbers on a synthetic planet: 3,993 organisms spread over the 2280 x 1800 view, heights 0 to 80,
        // the camera straight above at d = 108.4 (so a row at z is 108.4 - z from the eye).
        var rng = Generator(state: 42)
        let width = 2280, height = 1800
        let f = Float(height) / 2 / tan(fov / 2)
        let rows = 3_993
        var physical: [Disc] = [], capped: [Disc] = []
        let cover = OrganismSizePolicy.coverRadius(area: Float(width * height), rows: rows)
        let limits = OrganismSizePolicy.limits(mode: .capped, cover: cover, distance: 108.4, size: L, agentRadius: 1,
                                                viewHeight: Float(height), fovY: fov, pixelsPerPoint: 2)
        for _ in 0..<rows {
            let x = Float.random(in: 0..<Float(width), using: &rng), y = Float.random(in: 0..<Float(height), using: &rng)
            let z = Float.random(in: 0...80, using: &rng)
            let rPhys = f / (108.4 - z)
            physical.append(Disc(x: x, y: y, r: rPhys, z: z))
            capped.append(Disc(x: x, y: y, r: OrganismSizePolicy.radius(physical: rPhys, cap: limits.cap, minimum: limits.minimum), z: z))
        }
        let physicalShare = visibleShare(physical, width: width, height: height, cell: 2)
        let cappedShare = visibleShare(capped, width: width, height: height, cell: 2)
        #expect(cappedShare >= 0.80, "at least 80% of the organisms keep half their disc: \(cappedShare)")
        #expect(physicalShare < 0.60, "physical sizes bury them (the recorded frame: 21% to 30%): \(physicalShare)")
        #expect(cappedShare > physicalShare + 0.3)
    }
}

extension SIMD3 where Scalar == Float {
    /// The point with x and y wrapped into `[0, size)`.
    fileprivate func wrappedInto(_ size: Float) -> SIMD3<Float> {
        SIMD3(Torus.wrap(x, size: size), Torus.wrap(y, size: size), z)
    }
}

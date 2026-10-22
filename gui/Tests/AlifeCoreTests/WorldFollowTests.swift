import Foundation
import Testing
import simd

@testable import AlifeCore

/// The follow camera: the critically damped approach, the nearest image across the seam, and what a target resolves to.
@Suite struct CameraGlideTests {
    private let L: Float = 320

    private func camera(at target: SIMD3<Float>, distance: Float = 100) -> Camera {
        var c = Camera.planet(size: L)
        c.target = target
        c.distance = distance
        return c
    }

    @Test func theApproachIsTheCriticallyDampedSolution() {
        // x'' = -2w x' - w² x, integrated finely, against the closed form.
        let w: Float = 2.5
        var e: Float = 10, v: Float = -3
        let h: Float = 1e-5
        for _ in 0..<Int(0.8 / h) {
            let a = -2 * w * v - w * w * e
            v += a * h
            e += v * h
        }
        let exact = CameraGlide.advance(offset: 10, velocity: -3, rate: w, dt: 0.8)
        #expect(abs(exact.offset - e) < 2e-3 && abs(exact.velocity - v) < 2e-3, "\(exact) against \((e, v))")
    }

    @Test func oneSecondFromRestLeavesTwoOverEOfTheOffset() {
        // A time constant of one second: (1 + w t) exp(-w t) with w t = 1.
        let r = CameraGlide.advance(offset: 100, velocity: 0, rate: 1, dt: 1)
        #expect(abs(r.offset - 100 * 2 * exp(-1)) < 1e-3)
    }

    @Test func theStepOfTimeDoesNotChangeTheResult() {
        let ten = (0..<10).reduce((offset: Float(40), velocity: Float(5))) { s, _ in
            CameraGlide.advance(offset: s.offset, velocity: s.velocity, rate: 1.7, dt: 0.1)
        }
        let one = CameraGlide.advance(offset: 40, velocity: 5, rate: 1.7, dt: 1)
        #expect(abs(ten.offset - one.offset) < 1e-3 && abs(ten.velocity - one.velocity) < 1e-3)
    }

    @Test func theCameraNeverOvershootsFromRest() {
        var c = camera(at: SIMD3(100, 100, 0))
        var glide = CameraGlide()
        let goal = SIMD3<Float>(140, 100, 0)
        var last = c.target.x
        for _ in 0..<400 {
            glide.advance(&c, toward: goal, distance: nil, dt: 1.0 / 60, size: L)
            #expect(c.target.x >= last - 1e-5 && c.target.x <= goal.x + 1e-4, "monotone and never past the goal")
            last = c.target.x
        }
    }

    @Test func itArrivesAndSaysSo() {
        var c = camera(at: SIMD3(100, 100, 0))
        var glide = CameraGlide()
        let goal = SIMD3<Float>(103, 98, 0)
        var steps = 0
        while glide.advance(&c, toward: goal, distance: nil, dt: 1.0 / 60, size: L) {
            steps += 1
            #expect(steps < 1200, "arrives within 20 seconds")
            if steps >= 1200 { break }
        }
        #expect(steps > 30, "not at once")
        #expect(simd_length(c.target - goal) < 0.05)
        // Already there: nothing to do.
        var at = camera(at: goal)
        var rest = CameraGlide()
        #expect(!rest.advance(&at, toward: goal, distance: nil, dt: 1.0 / 60, size: L))
    }

    @Test func aGoalAcrossTheSeamIsApproachedTheShortWay() {
        // The target is two units inside the near edge, the goal two inside the far edge: four units apart, not 316.
        var c = camera(at: SIMD3(2, 160, 0))
        var glide = CameraGlide()
        let goal = SIMD3<Float>(318, 160, 0)
        var last = c.target.x
        var worst: Float = 0
        for _ in 0..<600 {
            glide.advance(&c, toward: goal, distance: nil, dt: 1.0 / 60, size: L)
            // The target is carried at the goal's image (318 + a little), which is the same place on the torus as -2 + a
            // little: the movement of a frame, measured on the torus, is small and the target never leaves the four units.
            worst = max(worst, abs(Torus.wrapDelta(c.target.x - last, size: L)))
            #expect(abs(Torus.wrapDelta(c.target.x - 2, size: L)) <= 4 + 1e-3, "no whip: \(c.target.x)")
            last = c.target.x
        }
        #expect(worst < 0.5, "no frame moves it by more than half a unit: \(worst)")
        let wrapped = Torus.wrap(c.target.x, size: L)
        #expect(abs(Torus.wrapDelta(wrapped - goal.x, size: L)) < 0.1)
    }

    @Test func aMovingGoalIsFollowedWithALagOfTwiceItsSpeedTimesTheConstant() {
        // A goal moving at 6 units a second, followed for 20 s with the follow constant of one second, trails by twice that
        // (two lags of the constant in series): 12 units.
        var c = camera(at: SIMD3(100, 100, 0))
        var glide = CameraGlide()
        var goal = SIMD3<Float>(100, 100, 0)
        for _ in 0..<(20 * 60) {
            goal.x += 6.0 / 60
            glide.advance(&c, toward: goal, distance: nil, dt: 1.0 / 60, size: L)
        }
        #expect(abs((goal.x - c.target.x) - 2 * 6 * CameraGlide.followConstant) < 0.1, "lag \(goal.x - c.target.x)")
    }

    @Test func aFlyIsQuickerThanAFollow() {
        func seconds(_ constant: Float) -> Int {
            var c = camera(at: SIMD3(100, 100, 0))
            var glide = CameraGlide()
            glide.timeConstant = constant
            var frames = 0
            while glide.advance(&c, toward: SIMD3(200, 100, 0), distance: nil, dt: 1.0 / 60, size: L), frames < 10_000 { frames += 1 }
            return frames
        }
        #expect(seconds(CameraGlide.flyConstant) * 2 < seconds(CameraGlide.followConstant))
        #expect(CameraGlide.followConstant == 1 && CameraGlide.flyConstant < 1)
    }

    @Test func theDistanceIsApproachedAndTheCameraDoesNotGoUnderTheMinimum() {
        var c = camera(at: SIMD3(100, 100, 0), distance: 300)
        var glide = CameraGlide()
        glide.timeConstant = CameraGlide.flyConstant
        var arrived = false
        for _ in 0..<600 {
            if !glide.advance(&c, toward: c.target, distance: 40, dt: 1.0 / 60, size: L) { arrived = true; break }
            #expect(c.distance >= 40 - 1e-3, "it comes down to the goal and not past it")
        }
        #expect(arrived && abs(c.distance - 40) < 0.2)
        var near = camera(at: SIMD3(100, 100, 0), distance: 10)
        var g2 = CameraGlide()
        for _ in 0..<600 { g2.advance(&near, toward: near.target, distance: 0.01, dt: 1.0 / 60, size: L) }
        #expect(near.distance >= Camera.minDistance)
    }

    @Test func aLargeStepOfTimeIsCappedSoAStalledFrameDoesNotJump() {
        var c = camera(at: SIMD3(100, 100, 0))
        var glide = CameraGlide()
        glide.advance(&c, toward: SIMD3(200, 100, 0), distance: nil, dt: 30, size: L)  // the window was in the background
        #expect(c.target.x < 110, "at most a tenth of a second of approach: \(c.target.x)")
        var back = camera(at: SIMD3(100, 100, 0))
        glide.advance(&back, toward: SIMD3(200, 100, 0), distance: nil, dt: -5, size: L)
        #expect(back.target.x >= 100 - 1e-4, "never backwards")
    }

    @Test func stoppingForgetsTheMomentum() {
        var c = camera(at: SIMD3(100, 100, 0))
        var glide = CameraGlide()
        for _ in 0..<30 { glide.advance(&c, toward: SIMD3(200, 100, 0), distance: 10, dt: 1.0 / 60, size: L) }
        #expect(simd_length(glide.velocity) > 0)
        glide.stop()
        #expect(glide.velocity == .zero && glide.zoomVelocity == 0)
    }
}

@Suite struct FollowStateTests {
    private let uid0 = WorldFixtures.uid0

    @Test func anOrganismIsFollowedByItsUid() throws {
        var store = try WorldFixtures.store()
        var state = FollowState(target: .organism(uid: uid0 + 100))
        #expect(state.goal(store: store, size: 48) == SIMD3(20.5, 8.5, 1))
        // It moves: the goal moves with it, wherever its row is now.
        store.apply(.frma(WorldFixtures.frame([SIMD3(3, 4, 5), SIMD3(21, 9, 2)], step: 110, uids: [7, uid0 + 100])))
        #expect(state.goal(store: store, size: 48) == SIMD3(21, 9, 2))
    }

    @Test func atItsDeathTheCameraHoldsThePointItDiedAt() throws {
        var store = try WorldFixtures.store()
        var state = FollowState(target: .organism(uid: uid0 + 100))
        _ = state.goal(store: store, size: 48)
        store.apply(.frma(WorldFixtures.frame([SIMD3(3, 4, 5)], step: 110, uids: [7])))  // it is not in the frame any more
        #expect(store.listedIndex(ofUID: uid0 + 100) == nil)
        #expect(state.goal(store: store, size: 48) == SIMD3(20.5, 8.5, 1), "the last place it was seen")
        #expect(state.lastPoint == SIMD3(20.5, 8.5, 1))
        // Retargeted (the hand-off): the new one is followed, and the old point is forgotten.
        state.set(.organism(uid: 7))
        #expect(state.lastPoint == nil && state.goal(store: store, size: 48) == SIMD3(3, 4, 5))
    }

    @Test func anOrganismNeverSeenHasNoGoal() throws {
        let store = try WorldFixtures.store()
        var state = FollowState(target: .organism(uid: 123))
        #expect(state.goal(store: store, size: 48) == nil)
        var none = FollowState()
        #expect(none.goal(store: store, size: 48) == nil)
    }

    @Test func aStormIsFollowedByItsPositionAndHoldsItWhenItEnds() throws {
        var store = try WorldFixtures.store()
        var state = FollowState(target: .storm(id: 7))
        #expect(state.goal(store: store, size: 48) == SIMD3(4.5, 20.25, 0))
        var record = try #require(store.storms)
        record.storms.storms.removeAll { $0.id == 7 }
        store.apply(.strm(record))
        #expect(state.goal(store: store, size: 48) == SIMD3(4.5, 20.25, 0), "where it was last seen")
    }

    @Test func aLineageIsFollowedByTheCentroidOfItsRows() throws {
        // Kin group 17 (slot 0) has two rows in the newest frame, at x = 1.25 and 47.25 (across the seam of L = 48), y = 24,
        // z = 6 and 7: the circular mean is next to the seam and not in the middle of the world.
        let store = try WorldFixtures.store()
        var state = FollowState(target: .lineage(kinID: 17))
        let found = state.goal(store: store, size: 48)
        let goal = try #require(found)
        #expect(abs(Torus.wrapDelta(goal.x - 0.25, size: 48)) < 1e-3, "x = \(goal.x)")
        #expect(abs(goal.y - 24) < 1e-3 && abs(goal.z - 6.5) < 1e-3)
        #expect(goal.x >= 0 && goal.x < 48)
    }

    @Test func aLineageWithNoRowsInTheFrameFallsBackToItsTablePosition() throws {
        // Kin group 22 (slot 2) lists rows that have died since: the table says (30.5, 40.5, 12.25).
        let store = try WorldFixtures.store()
        var state = FollowState(target: .lineage(kinID: 22))
        #expect(state.goal(store: store, size: 48) == SIMD3(30.5, 40.5, 12.25))
        var unknown = FollowState(target: .lineage(kinID: 999))
        #expect(unknown.goal(store: store, size: 48) == nil)
    }

    @Test func theCentroidIsRecomputedEveryFourthFrame() throws {
        var store = try WorldFixtures.store()
        var state = FollowState(target: .lineage(kinID: 21))  // slot 1: uid0 + 14 and uid0 + 21
        let firstGoal = state.goal(store: store, size: 48)
        let first = try #require(firstGoal)
        func move(to x: Float, step: UInt64) {
            store.apply(.frma(WorldFixtures.frame([SIMD3(x, 10, 2), SIMD3(x, 12, 2)], step: step, uids: [uid0 + 14, uid0 + 21])))
        }
        for k in 1...3 {
            move(to: 30, step: UInt64(110 + k))
            #expect(state.goal(store: store, size: 48) == first, "frame \(k) after: the cached centroid")
        }
        move(to: 30, step: 120)
        let updatedGoal = state.goal(store: store, size: 48)
        let updated = try #require(updatedGoal)
        #expect(abs(updated.x - 30) < 1e-3 && abs(updated.y - 11) < 1e-3)
    }

    @Test func aNewTargetStartsFresh() throws {
        let store = try WorldFixtures.store()
        var state = FollowState(target: .storm(id: 7))
        _ = state.goal(store: store, size: 48)
        state.set(.none)
        #expect(state.target.isNone && state.lastPoint == nil && state.goal(store: store, size: 48) == nil)
    }
}

@Suite struct FlyDistanceTests {
    @Test func aRegionFlyEndsInTheRegionLevelAndAnOrganismFlyInTheOrganismLevel() {
        for size: Float in [48, 320] {
            for aspect: Float in [0.8, 1.3, 2.5] {
                let fov = Camera.defaultFovY
                let region = WorldLevels.flyDistance(.region, size: size, fovY: fov, aspect: aspect)
                #expect(WorldLevels.level(distance: region, fovY: fov, aspect: aspect, size: size) == .region, "L \(size) aspect \(aspect)")
                let organism = WorldLevels.flyDistance(.organism, size: size, fovY: fov, aspect: aspect)
                #expect(WorldLevels.level(distance: organism, fovY: fov, aspect: aspect, size: size) == .organism || organism == WorldLevels.minDistance)
                #expect(organism < region && organism >= WorldLevels.minDistance)
                #expect(region <= WorldLevels.maxDistance(size: size, fovY: fov, aspect: aspect) + 1e-3)
            }
        }
    }
}

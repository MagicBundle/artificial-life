import Foundation
import Testing
import simd

@testable import AlifeCore

/// The planet's camera (DESIGN §46.8): the pose, the matrices, the rays, the grab, the zoom, the standard views and the flights.
@Suite struct SphereCameraTests {
    static let rp: Float = 36, lid: Float = 52

    static func camera(lat: Double = 20, lon: Double = 30, tilt: Float = 0, distance: Float = 60) -> SphereCamera {
        SphereCamera(pose: SpherePose(lat: lat, lon: lon, tilt: tilt, distance: distance), planetRadius: rp, lidRadius: lid)
    }

    static func close(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ tolerance: Float = 1e-4) -> Bool { simd_length(a - b) < tolerance }

    // MARK: The basis

    @Test func theEyeIsAboveThePointLookedAtAndLooksAtIt() {
        let c = Self.camera(tilt: 0, distance: 60)
        #expect(Self.close(c.target, c.pose.up * Self.rp))
        #expect(Self.close(c.eye, c.pose.up * (Self.rp + 60)))
        #expect(Self.close(c.forward, -c.pose.up))
        #expect(Self.close(c.screenUp, c.pose.heading) && simd_dot(c.right, c.forward) < 1e-5)
        let tilted = Self.camera(tilt: 1, distance: 30)
        // the eye looks at the floor point, from `distance` away
        let toTarget = tilted.target - tilted.eye
        #expect(abs(simd_length(toTarget) - 30) < 1e-3 && Self.close(simd_normalize(toTarget), tilted.forward))
        #expect(abs(simd_length(tilted.forward) - 1) < 1e-5 && abs(simd_length(tilted.right) - 1) < 1e-5 && abs(simd_length(tilted.screenUp) - 1) < 1e-5)
        #expect(simd_dot(tilted.forward, tilted.screenUp) < 1e-5 && simd_dot(tilted.forward, tilted.right) < 1e-5)
    }

    @Test func northIsUpTheScreenAtTiltZeroAndEastIsRight() {
        let c = Self.camera(lat: 10, lon: 80)
        let b = SphereFrame.basis(at: c.pose.up)
        #expect(Self.close(c.screenUp, b.north) && Self.close(c.right, b.east), "looking down from above, north is up and east is right")
        #expect(abs(c.pose.bearing) < 1e-5)
    }

    @Test func theEyeNeverGoesBelowTheFloor() {
        for tilt in stride(from: Float(0), through: SpherePose.maxTilt, by: 0.1) {
            for distance: Float in [3, 10, 60, 200] {
                let c = Self.camera(lat: -50, lon: 170, tilt: tilt, distance: distance)
                #expect(c.altitude >= -1e-3, "tilt \(tilt) distance \(distance)")
            }
        }
    }

    @Test func thePoseKeepsItsFramesOrthonormalAndClampsItsRange() {
        let p = SpherePose(up: SIMD3(0, 0, 3), heading: SIMD3(0, 0, 1), tilt: 9, distance: 0.1)  // a heading along `up` has no direction of its own
        #expect(abs(simd_length(p.up) - 1) < 1e-6 && abs(simd_length(p.heading) - 1) < 1e-6 && abs(simd_dot(p.up, p.heading)) < 1e-6)
        #expect(p.tilt == SpherePose.maxTilt && p.distance == SpherePose.minDistance)
        let q = SpherePose(up: SIMD3(1, 1, 0), heading: SIMD3(0, 0, 5), tilt: -2, distance: 10)
        #expect(q.tilt == 0)
    }

    // MARK: Projection and rays

    @Test func theCentreOfTheViewIsThePointLookedAt() {
        let c = Self.camera(tilt: 0.8, distance: 40)
        let ndc = c.ndc(of: c.target, aspect: 1.6)
        #expect(ndc != nil && abs(ndc!.x) < 1e-4 && abs(ndc!.y) < 1e-4)
        let r = c.ray(ndc: .zero, aspect: 1.6)
        #expect(Self.close(r.origin, c.eye) && Self.close(r.direction, c.forward))
        #expect(c.floorDirection(ndc: .zero, aspect: 1.6).map { Self.close($0, c.pose.up, 1e-4) } == true)
    }

    @Test func aRayThroughAProjectedPointComesBack() {
        let c = Self.camera(lat: -20, lon: -100, tilt: 0.6, distance: 45)
        for p in [SIMD3<Float>(20, 5, 10), SIMD3(c.target.x + 3, c.target.y - 2, c.target.z + 1), c.target * 1.1] {
            guard let ndc = c.ndc(of: p, aspect: 1.5) else { Issue.record("behind the eye"); continue }
            let r = c.ray(ndc: ndc, aspect: 1.5)
            let t = simd_dot(p - r.origin, r.direction)
            #expect(simd_length(r.origin + t * r.direction - p) < 1e-3, "the ray through the image of \(p) passes through it")
        }
    }

    @Test func aPointBehindTheEyeHasNoImage() {
        let c = Self.camera(tilt: 0, distance: 10)
        #expect(c.ndc(of: c.eye + c.pose.up * 5, aspect: 1) == nil)
    }

    @Test func theRayAndTheFloorMeetWhereTheGeometrySays() {
        let o = SIMD3<Float>(0, 0, 100)
        let hit = SphereCamera.intersect(origin: o, direction: SIMD3(0, 0, -1), radius: 36)
        #expect(hit != nil && abs(hit!.near - 64) < 1e-4 && abs(hit!.far - 136) < 1e-4)
        #expect(SphereCamera.intersect(origin: o, direction: SIMD3(1, 0, 0), radius: 36) == nil)
        // a ray beside the planet misses it, and the grab direction is the limb's
        let c = Self.camera(tilt: 0, distance: 120)
        #expect(c.floorDirection(ndc: SIMD2(1, 1), aspect: 1) == nil)
        let g = c.grabDirection(ndc: SIMD2(1, 1), aspect: 1)
        #expect(abs(simd_length(g) - 1) < 1e-5)
    }

    @Test func theFarSideIsHiddenAndTheNearSideIsNot() {
        let c = Self.camera(lat: 0, lon: 0, tilt: 0, distance: 60)
        #expect(!c.isOccluded(SIMD3<Float>(Self.rp + 2, 0, 0)))
        #expect(c.isOccluded(SIMD3<Float>(-(Self.rp + 2), 0, 0)))
        #expect(!c.isOccluded(SIMD3<Float>(Self.rp + 8, 0, 0)))
        #expect(c.isOccluded(SIMD3<Float>(-Self.rp, 0, 3)), "a point on the far floor")
    }

    // MARK: The grab

    @Test func theGrabKeepsTheAnchorUnderThePointer() {
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<60 {
            var c = Self.camera(lat: Double.random(in: -80...80, using: &rng), lon: Double.random(in: -180...180, using: &rng),
                                tilt: Float.random(in: 0...1.2, using: &rng), distance: Float.random(in: 20...150, using: &rng))
            let aspect = Float.random(in: 0.8...2, using: &rng)
            // a press on the planet, a drag to another point of the planet
            guard let anchor = c.floorDirection(ndc: SIMD2(Float.random(in: -0.3...0.3, using: &rng), Float.random(in: -0.3...0.3, using: &rng)), aspect: aspect) else { continue }
            let to = SIMD2(Float.random(in: -0.25...0.25, using: &rng), Float.random(in: -0.25...0.25, using: &rng))
            guard c.floorDirection(ndc: to, aspect: aspect) != nil else { continue }
            c.grab(anchor: anchor, at: to, aspect: aspect)
            let under = c.floorDirection(ndc: to, aspect: aspect)
            #expect(under != nil && SphereFrame.angle(under!, anchor) < 2e-4, "the grabbed point is under the pointer")
            #expect(abs(simd_length(c.pose.up) - 1) < 1e-5 && abs(simd_dot(c.pose.up, c.pose.heading)) < 1e-4, "the frames stay orthonormal")
        }
    }

    @Test func aGrabAtThePoleCarriesTheHeadingAlong() {
        var c = Self.camera(lat: 89, lon: 10, tilt: 0, distance: 80)
        let anchor = c.floorDirection(ndc: SIMD2(0.1, 0), aspect: 1)!
        c.grab(anchor: anchor, at: SIMD2(-0.2, 0.15), aspect: 1)
        #expect(c.floorDirection(ndc: SIMD2(-0.2, 0.15), aspect: 1).map { SphereFrame.angle($0, anchor) < 2e-4 } == true)
        #expect(simd_length(c.pose.heading - c.pose.heading).isZero, "no NaN")
        #expect(!c.pose.up.x.isNaN && !c.pose.heading.x.isNaN)
    }

    @Test func aZeroDragMovesNothing() {
        var c = Self.camera(tilt: 0.7, distance: 50)
        let before = c.pose
        let ndc = SIMD2<Float>(0.1, -0.1)
        c.grab(anchor: c.grabDirection(ndc: ndc, aspect: 1.3), at: ndc, aspect: 1.3)
        #expect(Self.close(c.pose.up, before.up, 1e-5) && Self.close(c.pose.heading, before.heading, 1e-5))
    }

    @Test func aPanFollowsTheFingersByTheContent() {
        var c = Self.camera(lat: 0, lon: 0, tilt: 0, distance: 70)
        let size = SIMD2<Float>(800, 500)
        let under = c.floorDirection(ndc: .zero, aspect: size.x / size.y)!
        c.pan(dx: 80, dy: 0, size: size)  // the fingers go right: the planet goes right, so the camera moves west
        let (_, lon) = SphereFrame.latLon(of: c.pose.up)
        #expect(lon < 0, "the content followed the fingers (the point that was at the centre is now to the right of it)")
        let screen = c.ndc(of: under * Self.rp, aspect: 1.6)!
        #expect(abs(screen.x - 2 * 80 / 800) < 2e-3 && abs(screen.y) < 2e-3)
    }

    @Test func aZoomAboutThePointerKeepsThePointUnderIt() {
        var c = Self.camera(lat: 35, lon: -60, tilt: 0.4, distance: 90)
        let ndc = SIMD2<Float>(0.2, 0.15)
        let before = c.floorDirection(ndc: ndc, aspect: 1.5)!
        c.zoom(by: 0.5, at: ndc, aspect: 1.5)
        #expect(abs(c.pose.distance - 45) < 1e-3)
        #expect(SphereFrame.angle(c.floorDirection(ndc: ndc, aspect: 1.5)!, before) < 3e-4)
        c.zoom(by: 1e-6, aspect: 1.5)
        #expect(c.pose.distance == SpherePose.minDistance, "the least distance")
        c.zoom(by: 1e6, aspect: 1.5)
        #expect(c.pose.distance <= c.maxDistance(aspect: 1.5) + 1e-3)
    }

    @Test func aTurnRotatesTheHeadingAndTiltsTheView() {
        var c = Self.camera(tilt: 0.2, distance: 50)
        let up = c.pose.up
        c.turn(dx: 100, dy: -50)
        #expect(Self.close(c.pose.up, up, 1e-6), "the point looked at does not move")
        #expect(abs(c.pose.bearing) > 0.3 && abs(simd_dot(c.pose.heading, up)) < 1e-5)
        #expect(c.pose.tilt > 0.2)
        c.turn(dx: 0, dy: 1e5)
        #expect(c.pose.tilt == 0)
    }

    // MARK: The standard views and the limits

    @Test func theHomeViewShowsThePlanetWholeNorthUp() {
        let aspect: Float = 1.6
        let pose = SphereCamera.home(planetRadius: Self.rp, lidRadius: Self.lid, centre: SIMD3(1, 1, 0.5), aspect: aspect)
        let c = SphereCamera(pose: pose, planetRadius: Self.rp, lidRadius: Self.lid)
        #expect(pose.tilt == 0 && pose.isNorthUp)
        // the lid's silhouette is inside the view: points of the limb project inside
        let eyeDir = simd_normalize(c.eye)
        let b = SphereFrame.basis(at: eyeDir)
        let limbAngle = asin(Self.lid / simd_length(c.eye))
        for k in 0..<12 {
            let phi = Float(k) * Float.pi / 6
            let side = cos(phi) * b.east + sin(phi) * b.north
            // the tangent from the eye to the lid sphere touches it at the angle acos(R / r) from the eye's direction
            let touch = simd_normalize(eyeDir * cos(Float.pi / 2 - limbAngle) + side * sin(Float.pi / 2 - limbAngle)) * Self.lid
            let ndc = c.ndc(of: touch, aspect: aspect)!
            #expect(abs(ndc.x) < 1 && abs(ndc.y) < 1, "the planet fits the window")
        }
        let top = c.ndc(of: c.pose.heading * Self.lid + c.pose.up * 0, aspect: aspect)
        _ = top
        #expect(c.pose.distance <= c.maxDistance(aspect: aspect), "Home is within the farthest view")
    }

    @Test func theLevelFollowsTheDistance() {
        #expect(SphereLevel.level(distance: 200, planetRadius: Self.rp, lidRadius: Self.lid) == .planet)
        #expect(SphereLevel.level(distance: 30, planetRadius: Self.rp, lidRadius: Self.lid) == .region)
        #expect(SphereLevel.level(distance: 6, planetRadius: Self.rp, lidRadius: Self.lid) == .organism)
    }

    @Test func theClipPlanesBoundThePlanetAtEveryDistance() {
        for distance: Float in [3, 20, 60, 200, 400] {
            let c = Self.camera(tilt: 0.5, distance: distance)
            let (near, far) = c.clipPlanes
            #expect(near > 0 && far > near)
            // the farthest point of the lid from the eye is inside the far plane
            #expect(far >= simd_length(c.eye) + Self.lid - 1e-3)
            #expect(near <= max(0.35 * (simd_length(c.eye) - Self.lid), 0.1) + 1e-4)
        }
    }

    @Test func northUpTurnsTheHeadingToNorth() {
        var pose = Self.camera(lat: 30, lon: 40).pose
        pose.heading = simd_normalize(SphereFrame.basis(at: pose.up).east)
        #expect(!pose.isNorthUp && abs(pose.bearing - .pi / 2) < 1e-4)
        #expect(pose.northUp().isNorthUp)
    }

    // MARK: Flights

    @Test func aFlightStartsAtTheStartEndsAtTheGoalAndStaysAboveTheFloor() {
        let from = SpherePose(lat: 10, lon: -80, tilt: 0.3, distance: 40)
        let to = SpherePose(lat: -45, lon: 120, tilt: 0.3, distance: 25)
        let flight = SphereFlight(from: from, to: to, planetRadius: Self.rp)
        #expect(flight.duration >= 0.5 && flight.duration <= 2.2)
        let first = flight.pose(at: 0), last = flight.pose(at: flight.duration + 1)
        #expect(Self.close(first.up, from.up, 1e-5) && abs(first.distance - from.distance) < 1e-2)
        #expect(Self.close(last.up, to.up) && last.distance == to.distance)
        var previous = SphereFrame.angle(from.up, to.up)
        for k in 0...40 {
            let p = flight.pose(at: flight.duration * Float(k) / 40)
            let c = SphereCamera(pose: p, planetRadius: Self.rp, lidRadius: Self.lid)
            #expect(c.altitude > 0 && p.distance >= SpherePose.minDistance)
            #expect(abs(simd_length(p.up) - 1) < 1e-4 && abs(simd_dot(p.up, p.heading)) < 1e-4)
            let remaining = SphereFrame.angle(p.up, to.up)
            #expect(remaining <= previous + 1e-4, "the way to the goal only shrinks")
            previous = remaining
        }
    }

    @Test func aLongFlightClimbsInTheMiddleAndAShortOneDoesNot() {
        let from = SpherePose(lat: 0, lon: 0, tilt: 0, distance: 40)
        let far = SphereFlight(from: from, to: SpherePose(lat: 0, lon: 170, tilt: 0, distance: 40), planetRadius: Self.rp)
        let near = SphereFlight(from: from, to: SpherePose(lat: 0, lon: 2, tilt: 0, distance: 40), planetRadius: Self.rp)
        #expect(far.pose(at: far.duration / 2).distance > 1.5 * 40)
        #expect(near.pose(at: near.duration / 2).distance < 1.2 * 40)
        let reduced = SphereFlight(from: from, to: SpherePose(lat: 0, lon: 170, tilt: 0, distance: 40), planetRadius: Self.rp, straight: true)
        #expect(reduced.duration == 0.25 && abs(reduced.pose(at: 0.125).distance - 40) < 1e-3)
    }

    @Test func aFlightThroughTheAntipodeDoesNotBreak() {
        let from = SpherePose(lat: 0, lon: 0, tilt: 0, distance: 40)
        let to = SpherePose(lat: 0, lon: 180, tilt: 0, distance: 40)
        let flight = SphereFlight(from: from, to: to, planetRadius: Self.rp)
        for k in 0...20 {
            let p = flight.pose(at: flight.duration * Float(k) / 20)
            #expect(!p.up.x.isNaN && !p.heading.x.isNaN && abs(simd_length(p.up) - 1) < 1e-4)
        }
    }
}

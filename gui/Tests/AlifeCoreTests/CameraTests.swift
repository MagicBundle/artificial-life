import Foundation
import Testing
import simd

@testable import AlifeCore

/// The camera's matrices and controls, checked by projecting points the way the shader will.
@Suite struct CameraTests {
    static let aspects: [Float] = [1, 16.0 / 9, 9.0 / 16]

    private func near(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ tolerance: Float = 1e-4) -> Bool {
        simd_length(a - b) <= tolerance
    }

    private func sample() -> Camera {
        Camera(target: SIMD3(60, 50, 12), distance: 90, yaw: 0.7, pitch: -0.4)
    }

    // MARK: Basis

    @Test(arguments: [Float(0), 0.7, -2.4, 3.1, -3.1], [-Float.pi / 2, -1, -0.3, 0, 0.5, 1.2, Float.pi / 2])
    func theBasisIsOrthonormalAndRightHanded(yaw: Float, pitch: Float) {
        let camera = Camera(target: .zero, distance: 10, yaw: yaw, pitch: pitch)
        let (f, r, u) = (camera.forward, camera.right, camera.up)
        #expect(abs(simd_length(f) - 1) < 1e-5 && abs(simd_length(r) - 1) < 1e-5 && abs(simd_length(u) - 1) < 1e-5)
        #expect(abs(dot(f, r)) < 1e-5 && abs(dot(f, u)) < 1e-5 && abs(dot(r, u)) < 1e-5)
        #expect(near(cross(r, u), -f), "right × up is the backward axis")
        #expect(r.z == 0, "the screen's x stays horizontal")
        if abs(pitch) < .pi / 2 - 1e-3 { #expect(u.z > 0, "up leans towards +z until the view is vertical") }
    }

    @Test func aLevelCameraLooksAlongPlusXWithPlusZUp() {
        let camera = Camera(target: .zero, distance: 5, yaw: 0, pitch: 0)
        #expect(near(camera.forward, SIMD3(1, 0, 0)) && near(camera.up, SIMD3(0, 0, 1)) && near(camera.right, SIMD3(0, -1, 0)))
        #expect(near(camera.position, SIMD3(-5, 0, 0)))
    }

    @Test func aVerticalViewIsWellDefined() {
        for pitch in [-Float.pi / 2, Float.pi / 2] {
            for yaw in [Float(0), 1, -2] {
                let camera = Camera(target: SIMD3(1, 2, 3), distance: 8, yaw: yaw, pitch: pitch)
                let view = camera.viewMatrix()
                #expect((0..<4).allSatisfy { c in (0..<4).allSatisfy { view[c][$0].isFinite } })
                #expect(abs(camera.up.z) < 1e-5, "straight up or down, the screen's up is horizontal")
                #expect(simd_length(view * SIMD4(camera.position, 1) - SIMD4(0, 0, 0, 1)) < 1e-3, "the eye is the origin")
            }
        }
    }

    // MARK: Matrices

    @Test(arguments: aspects)
    func theTargetIsAtTheCentreOfTheScreen(aspect: Float) throws {
        let camera = sample()
        let ndc = try #require(camera.ndc(of: camera.target, aspect: aspect))
        #expect(abs(ndc.x) < 1e-5 && abs(ndc.y) < 1e-5 && ndc.z > 0 && ndc.z < 1)
    }

    @Test(arguments: aspects)
    func screenRightIsPositiveXAndScreenUpIsPositiveY(aspect: Float) throws {
        let camera = sample()
        let centre = try #require(camera.ndc(of: camera.target, aspect: aspect))
        let right = try #require(camera.ndc(of: camera.target + camera.right * 5, aspect: aspect))
        let up = try #require(camera.ndc(of: camera.target + camera.up * 5, aspect: aspect))
        #expect(right.x > centre.x && abs(right.y - centre.y) < 1e-5)
        #expect(up.y > centre.y && abs(up.x - centre.x) < 1e-5)
    }

    @Test func aWiderViewportSqueezesTheHorizontalAxisOnly() throws {
        let camera = sample()
        let point = camera.target + camera.right * 7 + camera.up * 3
        let square = try #require(camera.ndc(of: point, aspect: 1))
        let wide = try #require(camera.ndc(of: point, aspect: 2))
        #expect(abs(wide.x - square.x / 2) < 1e-5 && abs(wide.y - square.y) < 1e-5 && abs(wide.z - square.z) < 1e-6)
    }

    @Test func theNearAndFarPlanesMapToZeroAndOne() throws {
        var camera = sample()
        camera.near = 2
        camera.far = 500
        let atNear = try #require(camera.ndc(of: camera.position + camera.forward * 2, aspect: 1))
        let atFar = try #require(camera.ndc(of: camera.position + camera.forward * 500, aspect: 1))
        #expect(abs(atNear.z) < 1e-4 && abs(atFar.z - 1) < 1e-4)
        let mid = try #require(camera.ndc(of: camera.position + camera.forward * 20, aspect: 1))
        #expect(mid.z > 0 && mid.z < 1, "depth grows with distance")
        #expect(atNear.z < mid.z && mid.z < atFar.z)
    }

    // MARK: Clip planes

    /// The eight corners of a 3D world's box, or the four of a 2D world's square.
    private func corners(size: Float, height: Float) -> [SIMD3<Float>] {
        var points: [SIMD3<Float>] = []
        for x in [Float(0), size] {
            for y in [Float(0), size] {
                for z in height > 0 ? [Float(0), height] : [0] { points.append(SIMD3(x, y, z)) }
            }
        }
        return points
    }

    @Test(arguments: [(Float(128), Float(24)), (1024, 0), (4096, 0), (100, 40)], aspects)
    func theWholeWorldIsInDepthAtHomeAndHoweverFarTheCameraIsPulledBack(world: (Float, Float), aspect: Float) throws {
        let (size, height) = world
        var camera = height > 0 ? Camera.overview(size: size, height: height, aspect: aspect)
                                : Camera.topDown(size: size, aspect: aspect)
        let points = corners(size: size, height: height) + [camera.target]
        // Zooming out by the factors a wheel gives, from the home pose to a hundred times as far.
        for factor in [Float(1), 2, 4, 10, 100] {
            let far = camera.distance * factor
            var pulled = camera
            pulled.dolly(by: factor)
            #expect(abs(pulled.distance - far) < far * 1e-5)
            for point in points {
                let ndc = try #require(pulled.ndc(of: point, aspect: aspect))
                #expect(ndc.z > 0 && ndc.z < 1, "\(point) is clipped at \(factor)x: depth \(ndc.z)")
            }
        }
        // Flying away (the target moves, the distance does not) is the same: the planes follow the eye.
        camera.move(forward: -3 * size)
        for point in points {
            let ndc = try #require(camera.ndc(of: point, aspect: aspect))
            #expect(ndc.z > 0 && ndc.z < 1, "\(point) is clipped after flying back")
        }
    }

    @Test(arguments: [(Float(128), Float(24)), (1024, 0), (4096, 0)], aspects)
    func theHomeDepthRangeIsTightEnoughForAnAgentToBeTold(world: (Float, Float), aspect: Float) {
        // A float32 depth value near 1 has steps of about 6e-8, and depth = 1 - near / z far from the near plane, so a step
        // is z^2 * 6e-8 / near world units at distance z. With the fixed planes the home pose of a 1024-wide world (near
        // 0.1) had a step of 0.75, most of an agent's radius of 1; fitted to the world it is under a thousandth.
        let (size, height) = world
        let camera = height > 0 ? Camera.overview(size: size, height: height, aspect: aspect)
                                : Camera.topDown(size: size, aspect: aspect)
        let (near, far) = camera.clipPlanes
        #expect(far / near < 40, "near \(near), far \(far)")
        let z = simd_length(camera.position - camera.target)
        let step = z * z * 6e-8 / near
        #expect(step < 0.01, "a depth step of \(step) world units at the centre")
    }

    @Test func aCameraWithoutASceneKeepsItsFixedPlanes() {
        var camera = sample()
        #expect(camera.clipPlanes.near == 0.1 && camera.clipPlanes.far == 10_000)
        camera.near = 2
        camera.far = 500
        camera.dolly(by: 3)
        #expect(camera.clipPlanes.near == 2 && camera.clipPlanes.far == 500)
    }

    @Test func insideTheWorldTheNearPlaneIsTheLeastOne() {
        var camera = Camera.overview(size: 128, height: 24)
        camera.dolly(by: 0.01)
        let (near, far) = camera.clipPlanes
        #expect(near == camera.near, "the eye is inside the bounding sphere: only the fixed least near plane applies")
        #expect(far > camera.distance)
    }

    @Test func aPointBehindTheEyeIsNotProjected() {
        let camera = sample()
        #expect(camera.ndc(of: camera.position - camera.forward * 3, aspect: 1) == nil)
        #expect(camera.ndc(of: camera.position, aspect: 1) == nil)
    }

    @Test func theViewAngleSetsTheScreenEdge() throws {
        let camera = Camera(target: .zero, distance: 10, yaw: 0, pitch: 0, fovY: 90 * .pi / 180)
        // At 90 degrees, a point as far above the target as the target is distant sits on the top edge.
        let top = try #require(camera.ndc(of: SIMD3(0, 0, 10), aspect: 1))
        #expect(abs(top.y - 1) < 1e-5)
    }

    // MARK: Starting poses

    @Test(arguments: aspects)
    func theOverviewShowsTheWholeBox(aspect: Float) throws {
        let (size, height): (Float, Float) = (320, 80)
        let camera = Camera.overview(size: size, height: height, aspect: aspect)
        for x in [Float(0), size] {
            for y in [Float(0), size] {
                for z in [Float(0), height] {
                    let ndc = try #require(camera.ndc(of: SIMD3(x, y, z), aspect: aspect), "corner \(x), \(y), \(z)")
                    #expect(abs(ndc.x) <= 1.0001 && abs(ndc.y) <= 1.0001 && ndc.z >= 0 && ndc.z <= 1,
                            "corner \(x), \(y), \(z) at \(ndc)")
                }
            }
        }
        let centre = try #require(camera.ndc(of: SIMD3(size / 2, size / 2, height / 2), aspect: aspect))
        #expect(abs(centre.x) < 1e-4 && abs(centre.y) < 1e-4)
    }

    @Test func theOverviewLooksFromTheSouthAboveTheFloorWithXToTheRight() {
        let camera = Camera.overview(size: 128, height: 24)
        #expect(near(camera.right, SIMD3(1, 0, 0)))
        #expect(camera.forward.y > 0 && camera.forward.z < 0, "into the world and down")
        #expect(camera.up.z > 0)
        #expect(camera.position.y < 0 && camera.position.z > 24, "outside the box, above its floor")
    }

    /// The 2D renderer draws column = x and row = y, so +x is to the right and +y is down on screen (DESIGN §20.1).
    @Test(arguments: aspects)
    func theTopDownCameraMatchesThe2DPNGOrientation(aspect: Float) throws {
        let size: Float = 128
        let camera = Camera.topDown(size: size, aspect: aspect)
        #expect(near(camera.right, SIMD3(1, 0, 0), 1e-4) && near(camera.up, SIMD3(0, -1, 0), 1e-4))
        #expect(near(camera.forward, SIMD3(0, 0, 1), 1e-4))
        #expect(camera.position.z < 0, "the eye is on the -z side of the plane")

        func screen(_ x: Float, _ y: Float) throws -> SIMD3<Float> {
            try #require(camera.ndc(of: SIMD3(x, y, 0), aspect: aspect))
        }
        let origin = try screen(0, 0), east = try screen(size, 0), south = try screen(0, size), far = try screen(size, size)
        #expect(origin.x < east.x, "x grows to the right")
        #expect(abs(origin.y - east.y) < 1e-5 && abs(origin.x - south.x) < 1e-5, "the sides are axis aligned")
        #expect(origin.y > south.y, "y grows downwards: the origin is at the top")
        #expect(origin.x < 0 && origin.y > 0 && far.x > 0 && far.y < 0, "the origin is the top-left corner")
        // The plane is a flat square seen face on: the map from the world to the screen is affine.
        let middle = try screen(size / 2, size / 2)
        #expect(abs(middle.x) < 1e-5 && abs(middle.y) < 1e-5)
        let quarter = try screen(size / 4, 3 * size / 4)
        #expect(abs(quarter.x - (origin.x + (east.x - origin.x) / 4)) < 1e-5)
        #expect(abs(quarter.y - (origin.y + (south.y - origin.y) * 3 / 4)) < 1e-5)
        for corner in [origin, east, south, far] { #expect(abs(corner.x) <= 1 && abs(corner.y) <= 1, "the world fits the view") }
    }

    /// The viewer starts a 3D stream at the overview of its box and a 2D stream face on.
    @Test(arguments: [("2d", 2), ("3d", 3)], aspects)
    func theHomePoseFollowsTheDimensionsOfTheStream(fixture: (String, Int), aspect: Float) throws {
        let events = try StreamFixture(kind: fixture.0).events()
        guard case .header(let header) = events[0] else { Issue.record("no header"); return }
        #expect(header.dims == fixture.1)
        let home = Camera.home(for: header, aspect: aspect)
        let size = Float(header.size)
        if fixture.1 == 3 {
            #expect(home == Camera.overview(size: size, height: Float(header.height), aspect: aspect))
        } else {
            #expect(home == Camera.topDown(size: size, aspect: aspect))
        }
    }

    // MARK: Controls

    @Test func orbitingKeepsTheTargetAndTheDistance() {
        var camera = sample()
        let target = camera.target, eye = camera.position
        camera.orbit(deltaYaw: 0.5, deltaPitch: 0.2)
        #expect(camera.target == target && camera.distance == 90)
        #expect(abs(simd_length(camera.position - target) - 90) < 1e-3)
        #expect(!near(camera.position, eye, 1))
        #expect(abs(camera.yaw - 1.2) < 1e-6 && abs(camera.pitch - -0.2) < 1e-6)
    }

    @Test func pitchStopsAtStraightUpAndDown() {
        var camera = sample()
        camera.orbit(deltaYaw: 0, deltaPitch: 10)
        #expect(camera.pitch == .pi / 2)
        camera.orbit(deltaYaw: 0, deltaPitch: -10)
        #expect(camera.pitch == -.pi / 2)
        camera.look(deltaYaw: 0, deltaPitch: 10)
        #expect(camera.pitch == .pi / 2)
    }

    @Test func yawWrapsIntoOneTurn() {
        var camera = sample()
        camera.orbit(deltaYaw: 7, deltaPitch: 0)
        #expect(camera.yaw >= -.pi && camera.yaw < .pi && abs(camera.yaw - (0.7 + 7 - 2 * .pi)) < 1e-5)
        camera.orbit(deltaYaw: -20, deltaPitch: 0)
        #expect(camera.yaw >= -.pi && camera.yaw < .pi)
        #expect(Camera(target: .zero, distance: 1, yaw: .pi, pitch: 0).yaw == -.pi)
    }

    @Test func lookingKeepsTheEyeAndTurnsTheView() {
        var camera = sample()
        let eye = camera.position, forward = camera.forward
        camera.look(deltaYaw: 0.4, deltaPitch: -0.1)
        #expect(near(camera.position, eye, 1e-3))
        #expect(dot(camera.forward, forward) < 0.999 && abs(camera.distance - 90) < 1e-6)
        #expect(near(camera.target, eye + camera.forward * 90, 1e-3))
    }

    /// Dragging by (dx, dy) pixels moves the picture with the cursor: what was at the centre ends up there.
    @Test(arguments: aspects)
    func panningFollowsTheCursor(aspect: Float) throws {
        var camera = sample()
        let viewportHeight: Float = 600
        let viewportWidth = viewportHeight * aspect
        let held = camera.target
        camera.pan(dx: 120, dy: -45, viewportHeight: viewportHeight)
        let ndc = try #require(camera.ndc(of: held, aspect: aspect))
        #expect(abs(ndc.x - 2 * 120 / viewportWidth) < 1e-4, "dragged right by 120 pixels")
        #expect(abs(ndc.y - 2 * 45 / viewportHeight) < 1e-4, "dragged up by 45 pixels (the window's y is down)")
        #expect(camera.distance == 90 && camera.yaw == sample().yaw && camera.pitch == sample().pitch)
        var unchanged = sample()
        unchanged.pan(dx: 5, dy: 5, viewportHeight: 0)
        #expect(unchanged == sample(), "an empty viewport pans nothing")
    }

    @Test func movingTranslatesTheEyeAndTheTarget() {
        var camera = sample()
        let (eye, target, f, r, u) = (camera.position, camera.target, camera.forward, camera.right, camera.up)
        camera.move(forward: 4, right: -2, up: 1)
        #expect(near(camera.position, eye + 4 * f - 2 * r + u, 1e-4) && near(camera.target, target + 4 * f - 2 * r + u, 1e-4))
        #expect(camera.distance == 90)
    }

    @Test func dollyingScalesTheDistanceWithinLimits() {
        var camera = sample()
        camera.dolly(by: 0.5)
        #expect(camera.distance == 45)
        camera.dolly(by: 1e-9)
        #expect(camera.distance == Camera.minDistance)
        camera.dolly(by: 0)
        camera.dolly(by: -2)
        camera.dolly(by: .nan)
        #expect(camera.distance == Camera.minDistance, "a factor that is not positive and finite is ignored")
    }

    @Test func orbitAndFlyEditTheSamePose() {
        // Fly forward and orbit back: the pose is one value, so a mode switch needs no conversion.
        var camera = sample()
        camera.move(forward: 10)
        camera.orbit(deltaYaw: 0.3, deltaPitch: 0)
        #expect(near(camera.target, sample().target + 10 * sample().forward, 1e-4))
    }
}

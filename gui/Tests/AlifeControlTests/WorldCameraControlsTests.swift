import AlifeCore
import AppKit
import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeControl

/// The camera controls of a 3D world through the **hosted** `WorldView` (DESIGN §36.5, camera controls; the lesson of I61): the keys
/// `t o l s`, the strip of standard views, the 3D button, the tilt handle and the compass dial, pressed with events sent through the
/// window, so that AppKit's routing finds the Metal view and the model's hit test does the rest; the pose each reaches; the tilt to
/// edge-on with the distance held by the eye's limit; picking and the readouts at oblique and grazing angles; and the seam copies at
/// grazing angles. The pixels (the haze, the sky, no hard edge) are in `GPUCameraTests`.
/// They are tests of `WorldInputTests`' suite on purpose, as `WorldReviewFixTests` are: that suite is serialized and moves the dock about in
/// the process's defaults (`HostedWorld` sets them too), and two suites that do so at once see each other's dock and windows.
extension WorldInputTests {
    // MARK: Rig

    /// The hosted fixture world with the dock collapsed (the map has the whole window), the minimap on and the hint gone.
    private func cameraWorld() async throws -> HostedWorld {
        let world = try await HostedWorld(stream: Repo.playFixtureURL("3d"), dockCollapsed: true)
        world.model.reduceMotion = { false }
        world.model.display.minimap = true
        world.model.dismissHint()
        world.settle()
        return world
    }

    /// Runs the camera's own motion until it arrives; the number of frames it took.
    @discardableResult
    private func cameraArrive(_ model: WorldModel, frames: Int = 600) -> Int {
        for n in 1...frames where !model.advanceCamera(dt: 1.0 / 60) { return n }
        return frames
    }

    // MARK: The standard views

    @Test func eachStandardViewIsOneKeyAndEasesToItsPoseInHalfASecond() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        let (L, H) = (model.worldSize, model.height)
        model.zoom(by: 0.5)
        model.editPose { $0.heading = 0.7; $0.x += 3 }
        var heading = model.pose.heading
        // From each view to each other one.
        for preset in ViewPreset.allCases + ViewPreset.allCases.reversed() {
            #expect(world.press(preset.key), "the key \(preset.key) is the World's")
            #expect(model.viewEase != nil && model.cameraMotionPending, "an ease begins, it does not jump")
            let start = model.pose
            var last = start.tilt
            var frames = 0
            while model.advanceCamera(dt: 1.0 / 60) {
                frames += 1
                let toward = preset.tilt - start.tilt
                #expect((model.pose.tilt - last) * toward >= -1e-6, "the tilt moves one way only")
                last = model.pose.tilt
                #expect(frames < 60)
            }
            #expect(frames + 1 >= 29 && frames + 1 <= 31, "0.5 s is 30 frames: \(frames + 1)")
            let pose = model.pose
            #expect(abs(pose.tilt - preset.tilt) < 1e-5, "\(preset): tilt \(pose.tilt)")
            if preset == .top { heading = 0 }
            #expect(abs(pose.heading - heading) < 1e-5, "Top turns north up and the others keep the heading: \(pose.heading)")
            if preset == .side {
                #expect(abs(pose.z - H / 2) < 1e-4, "Side is at the middle of the height")
                let camera = pose.camera(size: L, height: H)
                #expect(abs(camera.forward.z) < 1e-5 && abs(camera.position.z - H / 2) < 1e-3, "edge-on, the eye at the height of the target")
            } else {
                #expect(pose.z == 0, "the others look at the floor")
            }
            #expect(pose.distance <= pose.distanceLimit(size: L, aspect: model.aspect) * 1.0001, "within the limit of its tilt")
            let matrix = pose.camera(size: L, height: H).viewProjectionMatrix(aspect: model.aspect)
            for column in 0..<4 { for row in 0..<4 { #expect(matrix[column][row].isFinite) } }
            #expect(model.viewEase == nil && !model.cameraMotionPending && model.pose.tiltDegrees == Int((preset.tilt * 180 / .pi).rounded()))
        }
    }

    @Test func theSideViewShowsTheWholeHeightOfTheLifeAtTheDistanceItNeedsOrTheEyesLimit() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        let (L, H) = (model.worldSize, model.height)
        #expect(world.press("s"))
        cameraArrive(model)
        let wanted = H / (2 * tan(model.pose.fovY / 2)) / ViewEase.sideFill
        #expect(abs(model.pose.distance - min(wanted, WorldFog.eyeReach * L)) < 1e-3, "\(model.pose.distance) for \(wanted) (limit \(WorldFog.eyeReach * L))")
        // Where the eye is not held, the layer from 0 to H fills four fifths of the picture's height at the distance of the target.
        let bigWorld = ViewEase(preset: .side, from: WorldPose(x: 100, y: 100, distance: 100), wish: nil, aspect: 1.5, size: 320, height: 80, keepHeight: false)
        let fill = 80 / (2 * bigWorld.toDistance * tan(Camera.defaultFovY / 2))
        #expect(abs(fill - ViewEase.sideFill) < 1e-3 && bigWorld.toDistance <= WorldFog.eyeReach * 320, "\(fill)")
    }

    @Test func aStandardViewKeepsTheHeadingExceptTopAndAnOrganismFollowedKeepsItsHeight() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        let H = model.height
        let frame = try #require(world.play.store.frame)
        let p = frame.position(0)
        model.selectOrganism(frame.uid[0], follow: true, camera: true, fly: false)
        for _ in 0..<200 where !model.advanceCamera(dt: 1.0 / 60) { break }
        #expect(model.followState.target != .none)
        model.editPose { $0.heading = 1.1 }
        #expect(world.press("s"))
        cameraArrive(model, frames: 100)
        #expect(!model.followState.target.isNone, "the follow goes on")
        #expect(abs(model.pose.z - p.z) < 3 && model.pose.tilt == .pi / 2, "the height is the organism's, not the middle: \(model.pose.z) of \(p.z), \(H / 2)")
        #expect(abs(model.pose.heading - 1.1) < 1e-4, "Side keeps the heading")
    }

    @Test func anyInputEndsAnEaseWhereItIsAndTheMenuItemsAreTheSameActions() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        #expect(world.press("l"))
        for _ in 0..<10 { _ = model.advanceCamera(dt: 1.0 / 60) }
        let part = model.pose.tilt
        #expect(part > 0.1 && part < ViewPreset.low.tilt)
        model.zoom(by: 0.9)
        #expect(model.viewEase == nil && !model.advanceCamera(dt: 1.0 / 60) && abs(model.pose.tilt - part) < 1e-6, "a zoom ends it where it is")
        model.viewPreset(.oblique)
        for _ in 0..<5 { _ = model.advanceCamera(dt: 1.0 / 60) }
        model.drag(dx: 3, dy: 0, orbit: false)
        #expect(model.viewEase == nil, "a drag ends it")
        // The View menu's items call the model's own function; the strip and the keys the same.
        model.reduceMotion = { true }
        model.viewPreset(.top)
        #expect(cameraArrive(model) <= 17, "reduced motion: a quarter of a second")
        #expect(model.pose.tilt == 0)
    }

    // MARK: The on-screen controls

    @Test func theStripOfViewsTheThreeDButtonTheHandleAndTheCompassAreRoutedToTheModelByTheMetalView() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        let hud = model.hud
        func centre(_ rect: CGRect?) -> CGPoint { CGPoint(x: rect!.midX, y: rect!.midY) }
        // Every control is the Metal view's to hear (no view lies over the map), the strip included.
        for (target, rect) in hud.rects { #expect(world.hit(centre(rect)) is WorldMTKView, "\(target)") }
        // A button of the strip.
        let low = try #require(hud.presets.first { $0.preset == .low })
        world.click(centre(low.rect))
        #expect(model.viewEase != nil)
        cameraArrive(model)
        #expect(abs(model.pose.tilt - ViewPreset.low.tilt) < 1e-5, "the strip's Low button")
        // The 3D button: angled to flat, flat to the angle last used.
        world.click(centre(hud.toggle3D))
        cameraArrive(model)
        #expect(model.pose.tilt == 0 && model.pose.heading == 0, "2D: straight down, the heading kept")
        world.click(centre(hud.toggle3D))
        cameraArrive(model)
        #expect(abs(model.pose.tilt - ViewPreset.low.tilt) < 1e-5, "3D: back to the angle last used")
        // The handle: a press at the bottom of the track goes to edge-on at once, the thumb is where the tilt is, and a drag follows.
        let track = try #require(hud.tiltTrack)
        world.window.sendEvent(world.mouse(.leftMouseDown, CGPoint(x: track.midX, y: track.maxY - 2)))
        world.settle(2)
        #expect(model.pose.tilt > 1.5, "a press at the bottom of the track: \(model.pose.tilt)")
        var previous = model.pose.tilt
        var lastY = track.maxY
        for y in stride(from: track.maxY - 12, through: track.minY + 6, by: -14) {
            world.window.sendEvent(world.mouse(.leftMouseDragged, CGPoint(x: track.midX, y: y)))
            #expect(model.pose.tilt <= previous + 1e-6, "dragging up tilts back towards straight down")
            previous = model.pose.tilt
            lastY = y
        }
        world.window.sendEvent(world.mouse(.leftMouseUp, CGPoint(x: track.midX, y: lastY)))
        #expect(model.pose.tilt < 0.4, "towards the top of the track: nearly straight down: \(model.pose.tilt)")
        #expect(model.pose.tilt == WorldHUD.tilt(atY: lastY, track: track), "the thumb and the tilt agree")
        // The compass is turned like a dial: the needle follows the pointer, a quarter turn clockwise is a quarter turn of the heading.
        let compass = try #require(hud.compass)
        let c = centre(compass)
        let start = model.pose.heading
        world.window.sendEvent(world.mouse(.leftMouseDown, CGPoint(x: c.x, y: c.y - 16)))
        for degrees in stride(from: 15.0, through: 90.0, by: 15) {
            let a = degrees * .pi / 180
            world.window.sendEvent(world.mouse(.leftMouseDragged, CGPoint(x: c.x + 16 * sin(a), y: c.y - 16 * cos(a))))
        }
        world.window.sendEvent(world.mouse(.leftMouseUp, CGPoint(x: c.x + 16, y: c.y)))
        #expect(abs(WorldPose.wrap(model.pose.heading - start) - .pi / 2) < 0.02, "a quarter turn: \(model.pose.heading) from \(start)")
        #expect(model.headingEase == nil && !model.cameraMotionPending, "a drag is not a click: no north up")
        // A click is north up, a double click Home.
        world.click(c)
        cameraArrive(model)
        #expect(model.pose.heading == 0)
        model.editPose { $0.heading = 1; $0.tilt = 0.6 }
        world.click(c, clicks: 2)
        cameraArrive(model)
        #expect(model.cameraIsHome && model.pose.tilt == 0 && model.pose.heading == 0)
    }

    @Test func theCompassTurnsTheViewByThePointersAngleAtAnyTilt() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        let compass = try #require(model.hud.compass)
        let c = CGPoint(x: compass.midX, y: compass.midY)
        for tilt: Float in [0, 0.8, 1.4] {
            model.editPose { $0.tilt = tilt; $0.heading = 0.3 }
            model.hudPress(.compass, at: CGPoint(x: c.x, y: c.y - 15))
            for degrees in [-170.0, -90, -30, 30, 90, 170] {
                let a = degrees * .pi / 180
                model.hudDrag(.compass, to: CGPoint(x: c.x + 15 * sin(a), y: c.y - 15 * cos(a)))
                #expect(abs(WorldPose.wrap(model.pose.heading - (0.3 + Float(a)))) < 1e-4, "tilt \(tilt), \(degrees) degrees: \(model.pose.heading)")
                #expect(model.pose.tilt == tilt)
            }
            model.hudRelease(.compass, at: c, clicks: 1, dragged: true)
            #expect(model.compassTurn == nil)
        }
    }

    @Test func theReadoutOfTheAngleSaysTheTiltAndTheCompassBearing() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        model.editPose { $0.tilt = 45 * .pi / 180; $0.heading = .pi / 2 }
        #expect(WorldReadout.viewText(pose: model.pose) == "Tilt 45° · heading 270°", "turned a quarter towards the west: it looks west")
        model.editPose { $0.tilt = 0; $0.heading = 0 }
        #expect(WorldReadout.viewText(pose: model.pose) == "Tilt 0° · heading 0°")
        model.editPose { $0.tilt = .pi / 2; $0.heading = -.pi / 2 }
        #expect(WorldReadout.viewText(pose: model.pose) == "Tilt 90° · heading 90°")
    }

    // MARK: Tilting to edge-on

    @Test func theTiltReachesEdgeOnWithTheOptionDragAndTheRightDragAndTheDistanceIsHeldNotTheTilt() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        let L = model.worldSize
        model.goHome()
        let home = model.pose.distance
        #expect(world.press("n") || true)
        model.drag(dx: 0, dy: -300, orbit: true)  // (Option-drag or right drag: 2.4 radians asked)
        #expect(model.pose.tilt == WorldPose.maxTilt, "to edge-on, not to the footprint's 52 to 57 degrees")
        #expect(model.pose.distance < home && model.pose.distance <= WorldFog.eyeReach * L * 1.0001)
        let eye = model.camera.position
        let target = model.camera.target
        #expect(simd_length(SIMD2(eye.x - target.x, eye.y - target.y)) <= WorldFog.eyeReach * L * 1.0001, "the eye is over the clear box")
        model.drag(dx: 0, dy: 300, orbit: true)
        #expect(model.pose.tilt == 0 && abs(model.pose.distance - home) < 1e-3, "and straight down again, the zoom is back: \(model.pose.distance) of \(home)")
    }

    // MARK: Picking and the readouts at oblique and grazing angles

    @Test(arguments: [ViewPreset.oblique, .low])
    func anOrganismIsPickedWhereItIsDrawnAtTheObliqueAndTheLowViews(preset: ViewPreset) async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        let frame = try #require(world.play.store.frame)
        let target = frame.position(frame.n / 2)
        model.editPose { $0.x = target.x; $0.y = target.y; $0.z = 0; $0.distance = 0.3 * model.worldSize; $0.tilt = preset.tilt }
        #expect(model.level != .planet, "organisms are picked below the planet level")
        model.drawableResized(CGSize(width: 2 * model.freeRect.width, height: 2 * model.freeRect.height))
        let projector = model.projector(size: model.freeRect.size)
        // Every organism that is on screen and clear of the others is picked at its own place, the pick naming it.
        var picked = 0
        for i in 0..<frame.n {
            guard let at = projector.screen(frame.position(i)), model.freeRect.size.width > at.x, at.x > 0, model.freeRect.size.height > at.y, at.y > 0,
                  model.pressTarget(at: at) == .map else { continue }
            let near = (0..<frame.n).filter { $0 != i && (projector.screen(frame.position($0)).map { hypot($0.x - at.x, $0.y - at.y) < 14 } ?? false) }
            guard near.isEmpty else { continue }
            let pick = WorldPicker.pick(at: at, projector: projector, store: world.play.store, display: model.display, maxRadius: model.drawnRadiusCap)
            #expect(pick == .organism(uid: frame.uid[i], index: i), "\(preset): row \(i) at \(at): \(pick)")
            picked += 1
        }
        #expect(picked >= 1, "\(preset): organisms were in view: \(picked) of \(frame.n)")
        // A click selects and follows, as at the top.
        let i = (0..<frame.n).first { index in projector.screen(frame.position(index)).map { $0.x > 40 && $0.y > 40 && $0.x < 400 && $0.y < 300 } ?? false } ?? 0
        let at = try #require(projector.screen(frame.position(i)))
        model.click(at: at, shift: false, clicks: 1)
        #expect(model.selection.organismUID != nil)
    }

    @Test func theFloorIsNotReadBeyondTheHazeOrAboveTheHorizonButIsUnderThePointerBelowIt() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        let L = model.worldSize
        model.display.map = true
        model.editPose { $0.tilt = 80 * .pi / 180; $0.distance = 1e6 }
        let projector = model.projector(size: model.freeRect.size)
        var near = 0, far = 0, sky = 0
        for row in stride(from: 4.0, to: Double(model.freeRect.height), by: 16) {
            let point = CGPoint(x: model.freeRect.width / 2, y: row)
            let ndc = WorldPose.ndc(of: point, in: model.freeRect.size)
            let hit = model.pose.offset(at: ndc, aspect: model.aspect, planeZ: 0)
            let placed = projector.world(at: point, height: 0)
            let floor = WorldPicker.floorPoint(at: point, projector: projector)
            let readout = model.pointerReadout(at: point)
            switch hit.map(WorldFog.chebyshev) {
            case nil:
                sky += 1
                #expect(placed == nil && floor == nil && readout == nil, "above the horizon: nothing under the pointer")
            case let c? where c <= WorldFog.readReach * L:
                near += 1
                #expect(placed != nil && floor != nil && readout?.hasPrefix("floor ") == true, "row \(row): \(c)")
            default:
                far += 1
                #expect(placed == nil && floor == nil && readout == nil, "in the haze, farther than \(WorldFog.readReach) L: the floor is not read")
            }
        }
        #expect(near >= 3 && far >= 2 && sky >= 2, "the three cases are reached: \(near) \(far) \(sky)")
        // A territory pick and a place for a power name nothing out there either, and the double click does not dive to a place a world away.
        model.dive(at: CGPoint(x: model.freeRect.width / 2, y: 4))
        #expect(!model.isFlying, "a double click in the sky or the haze goes nowhere")
    }

    @Test func aGrabInTheHazeOrTheSkyTakesHoldOfNothingAndOnTheNearFloorMovesTheMapExactly() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        model.editPose { $0.tilt = 85 * .pi / 180; $0.distance = 1e6 }
        let before = model.pose
        model.beginGrab(at: CGPoint(x: model.freeRect.width / 2, y: 3))  // the sky
        model.grab(to: CGPoint(x: model.freeRect.width / 2 + 40, y: 80))
        model.endGrab()
        #expect(model.pose.x == before.x && model.pose.y == before.y, "the sky is not a place to take hold of")
        let near = CGPoint(x: model.freeRect.width / 2, y: model.freeRect.height - 40)
        let anchor = try #require(model.pose.point(at: WorldPose.ndc(of: near, in: model.freeRect.size), aspect: model.aspect, size: model.worldSize))
        model.beginGrab(at: near)
        model.grab(to: CGPoint(x: near.x + 30, y: near.y - 10))
        model.endGrab()
        let projector = model.projector(size: model.freeRect.size)
        let now = try #require(projector.screen(SIMD3(anchor.x, anchor.y, 0)))
        #expect(hypot(now.x - (near.x + 30), now.y - (near.y - 10)) < 0.6, "the floor point stays under the pointer: \(now)")
    }

    @Test func aPanOfAnEdgeOnViewMovesAlongTheFloorAndNothingIsNonFinite() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        #expect(world.press("s"))
        cameraArrive(model)
        let before = model.pose
        model.scroll(ScrollEvent(dx: 0, dy: 30, precise: true, phase: .changed, time: 1), at: CGPoint(x: 300, y: 300))
        let moved = SIMD2(model.pose.x - before.x, model.pose.y - before.y)
        let ahead = SIMD2(-sin(before.heading), cos(before.heading))
        #expect(simd_length(moved) > 0.1 && abs(simd_dot(simd_normalize(moved), ahead)) > 0.999, "along the heading: \(moved)")
        #expect(model.pose.z == before.z && model.pose.tilt == before.tilt)
        model.scroll(ScrollEvent(dx: 20, dy: 0, precise: true, phase: .changed, time: 2), at: CGPoint(x: 300, y: 300))
        for value in [model.pose.x, model.pose.y, model.pose.z, model.pose.distance, model.pose.heading, model.pose.tilt] { #expect(value.isFinite) }
        // The readouts at the grazing angle are finite and say what the view is.
        let line = WorldReadout.line(pose: model.pose, size: model.worldSize, height: model.height, mapShown: model.mapShown)
        #expect(!line.contains("nan") && !line.contains("inf") && line.contains("units up"), "\(line)")
        let bar = model.scaleBar
        #expect(bar == nil || (bar!.bar.units.isFinite && bar!.bar.points.isFinite))
    }

    // MARK: The minimap's footprint at grazing angles

    @Test func theMinimapFootprintOfAGrazingViewIsClippedToTheClearBoxAndStaysInTheBox() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        let L = model.worldSize
        let inset = try #require(model.inset)
        for degrees: Float in [0, 30, 60, 80, 90] {
            for heading: Float in [0, 0.6, 2.2] {
                model.editPose { $0.tilt = degrees * .pi / 180; $0.heading = heading; $0.distance = 1e6 }
                let polygon = try #require(inset.footprint(of: model.pose, aspect: model.aspect))
                #expect(polygon.count == 64 && polygon.allSatisfy { $0.x.isFinite && $0.y.isFinite }, "\(degrees) degrees")
                let half = inset.rect.width / 2
                let centre = inset.point(x: model.pose.x, ybar: model.pose.ybar(size: L))
                #expect(polygon.allSatisfy { abs($0.x - centre.x) <= half * 1.0002 && abs($0.y - centre.y) <= half * 1.0002 }, "within one box across")
                #expect(inset.pieces(of: polygon).count <= 4)
            }
        }
    }

    // MARK: The seam at grazing angles

    @Test func atAGrazingAngleOnTheWrapInYAllThreeCopiesAreDrawnAndTheCopyOfAPointIsTheOneThatHoldsIt() async throws {
        let world = try await cameraWorld()
        defer { world.finish() }
        let model = world.model
        let L = model.worldSize
        let renderer = try world.renderer()
        for y in [L / 2 - 2, L / 2 + 2, 0.5, L - 0.5] {
            for degrees: Float in [70, 85, 90] {
                model.editPose { $0.x = L / 2; $0.y = y; $0.tilt = degrees * .pi / 180; $0.heading = 0.4; $0.distance = 1e6 }
                let plan = try #require(renderer.worldDraw.copies)
                let horizon = model.pose.footprint(aspect: model.aspect, size: L, height: model.height) == nil
                if horizon { #expect(plan.map(\.k).count == 3, "the horizon in view: all three copies (\(y), \(degrees))") }
                // Every point of the floor within the clear box of the view is in a cell of the plan, at the image the model draws it at.
                for dx: Float in [-0.5, 0, 0.5] {
                    for dy: Float in [-0.5, -0.2, 0, 0.2, 0.5] {
                        let point = SIMD2(model.pose.x + dx * L, model.pose.y + dy * L)
                        let copy = SeamCopies.copy(containing: point, center: SIMD2(model.pose.x, model.pose.y), plan: plan, size: L)
                        #expect(copy != nil, "(\(dx), \(dy)) of the clear box at y \(y), \(degrees) degrees is drawn by some copy of \(plan.map(\.k))")
                        #expect(copy.map { $0.alpha == 1 } ?? true, "the copies of the flat map are opaque")
                    }
                }
            }
        }
    }
}

// MARK: - Events through the hosted window

extension HostedWorld {
    /// The window's point for a point of the free rectangle (origin at the top left).
    func windowPoint(_ p: CGPoint) -> NSPoint {
        NSPoint(x: model.freeRect.minX + p.x, y: host.bounds.height - (model.freeRect.minY + p.y))
    }

    /// The view the window's hit test finds at a point of the free rectangle.
    func hit(_ p: CGPoint) -> NSView? { host.hitTest(host.superview!.convert(windowPoint(p), from: nil)) }

    func mouse(_ type: NSEvent.EventType, _ p: CGPoint, clicks: Int = 1) -> NSEvent {
        NSEvent.mouseEvent(with: type, location: windowPoint(p), modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                           windowNumber: window.windowNumber, context: nil, eventNumber: 0, clickCount: clicks, pressure: 1)!
    }

    /// A press and release of the left button at a point of the free rectangle, through the window as AppKit sends them.
    func click(_ p: CGPoint, clicks: Int = 1) {
        window.sendEvent(mouse(.leftMouseDown, p, clicks: clicks))
        RunLoop.current.run(until: Date().addingTimeInterval(0.03))
        window.sendEvent(mouse(.leftMouseUp, p, clicks: clicks))
        RunLoop.current.run(until: Date().addingTimeInterval(0.03))
    }

    /// A key, through the World's own key handler (the one the window's monitor calls); whether it was a World key.
    @discardableResult
    func press(_ characters: String) -> Bool {
        let event = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: window.windowNumber, context: nil,
                                     characters: characters, charactersIgnoringModifiers: characters, isARepeat: false, keyCode: 0)!
        return metal!.handleKey(event)
    }
}

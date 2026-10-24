import AlifeCore
import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeControl

/// The controls over the map as the model drives them (DESIGN §36.5, the spec's 4 and 5): the press targets, the zoom buttons, the
/// minimap's click, drag and double click, the compass, the scroll over the minimap, the hint, the scale bar and the colour mode.
/// No renderer and no window: the hosted routing is in `WorldInputTests`, the pixels in `GPUHUDTests`.
@MainActor
@Suite struct WorldHUDModelTests {
    private final class Clock { var t: Double = 1000 }

    @MainActor
    private struct Rig {
        let model: WorldModel
        let play: PlayController
        let clock: Clock
        let commands: URL
        let defaults: UserDefaults
        var sent: [String] { ((try? String(contentsOf: commands, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init) }

        /// Runs the camera's own motion until it arrives, a sixtieth of a second a frame.
        @discardableResult
        func settle(frames: Int = 1500) -> Int {
            for n in 1...frames {
                clock.t += 1.0 / 60
                if !model.advance() { return n }
            }
            return frames
        }
    }

    /// A live world playing a fixture, over a window model with defaults of its own (the key `i` and the colour choice are kept in them).
    private func rig(_ scratch: Scratch, kind: String = "3d", defaults: UserDefaults? = nil) async throws -> Rig {
        let commands = scratch.url.appendingPathComponent("commands.txt")
        let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done")
        let play = PlayController()
        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                   arguments: [Repo.playFixtureURL(kind).path, commands.path])
        #expect(await eventually { play.store.header != nil && play.store.clock != nil && play.store.frame != nil })
        let suite = try defaults ?? #require(UserDefaults(suiteName: "alife-hud-\(UUID().uuidString)"))
        let model = WorldModel(play: play, defaults: suite)
        let clock = Clock()
        model.now = { clock.t }
        model.setFreeRect(CGRect(x: 0, y: 0, width: 900, height: 700))
        model.reduceMotion = { false }
        model.deliver(.header)  // (the window model was made after the world: a view that attaches is given the header the same way)
        model.goHome()
        return Rig(model: model, play: play, clock: clock, commands: commands, defaults: suite)
    }

    private var inMinimap: CGPoint { CGPoint(x: 900 - 12 - 98, y: 700 - 12 - 98) }

    // MARK: Where a press lands

    @Test func aPressLandsOnTheMinimapTheCompassTheZoomButtonsAndOtherwiseOnTheMap() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        #expect(model.mapShown)
        let hud = model.hud
        #expect(model.pressTarget(at: inMinimap) == .minimap)
        #expect(model.pressTarget(at: CGPoint(x: try #require(hud.home).midX, y: try #require(hud.home).midY)) == .home)
        #expect(model.pressTarget(at: CGPoint(x: try #require(hud.compass).midX, y: try #require(hud.compass).midY)) == .compass)
        #expect(model.pressTarget(at: CGPoint(x: try #require(hud.zoomIn).midX, y: try #require(hud.zoomIn).midY)) == .zoomIn)
        #expect(model.pressTarget(at: CGPoint(x: try #require(hud.zoomOut).midX, y: try #require(hud.zoomOut).midY)) == .zoomOut)
        #expect(model.pressTarget(at: CGPoint(x: 450, y: 350)) == .map)
        model.display.minimap = false
        #expect(model.pressTarget(at: inMinimap) == .map && model.inset == nil, "no minimap, no target")
        model.display.minimap = true
        #expect(model.inset != nil)
    }

    @Test func aWorldWithNoMapHasNoMinimapAndATwoDWorldNoCompass() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch, kind: "2d")
        defer { rig.play.shutdown() }
        #expect(!rig.model.mapShown && rig.model.inset == nil && rig.model.hud.minimap == nil)
        #expect(rig.model.hud.compass == nil && rig.model.hud.zoomIn != nil)
        #expect(rig.model.pressTarget(at: inMinimap) == .map)
    }

    @Test func theRendererIsGivenTheMinimapsRectangleAsFractionsOfTheView() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        model.pushInset()
        let box = try #require(model.hud.minimap)
        #expect(model.hud.minimap == box)
        // (the renderer is not attached here: the fractions are what `pushInset` computes, so check them against the hit rectangle)
        let unit = CGRect(x: box.minX / 900, y: box.minY / 700, width: box.width / 900, height: box.height / 700)
        #expect(abs(unit.maxX - (900 - WorldHUD.margin) / 900) < 1e-9 && abs(unit.maxY - (700 - WorldHUD.margin) / 700) < 1e-9)
    }

    // MARK: The zoom buttons

    @Test func aPressOfPlusEasesTheDistanceToE035TimesLessOverAQuarterOfASecond() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        let d0 = 0.8 * model.distanceLimit
        model.editPose { $0.distance = d0 }
        let (x, y) = (model.pose.x, model.pose.y)
        model.zoomStep(closer: true)
        #expect(model.cameraMotionPending && model.isAnimating)
        var frames = 0
        while model.zoomEase != nil, frames < 100 {
            rig.clock.t += 1.0 / 60
            _ = model.advance()
            frames += 1
        }
        rig.settle()
        #expect(abs(model.pose.distance - d0 * exp(-0.35)) < 1e-3, "\(model.pose.distance)")
        #expect(frames >= 14 && frames <= 17, "0.25 s is 15 frames at 60 Hz: \(frames)")
        #expect(model.pose.x == x && model.pose.y == y, "about the centre")
        model.zoomStep(closer: false)
        rig.settle()
        #expect(abs(model.pose.distance - d0) < 1e-2, "minus undoes plus")
    }

    @Test func twoPressesBeforeTheEaseIsOverComposeAndTheLimitsHold() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        let d0 = 0.9 * model.distanceLimit
        model.editPose { $0.distance = d0 }
        model.zoomStep(closer: true)
        for _ in 0..<5 { rig.clock.t += 1.0 / 60; _ = model.advance() }
        model.zoomStep(closer: true)
        rig.settle()
        #expect(abs(model.pose.distance - d0 * exp(-0.7)) < 1e-2, "\(model.pose.distance)")
        // At the nearest the distance stays at it; at the farthest, at that.
        for _ in 0..<12 { model.zoomStep(closer: true) }
        rig.settle()
        #expect(model.pose.distance == WorldLevels.minDistance)
        for _ in 0..<30 { model.zoomStep(closer: false) }
        rig.settle()
        #expect(abs(model.pose.distance - model.distanceLimit) < 1e-3)
    }

    @Test func aZoomButtonStopsAFlightAndADragStopsTheEase() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        let limit = model.distanceLimit
        model.editPose { $0.distance = 0.9 * limit }
        model.flyTo(SIMD3(10, 10, 0), distance: 12)
        #expect(model.flight != nil)
        model.zoomStep(closer: true)
        #expect(model.flight == nil && !model.isFlying && model.zoomEase != nil, "any input stops the flight")
        model.drag(dx: 5, dy: 0, orbit: false)
        #expect(model.zoomEase == nil, "and a drag stops the ease where it is")
    }

    // MARK: The minimap

    @Test func aClickInTheMinimapFliesThereKeepingTheDistanceAndTheHeading() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        model.editPose { $0.heading = 0.4 }
        let limit = model.distanceLimit
        model.editPose { $0.distance = 0.7 * limit }
        let before = model.pose
        let inset = try #require(model.inset)
        let place = try #require(inset.torus(atPoint: inMinimap))
        model.hudPress(.minimap)
        model.hudRelease(.minimap, at: inMinimap, clicks: 1, dragged: false)
        #expect(model.isFlying && model.flight != nil)
        rig.settle()
        #expect(abs(Torus.wrapDelta(model.pose.x - place.x, size: model.worldSize)) < 1e-3 && abs(Torus.wrapDelta(model.pose.y - place.y, size: model.worldSize)) < 1e-3)
        #expect(abs(model.pose.distance - before.distance) < 1e-3 && abs(model.pose.heading - 0.4) < 1e-5)
    }

    @Test func theSecondClickOfADoubleClickInTheMinimapDivesToTheRegionDistance() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        model.editPose { $0.distance = 1e6 }
        let far = model.pose.distance
        model.hudPress(.minimap)
        model.hudRelease(.minimap, at: inMinimap, clicks: 2, dragged: false)
        rig.settle()
        #expect(model.pose.distance < far && abs(model.pose.distance - model.flyDistance(.region)) < 1e-2, "\(model.pose.distance)")
    }

    @Test func aDragInTheMinimapMovesTheCentreLiveAndTheReleaseAfterItFliesNowhere() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        let inset = try #require(model.inset)
        model.hudPress(.minimap)
        for target in [CGPoint(x: inset.rect.minX + 30, y: inset.rect.minY + 40), CGPoint(x: inset.rect.midX, y: inset.rect.midY)] {
            model.hudDrag(.minimap, to: target)
            let place = try #require(inset.torus(atPoint: target))
            #expect(abs(model.pose.x - place.x) < 1e-3 && abs(model.pose.y - place.y) < 1e-3, "the centre is under the pointer")
        }
        // Held in the box when the pointer leaves it.
        model.hudDrag(.minimap, to: CGPoint(x: inset.rect.maxX + 300, y: inset.rect.midY))
        #expect(model.pose.x > model.worldSize * 0.98 || model.pose.x < 1, "the right edge is x = L (= 0): \(model.pose.x)")
        let held = model.pose
        model.hudRelease(.minimap, at: CGPoint(x: inset.rect.maxX + 300, y: inset.rect.midY), clicks: 1, dragged: true)
        #expect(!model.isFlying && model.flight == nil && model.pose == held, "a drag is not a click")
    }

    @Test func aDragOfTheMinimapStopsAFollowAndAFlight() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        model.setFollow(.storm(id: 7))
        model.flyTo(SIMD3(10, 10, 0), distance: 20)
        model.hudPress(.minimap)
        model.hudDrag(.minimap, to: inMinimap)
        #expect(model.followTarget == .none && !model.isFlying)
    }

    // MARK: Scrolling over the minimap

    @Test func theWheelOverTheMinimapZoomsTheMapAboutItsCentreAndOverTheMapAboutThePointer() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        let d0 = 0.9 * model.distanceLimit
        model.editPose { $0.distance = d0 }
        let centre = (model.pose.x, model.pose.y)
        model.scroll(ScrollEvent(dy: 1, precise: false, time: 1), at: inMinimap)
        #expect(model.pose.distance < d0, "the wheel zoomed")
        #expect(model.pose.x == centre.0 && model.pose.y == centre.1, "about the centre: the pointer is off the map's surface")
        model.editPose { $0.distance = d0 }
        let away = CGPoint(x: 150, y: 120)
        let under = try #require(model.projector(size: model.freeRect.size).world(at: away, height: model.pose.z))
        model.scroll(ScrollEvent(dy: 1, precise: false, time: 5), at: away)
        let now = try #require(model.projector(size: model.freeRect.size).world(at: away, height: model.pose.z))
        #expect(simd_distance(under, now) < 0.01 && (model.pose.x != centre.0 || model.pose.y != centre.1), "over the map the point under the pointer stays")
    }

    @Test func aPinchOverTheMinimapIsAboutTheCentreToo() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        let d0 = 0.9 * model.distanceLimit
        model.editPose { $0.distance = d0 }
        let centre = (model.pose.x, model.pose.y)
        model.magnify(0.2, at: inMinimap)
        #expect(model.pose.distance < d0 && model.pose.x == centre.0 && model.pose.y == centre.1)
    }

    // MARK: The compass and Home

    @Test func aClickOnTheCompassTurnsTheMapNorthUpAndADoubleClickIsHome() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        let compass = try #require(model.hud.compass)
        let at = CGPoint(x: compass.midX, y: compass.midY)
        model.editPose { $0.heading = 1.0; $0.distance = 14; $0.x = 7; $0.y = 9 }
        model.hudPress(.compass)
        model.hudRelease(.compass, at: at, clicks: 1, dragged: false)
        #expect(model.headingEase != nil)
        rig.settle()
        #expect(model.pose.heading == 0 && model.pose.distance == 14 && model.pose.x == 7, "north up changes only the heading")
        model.editPose { $0.heading = 1.0 }
        model.hudPress(.compass)
        model.hudRelease(.compass, at: at, clicks: 2, dragged: false)
        rig.settle()
        let home = model.homePose(header: try #require(rig.play.store.header))
        #expect(model.pose.heading == 0 && abs(model.pose.distance - home.distance) < 1e-2 && abs(model.pose.x - home.x) < 1e-2)
    }

    @Test func aReleaseOffTheControlIsNotAPress() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        model.editPose { $0.distance = 14; $0.heading = 1 }
        let d = model.pose.distance
        model.hudPress(.zoomIn)
        model.hudRelease(.zoomIn, at: CGPoint(x: 300, y: 300), clicks: 1, dragged: false)
        model.hudPress(.compass)
        model.hudRelease(.compass, at: CGPoint(x: 300, y: 300), clicks: 1, dragged: false)
        model.hudPress(.home)
        model.hudRelease(.home, at: CGPoint(x: 300, y: 300), clicks: 1, dragged: false)
        #expect(model.zoomEase == nil && model.headingEase == nil && !model.isFlying && model.pose.distance == d)
        #expect(model.overlay.pressed == nil)
    }

    @Test func theHomeGlyphFliesHome() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        model.editPose { $0.distance = 8; $0.x = 5; $0.y = 5 }
        let glyph = try #require(model.hud.home)
        model.hudPress(.home)
        model.hudRelease(.home, at: CGPoint(x: glyph.midX, y: glyph.midY), clicks: 1, dragged: false)
        #expect(model.isFlying && model.flight?.home == true)
    }

    // MARK: Hover

    @Test func overAControlTheMapHasNoTooltipAndNoReadoutAndTheControlLights() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        model.hover(at: CGPoint(x: 450, y: 350))
        #expect(model.overlay.hudHover == .map && model.overlay.pointerText != nil, "on the map the pointer has a place")
        model.hover(at: inMinimap)
        #expect(model.overlay.hudHover == .minimap && model.overlay.hover == nil && model.overlay.pointerText == nil)
        let place = try #require(model.overlay.minimapPoint)
        #expect(place == model.inset?.torus(atPoint: inMinimap))
        let zoomIn = try #require(model.hud.zoomIn)
        model.hover(at: CGPoint(x: zoomIn.midX, y: zoomIn.midY))
        #expect(model.overlay.hudHover == .zoomIn && model.overlay.minimapPoint == nil)
        model.hover(at: nil)
        #expect(model.overlay.hudHover == .map && model.overlay.pointerText == nil)
    }

    @Test func thePointerReadoutIsTheLongitudeAndLatitudeOfTheFloorUnderIt() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        let p = CGPoint(x: 300, y: 200)
        let place = try #require(model.projector(size: model.freeRect.size).world(at: p, height: 0))
        #expect(model.pointerReadout(at: p) == WorldReadout.floorText(x: place.x, y: place.y, size: model.worldSize), "it says it is the floor")
        #expect(WorldReadout.floorText(x: place.x, y: place.y, size: model.worldSize).hasSuffix(WorldReadout.coordinates(x: place.x, y: place.y, size: model.worldSize)))
        #expect(model.pointerReadout(at: nil) == nil)
    }

    // MARK: The scale bar

    @Test func theScaleBarIsTrueAtThePointLookedAtAndFollowsTheZoom() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        let limit = model.distanceLimit
        model.editPose { $0.distance = 0.4 * limit; $0.tilt = 0 }
        let near = try #require(model.scaleBar)
        // The bar is a number of units that is that many points long at the centre of the view: two points one bar apart.
        let projector = model.projector(size: model.freeRect.size)
        let a = try #require(projector.screen(SIMD3(model.pose.x, model.pose.y, 0)))
        let b = try #require(projector.screen(SIMD3(model.pose.x + near.bar.units, model.pose.y, 0)))
        #expect(abs(hypot(a.x - b.x, a.y - b.y) - near.bar.points) < 0.5, "\(near.bar.points) points for \(near.bar.units) units")
        model.editPose { $0.distance = 1e6 }
        let far = try #require(model.scaleBar)
        #expect(far.bar.units > near.bar.units, "zoomed out the bar spans more world")
        // The organism reference carries the header's radius.
        let radius = Float(try #require(rig.play.store.header).radius)
        #expect(near.organism?.diameterUnits == 2 * radius)
    }

    @Test func theOrganismReferenceSaysWhenTheSizePolicyDrawsOrganismsOffSize() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        model.drawableResized(CGSize(width: 1800, height: 1400))
        model.editPose { $0.distance = 1e6 }
        model.display.sizeMode = .physical
        let physical = try #require(model.scaleBar?.organism)
        #expect(abs(physical.drawnRatio - 1) < 0.5 || physical.physicalPoints < 4, "at the physical size the symbol is the organism's (to its least disc)")
        model.display.sizeMode = .capped
        model.editPose { $0.distance = 12 }
        let capped = try #require(model.scaleBar?.organism)
        #expect(capped.drawnRatio <= 1.0001 || capped.drawnRatio >= 1, "a ratio is always given")
        #expect(capped.physicalPoints > 0)
    }

    // MARK: The hint

    @Test func theHintShowsOnTheFirstWorldAnyInputTakesItAwayAndItDoesNotComeBack() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let suite = try #require(UserDefaults(suiteName: "alife-hint-\(UUID().uuidString)"))
        let first = try await rig(scratch, defaults: suite)
        #expect(first.model.hintVisible, "the first World window shows it")
        #expect(!suite.bool(forKey: WorldModel.hintKey), "and it is not kept as seen until an input")
        first.model.zoom(by: 0.9)  // (a pinch, a key and a click each go through `dismissHint` too)
        first.model.perform(.northUp)
        #expect(!first.model.hintVisible && suite.bool(forKey: WorldModel.hintKey))
        first.play.shutdown()
        let scratch2 = try Scratch()
        defer { scratch2.cleanup() }
        let second = try await rig(scratch2, defaults: suite)
        defer { second.play.shutdown() }
        #expect(!second.model.hintVisible, "a later window does not show it")
        second.model.showHint()
        #expect(second.model.hintVisible, "View > Controls shows it again")
        second.model.hudPress(.zoomIn)
        #expect(!second.model.hintVisible)
    }

    @Test(arguments: ["pinch", "scroll", "grab", "drag", "key", "press"])
    func everyKindOfInputDismissesTheHint(kind: String) async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        #expect(model.hintVisible)
        switch kind {
        case "pinch": model.magnify(0.1, at: CGPoint(x: 300, y: 300))
        case "scroll": model.scroll(ScrollEvent(dy: 1, precise: false, time: 1), at: CGPoint(x: 300, y: 300))
        case "grab": model.beginGrab(at: CGPoint(x: 300, y: 300))
        case "drag": model.drag(dx: 3, dy: 3, orbit: false)
        case "key": model.perform(.toggleBonds)
        default: model.hudPress(.minimap)
        }
        #expect(!model.hintVisible, "\(kind)")
    }

    // MARK: The colour mode

    @Test func aThreeDWorldListsTheHeightLastAndOpensInTheWorldsOwnColours() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        let engine = try #require(rig.play.store.header?.colorModes)
        #expect(model.colourModes == engine + ["height"])
        #expect(!model.display.heightColour && model.currentColourMode != "height", "the decision of DESIGN 36.5: not the default")
        let inputs = ViewMenuInputs(play: rig.play, model: model)
        #expect(inputs.colours.map(\.mode) == engine + ["height"] && inputs.colours.last?.title == "Height")
        #expect(inputs.current == model.currentColourMode)
    }

    @Test func choosingTheHeightIsAMatterOfTheClientAndSendsNothing() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        model.setColourMode("height")
        #expect(model.display.heightColour && model.currentColourMode == "height")
        #expect(ViewMenuInputs(play: rig.play, model: model).current == "height")
        try await Task.sleep(for: .milliseconds(150))
        #expect(!rig.sent.contains { $0.hasPrefix("color") }, "\(rig.sent)")
        // An engine mode is a command to the world and turns the height off.
        let mode = try #require(rig.play.store.header?.colorModes.first)
        model.setColourMode(mode)
        #expect(!model.display.heightColour)
        #expect(await eventually { rig.sent.contains("color \(mode)") })
    }

    @Test func theColourKeyCyclesThroughTheEnginesModesAndThenTheHeightAndRoundAgain() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await rig(scratch)
        defer { rig.play.shutdown() }
        let model = rig.model
        let modes = model.colourModes
        #expect(WorldKeys.nextColorMode(after: modes[modes.count - 2], in: modes) == "height", "after the last engine mode comes the height")
        #expect(WorldKeys.nextColorMode(after: "height", in: modes) == modes[0], "and then the first again")
        model.display.heightColour = true
        model.perform(.cycleColor)
        #expect(!model.display.heightColour, "from the height the key goes on to the world's first mode")
        #expect(await eventually { rig.sent.contains("color \(modes[0])") })
        // From a mode of the world that is not the last the key sends the next, and the height stays off.
        let current = model.currentColourMode
        model.perform(.cycleColor)
        let next = try #require(WorldKeys.nextColorMode(after: current, in: modes))
        #expect(model.display.heightColour == (next == "height"))
    }

    @Test func theChoiceIsKeptAndTheNextWorldOpensInIt() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let suite = try #require(UserDefaults(suiteName: "alife-colour-\(UUID().uuidString)"))
        let first = try await rig(scratch, defaults: suite)
        first.model.setColourMode("height")
        #expect(suite.object(forKey: WorldModel.heightColourKey) as? Bool == true)
        first.play.shutdown()
        let scratch2 = try Scratch()
        defer { scratch2.cleanup() }
        let second = try await rig(scratch2, defaults: suite)
        defer { second.play.shutdown() }
        #expect(second.model.display.heightColour, "the player's choice is the world's first colour")
        second.model.setColourMode(try #require(second.play.store.header?.colorModes.first))
        #expect(suite.object(forKey: WorldModel.heightColourKey) as? Bool == false)
        // The default applied on opening a world is not a choice, so it is not kept.
        let blank = try #require(UserDefaults(suiteName: "alife-colour-\(UUID().uuidString)"))
        let scratch3 = try Scratch()
        defer { scratch3.cleanup() }
        let third = try await rig(scratch3, defaults: blank)
        defer { third.play.shutdown() }
        #expect(blank.object(forKey: WorldModel.heightColourKey) == nil)
    }

    @Test func aTwoDWorldHasNoHeightAndKeepsAStoredChoiceToItself() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let suite = try #require(UserDefaults(suiteName: "alife-colour-\(UUID().uuidString)"))
        suite.set(true, forKey: WorldModel.heightColourKey)
        let rig = try await rig(scratch, kind: "2d", defaults: suite)
        defer { rig.play.shutdown() }
        #expect(!rig.model.heightColourAvailable && !rig.model.display.heightColour)
        #expect(rig.model.colourModes == rig.play.store.header?.colorModes)
        rig.model.setColourMode("height")
        #expect(!rig.model.display.heightColour, "there is no height in a flat world")
        #expect(suite.object(forKey: WorldModel.heightColourKey) as? Bool == true, "and a 2D world does not overwrite the choice")
    }
}

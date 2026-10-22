import AlifeCore
import AppKit
import Foundation
import MetalKit
import QuartzCore

/// A counter the overlay `Canvas` watches. Held apart from `WorldModel` so that a drag of the camera, which changes it sixty
/// times a second, re-evaluates the overlay and not the whole window.
@MainActor
final class OverlayState: ObservableObject {
    @Published private(set) var revision = 0
    /// What the pointer is over, with the words for its tooltip (nil over nothing, or off the view).
    private(set) var hover: HoverInfo?
    /// The cast row the pointer is over: its storm or territory is highlighted on the map.
    private(set) var castHover: WorldSelection?

    func invalidate() { revision &+= 1 }

    func setHover(_ info: HoverInfo?) {
        guard info != hover else { return }
        hover = info
        invalidate()
    }

    func setCastHover(_ selection: WorldSelection?) {
        guard selection != castHover else { return }
        castHover = selection
        invalidate()
    }
}

/// A hover: where the pointer is (in the free rectangle's points), what is under it and the line to show.
struct HoverInfo: Equatable {
    var point: CGPoint
    var pick: WorldPick
    var text: String
}

/// What the World window does that is not the world's: the display options, the camera and its controls, the level, the
/// fade between maps, the selection, the free rectangle the map is drawn into, and the hand-over of what the world sends to
/// the renderer (DESIGN §33.4.1). The world's own commands go to the `PlayController`.
///
/// The camera lives here, not in the renderer: the renderer is given it before each draw. Everything is placed at the minimum
/// image of the camera target, which is kept in `[0, L)^2`.
@MainActor
final class WorldModel: ObservableObject {
    let play: PlayController
    let overlay = OverlayState()
    @Published var display = WorldDisplay() {
        didSet { if display != oldValue { pushDisplay() } }
    }
    /// The launcher sheet.
    @Published var launcherOpen = false
    /// The last thing worth telling the user.
    @Published private(set) var note: String?
    /// Why nothing can be drawn (no Metal device, shaders that do not compile, a draw that failed); nil while it works.
    @Published private(set) var drawingFailure: String?
    /// The level of the view (`WorldLevels`): what the View menu's agent toggle refers to.
    @Published private(set) var level = WorldLevel.planet

    /// The part of the window the map is drawn into (the window less the dock), in the window's coordinates: the camera's
    /// aspect and every pixel rule are its.
    private(set) var freeRect = CGRect.zero
    private(set) var camera = Camera.planet(size: 1)
    /// While the camera is where the window put it, a resize of the free rectangle fits the world to it again.
    private(set) var cameraIsHome = true
    /// What the player picked: the inspector shows it, the overlay marks it (§33.4.8).
    @Published var selection = WorldSelection.none
    /// What the camera follows (a manual drag cancels it).
    @Published var followTarget = FollowTarget.none
    /// The toasts on screen, oldest first.
    @Published var toasts: [ToastQueue.Toast] = []
    /// Which families of event the Chronicle tab lists.
    @Published var chronicleFilter = ChronicleFilter.all
    /// Where the follow passed when a followed organism died, by the old uid (the inspector's Family row).
    var successors: [Int64: Int64] = [:]

    var followState = FollowState()
    var glide = CameraGlide()
    /// A fly-to in progress: the place (when nothing is followed) and the distance to end at.
    var flyGoal: SIMD3<Float>?
    var flyDistance: Float?
    var isFlying = false
    var toastQueue = ToastQueue()
    var lastAdvance: Double?
    var lastHoverPick: Double = -1
    /// The listed index the selected organism had in the last frame (where to start looking in the next).
    var selectionHint = 0

    /// The organism selected, if one is.
    var selectedUID: Int64? { selection.organismUID }

    weak var renderer: Renderer?
    weak var view: WorldMTKView?
    private var fade = MapFade()
    private var lastMapSerial = -1
    private var configuredFor: StreamHeader?
    private var interactionFor: StreamHeader?
    /// Seconds from a monotonic origin; tests inject theirs.
    var now: () -> Double = { CACurrentMediaTime() }

    init(play: PlayController) {
        self.play = play
        play.onDelivery = { [weak self] changes in self?.deliver(changes) }
        play.onEvents = { [weak self] events, backlog in self?.handle(events: events, backlog: backlog) }
    }

    // MARK: Free rectangle

    func failDrawing(_ message: String?) { drawingFailure = message }

    func tell(_ message: String?) { note = message }

    func setFreeRect(_ rect: CGRect) {
        guard rect != freeRect else { return }
        freeRect = rect
    }

    /// The view's width over its height (1 before there is one).
    var aspect: Float { DockLayout.aspect(of: freeRect) }

    var worldSize: Float { Float(play.store.header?.size ?? 1) }

    var height: Float { Float(play.store.header?.height ?? 0) }

    // MARK: What the world sent

    /// Hands what the world sent to the renderer: `changes` is what the delivery changed. Also the first call of a freshly
    /// created view (`attach`), with everything.
    func deliver(_ changes: WorldStore.Changes) {
        let store = play.store
        // A new world forgets what the old one had selected, followed and announced, drawn or not.
        if changes.contains(.header) || interactionFor != store.header {
            interactionFor = store.header
            resetInteraction()
        }
        guard let renderer else { return }
        if changes.contains(.header) || configuredFor != store.header {
            configuredFor = store.header
            renderer.configureWorld(header: store.header)
            fade.reset()
            lastMapSerial = -1
            if store.header != nil { goHome() }
        }
        if changes.contains(.frame) { renderer.showWorld(frame: store.frame) }
        if changes.contains(.map), let map = store.map, store.mapSerial != lastMapSerial {
            lastMapSerial = store.mapSerial
            let start = fade.deliver(at: now())
            renderer.deliverWorld(map: map, serial: store.mapSerial, start: start)
            renderer.worldDraw.mapBlend = fade.blend(at: now())
            startAnimating()
        }
        if changes.contains(.territory) { renderer.showWorld(territory: store.territory, serial: store.territorySerial) }
        if changes.contains(.storms) || changes.contains(.environment) || changes.contains(.header)
            || changes.contains(.territory) || (changes.contains(.frame) && !selection.isNone) {
            overlay.invalidate()
        }
        if changes.contains(.frame) && !followState.target.isNone { startAnimating() }
        view?.needsDisplay = true
    }

    /// A newly created view takes over: it is given everything the store holds.
    func attach(renderer: Renderer, view: WorldMTKView) {
        self.renderer = renderer
        self.view = view
        renderer.onResize = { [weak self] size in self?.drawableResized(size) }
        configuredFor = nil
        let store = play.store
        var all: WorldStore.Changes = [.frame, .territory]
        if store.header != nil { all.formUnion(.header) }
        if store.map != nil { all.formUnion(.map) }
        lastMapSerial = -1
        fade.reset()
        deliver(all)
        pushDisplay()
        pushCamera()
    }

    private func pushDisplay() {
        renderer?.worldDraw.display = display
        view?.needsDisplay = true
    }

    // MARK: The map fade

    var isAnimating: Bool { fade.isAnimating(at: now()) || cameraMotionPending }

    func startAnimating() { view?.startAnimating() }

    /// One tick of the display link: the fade's blend for this moment. False when nothing animates any more.
    func advance() -> Bool {
        let t = now()
        let dt = lastAdvance.map { Float(t - $0) } ?? 1.0 / 60
        lastAdvance = t
        renderer?.worldDraw.mapBlend = fade.blend(at: t)
        let moving = advanceCamera(dt: dt)
        if !moving { lastAdvance = nil }
        view?.needsDisplay = true
        return fade.isAnimating(at: t) || moving
    }

    // MARK: Camera

    func goHome() {
        guard let header = play.store.header else { return }
        camera = Camera.world(for: header, aspect: aspect)
        cameraIsHome = true
        pushCamera()
    }

    /// `R`, the View menu: the starting pose; nothing follows any more (the selection stays).
    func resetCamera() {
        cancelCameraMotion()
        goHome()
    }

    /// The drawable changed size (the free rectangle, or the window): while the camera is home, fit the world to the new shape.
    func drawableResized(_ size: CGSize) {
        guard cameraIsHome, size.height >= 1, let header = play.store.header else { return }
        camera = Camera.world(for: header, aspect: Float(size.width / size.height))
        pushCamera()
    }

    func edit(_ change: (inout Camera) -> Void) {
        change(&camera)
        let size = worldSize
        camera.wrapTarget(size: size)
        camera.target.z = min(max(camera.target.z, 0), max(height, 0))
        let limit = WorldLevels.maxDistance(size: size, fovY: camera.fovY, aspect: aspect)
        camera.distance = min(max(camera.distance, WorldLevels.minDistance), max(limit, WorldLevels.minDistance))
        cameraIsHome = false
        pushCamera()
    }

    /// A drag of `dx`, `dy` points: a pan (the map follows the pointer) or, with `orbit`, a turn about the target.
    func drag(dx: Float, dy: Float, orbit: Bool) {
        cancelCameraMotion()  // a manual drag cancels the follow (the selection stays)
        let viewportHeight = Float(freeRect.height >= 1 ? freeRect.height : (view?.bounds.height ?? 1))
        edit { $0.drag(dx: dx, dy: dy, mode: .orbit, pan: !orbit, viewportHeight: viewportHeight) }
    }

    func scroll(delta: Float, precise: Bool) { zoom(by: ViewerInput.dollyFactor(scroll: delta, precise: precise)) }

    func magnify(_ magnification: Float) { zoom(by: ViewerInput.dollyFactor(magnification: magnification)) }

    func zoom(by factor: Float) {
        flyDistance = nil  // the user's zoom wins over the distance a fly was heading for
        edit { $0.dolly(by: factor) }
    }

    func pushCamera() {
        let newLevel = WorldLevels.level(distance: camera.distance, fovY: camera.fovY, aspect: aspect, size: worldSize)
        if newLevel != level { level = newLevel }
        overlay.invalidate()
        guard let renderer else { return }
        renderer.camera = camera
        renderer.worldDraw.level = newLevel
        view?.needsDisplay = true
    }

    /// The projection the overlay and the picker use: the camera on the free rectangle (`size` in points).
    func projector(size: CGSize) -> WorldProjector {
        WorldProjector(camera: camera, size: worldSize, view: size)
    }

    // MARK: Selection

    /// The lowest-level change of the selection: no command is sent (`selectOrganism` and the rest do that). The overlay is
    /// redrawn once, and only on a change.
    func setSelection(_ new: WorldSelection) {
        guard selection != new else { return }
        selection = new
        selectionHint = 0
        overlay.invalidate()
    }
}

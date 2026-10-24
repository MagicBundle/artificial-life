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
    /// While a power is being aimed: where the pointer is on the map, in points of the free rectangle, and the place on the floor
    /// under it.
    private(set) var aimPoint: CGPoint?
    private(set) var aimWorld: SIMD2<Float>?

    func invalidate() { revision &+= 1 }

    func setAim(point: CGPoint?, world: SIMD2<Float>?) {
        guard point != aimPoint || world != aimWorld else { return }
        aimPoint = point
        aimWorld = world
        invalidate()
    }

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

/// The tabs of the control center (DESIGN §34.7).
enum DockTab: String, CaseIterable, Identifiable {
    case cast = "Cast"
    case chronicle = "Chronicle"
    case report = "Report"
    case powers = "Powers"
    var id: String { rawValue }
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
    /// The size of the world on disk (DESIGN §34.10), measured while a world is in play.
    let storage = StorageMonitor()
    @Published var display = WorldDisplay() {
        didSet {
            if display != oldValue {
                if display.exposure != oldValue.exposure { defaults.set(Double(display.exposure), forKey: Self.exposureKey) }
                pushDisplay()
                if display.map != oldValue.map { pushCamera() }  // the planet level exists only while a map is shown
                // The size rule counts rows only where it caps them, and not for an organism layer that is off.
                if display.sizeMode != oldValue.sizeMode || display.agents != oldValue.agents { refreshSizePolicy(force: true) }
            }
        }
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
    /// The open tab of the control center.
    @Published var tab = DockTab.cast {
        didSet { if tab != oldValue { overlay.invalidate() } }
    }
    /// The exchange with the world about a power (aim, preview, confirm).
    @Published var powers = PowersModel() {
        didSet { if powers != oldValue { overlay.invalidate() } }
    }
    /// The kin groups the player watches.
    @Published var watch = WatchSet()
    /// The line at the bottom of the map.
    @Published var tickerLine: Ticker.Line?
    /// The chapter card is folded to a one-line bar (`Esc`, or a power being aimed folds it by itself).
    @Published var cardCollapsed = false
    /// The chapter whose notice is on screen: the pause is off, so the world runs on and the card is a notice that goes by itself.
    @Published var chapterNotice: Int?
    /// The pauses the player armed or disarmed; sent back to every world after its header.
    @Published var autopause: AutopausePrefs
    /// Developer filters of the Chronicle tab (the raw kin events).
    @Published var debugFilters = false
    /// Where the follow passed when a followed organism died, by the old uid (the inspector's Family row).
    var successors: [Int64: Int64] = [:]

    var ticker = Ticker()
    var attention = AttentionTracker()
    var lastCardChapter = 0
    /// When the notice of a chapter whose pause is off goes (seconds of `now`).
    var noticeExpires: Double?
    /// The newest chapter whose notice was shown (a notice is shown once, not at every clock of the chapter).
    var noticeShownFor = 0
    var windowObservers: [any NSObjectProtocol] = []
    let defaults: UserDefaults
    /// Whether the World window is the one in front; the dock icon is only called when it is not. Tests inject theirs.
    var windowIsKey: () -> Bool = { true }
    var requestAttention: () -> Void = {}
    var setBadge: (String?) -> Void = { _ in }
    static let autopauseKey = "world.autopause"
    /// The map's brightness (the View menu's Brightness), kept between runs.
    static let exposureKey = "world.exposure"

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
    /// What the View menu shows, apart from the objects that change with every clock (see `DistinctState`).
    private(set) lazy var viewMenu = DistinctState<ViewMenuInputs>(
        initial: ViewMenuInputs(play: play, model: self), sources: [play.objectWillChange, objectWillChange]
    ) { [weak self] in
        guard let self else { return nil }
        return ViewMenuInputs(play: play, model: self)
    }
    /// The colour mode of the frame the View menu last read (a frame does not publish, so a change of mode is looked for in each).
    private var menuColourIndex = -1
    private var fade = MapFade()
    private var lastMapSerial = -1
    private var configuredFor: StreamHeader?
    private var interactionFor: StreamHeader?
    // The size policy (DESIGN 36.7): the rows in view are counted at most ten times a second, and the radius they give is
    // smoothed with the display link.
    /// The drawable's size in pixels (the free rectangle at the display's scale); zero before the first draw.
    private(set) var drawableSize = CGSize.zero
    /// The drawable's pixels per point of the free rectangle.
    private(set) var pixelsPerPoint: Float = 2
    /// The coverage radius the newest count gave, and the smoothed one the renderer is given (infinite where no cap applies).
    private(set) var coverTarget = Float.infinity
    private(set) var coverRadius = Float.infinity
    private var lastSizeCount: Double = -1
    /// A count was asked for inside the period: the display link makes it once the period is over.
    private(set) var sizeCountPending = false
    /// Seconds from a monotonic origin; tests inject theirs.
    var now: () -> Double = { CACurrentMediaTime() }

    init(play: PlayController, defaults: UserDefaults = .standard) {
        self.play = play
        self.defaults = defaults
        autopause = AutopausePrefs(json: defaults.string(forKey: Self.autopauseKey))
        if let stored = defaults.object(forKey: Self.exposureKey) as? Double, stored.isFinite, stored > 0 {
            display.exposure = Float(min(max(stored, 0.5), 2))
        }
        play.onDelivery = { [weak self] changes in self?.deliver(changes) }
        play.onEvents = { [weak self] events, backlog in self?.handle(events: events, backlog: backlog) }
        play.onAlerts = { [weak self] changes, backlog in self?.handle(alerts: changes, backlog: backlog) }
        play.onReports = { [weak self] reports in self?.handle(reports: reports) }
        play.onPlans = { [weak self] plans in self?.handle(plans: plans) }
        play.onReset = { [weak self] in self?.resetInteraction() }
        storage.isLive = { [weak play] in play?.phase == .live }
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

    /// Whether a weather map is in view: the planet level (no organisms, the map is the picture) exists only then.
    var mapShown: Bool { display.map && play.store.map != nil }

    /// The farthest the camera may be in the window as it is now: the rule of `WorldLevels.maxDistance`, and never nearer
    /// than the starting pose (which a wide or tall free rectangle puts farther out).
    var distanceLimit: Float {
        let home = play.store.header.map { Camera.world(for: $0, aspect: aspect).distance } ?? 0
        return WorldLevels.maxDistance(size: worldSize, fovY: camera.fovY, aspect: aspect, home: home)
    }
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
        if changes.contains(.header) {
            resendAutopause()
            startStorage()
        }
        if changes.contains(.clock) { handleClock() }
        if changes.contains(.frame), let index = store.frame?.colorIndex, index != menuColourIndex {
            menuColourIndex = index
            viewMenu.refresh()
        }
        if changes.contains(.header) || changes.contains(.frame) { refreshSizePolicy(force: changes.contains(.header)) }
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
            let first = lastMapSerial < 0
            lastMapSerial = store.mapSerial
            if first { pushCamera() }  // the first map makes the planet level possible
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
        // The size policy as it stands (a view made later than the model has seen no change of it: A0 review, finding 3).
        renderer.worldDraw.pixelsPerPoint = pixelsPerPoint
        pushCoverRadius()
    }

    private func pushDisplay() {
        renderer?.worldDraw.display = display
        view?.needsDisplay = true
    }

    // MARK: The size policy

    /// The rows in view and the radius they give (DESIGN §36.7): `A` is the drawable's area, `N` the listed rows whose centre is
    /// in view at their minimum image. Counted at most `OrganismSizePolicy.countPeriod` apart (a request inside the period is made
    /// by the display link when it is over), unless `force`. The radius the renderer is given follows the count with a one-pole
    /// filter (`settleCoverRadius`), except when it was or becomes unbounded.
    func refreshSizePolicy(force: Bool = false) {
        let t = now()
        if !force, lastSizeCount >= 0, t >= lastSizeCount, t - lastSizeCount < OrganismSizePolicy.countPeriod {
            if !sizeCountPending {
                sizeCountPending = true
                startAnimating()
            }
            return
        }
        lastSizeCount = t
        sizeCountPending = false
        var target = Float.infinity
        let area = Float(drawableSize.width * drawableSize.height)
        if display.sizeMode == .capped, display.agents, area >= 1, let frame = play.store.frame, frame.n > 0 {
            let rows = OrganismSizePolicy.rowsInView(frame: frame, camera: camera, aspect: Float(drawableSize.width / drawableSize.height),
                                                     size: worldSize)
            target = OrganismSizePolicy.coverRadius(area: area, rows: rows)
        }
        coverTarget = target
        if !coverRadius.isFinite || !target.isFinite {
            coverRadius = target
            pushCoverRadius()
        } else if coverRadius != target {
            startAnimating()
        }
    }

    /// One step of the smoothing of the cover radius towards the last count. True while it has not arrived.
    func settleCoverRadius(dt: Float) -> Bool {
        guard coverRadius != coverTarget else { return false }
        var next = OrganismSizePolicy.smoothed(coverRadius, toward: coverTarget, dt: dt)
        if !next.isFinite || abs(next - coverTarget) < 0.01 * max(coverTarget, 1) { next = coverTarget }
        coverRadius = next
        pushCoverRadius()
        return coverRadius != coverTarget
    }

    private func pushCoverRadius() {
        renderer?.worldDraw.coverRadius = coverRadius
        view?.needsDisplay = true
    }

    /// The most an organism is drawn with now, in points of the free rectangle (infinite at the physical size): the picker's radius.
    var drawnRadiusCap: Float {
        guard display.sizeMode == .capped, let header = play.store.header, drawableSize.height >= 1 else { return .infinity }
        let limits = OrganismSizePolicy.limits(
            mode: .capped, cover: coverRadius, distance: camera.distance, size: worldSize, agentRadius: Float(header.radius),
            viewHeight: Float(drawableSize.height), fovY: camera.fovY, pixelsPerPoint: pixelsPerPoint)
        return max(limits.cap, limits.minimum) / pixelsPerPoint
    }

    // MARK: The map fade

    var isAnimating: Bool { fade.isAnimating(at: now()) || cameraMotionPending || sizeCountPending || coverRadius != coverTarget }

    func startAnimating() { view?.startAnimating() }

    /// One tick of the display link: the fade's blend for this moment. False when nothing animates any more.
    func advance() -> Bool {
        let t = now()
        let dt = lastAdvance.map { Float(t - $0) } ?? 1.0 / 60
        lastAdvance = t
        renderer?.worldDraw.mapBlend = fade.blend(at: t)
        let moving = advanceCamera(dt: dt)
        if sizeCountPending { refreshSizePolicy() }
        let settling = settleCoverRadius(dt: dt)
        if !moving && !settling { lastAdvance = nil }
        view?.needsDisplay = true
        return fade.isAnimating(at: t) || moving || settling || sizeCountPending
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
        guard size.height >= 1 else { return }
        drawableSize = size
        pixelsPerPoint = freeRect.height >= 1 ? Float(size.height / freeRect.height) : 2
        renderer?.worldDraw.pixelsPerPoint = pixelsPerPoint
        if cameraIsHome, let header = play.store.header {
            camera = Camera.world(for: header, aspect: Float(size.width / size.height))
            pushCamera()
        }
        refreshSizePolicy(force: true)  // (after the refit: the count is of the view that is drawn)
    }

    func edit(_ change: (inout Camera) -> Void) {
        change(&camera)
        let size = worldSize
        camera.wrapTarget(size: size)
        camera.target.z = min(max(camera.target.z, 0), max(height, 0))
        camera.distance = min(max(camera.distance, WorldLevels.minDistance), max(distanceLimit, WorldLevels.minDistance))
        camera.keepFloorInReach(size: size, aspect: aspect)  // no edge of the finite tiled map in view
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
        let newLevel = WorldLevels.level(
            distance: camera.distance, fovY: camera.fovY, aspect: aspect, size: worldSize, mapShown: mapShown)
        if newLevel != level { level = newLevel }
        overlay.invalidate()
        refreshSizePolicy()
        guard let renderer else { return }
        renderer.camera = camera
        renderer.worldDraw.level = newLevel
        view?.needsDisplay = true
    }

    /// The projection the overlay and the picker use: the camera on the free rectangle (`size` in points).
    func projector(size: CGSize) -> WorldProjector {
        WorldProjector(camera: camera, size: worldSize, view: size, mapShown: mapShown)
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

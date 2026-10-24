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
    /// The control the pointer is over, and the one a pressed button went down on (the overlay lights them), the world point under the
    /// pointer over the minimap, and the readout of the point under it on the map ("270 E  28 N").
    private(set) var hudHover = HUDTarget.map
    private(set) var pressed: HUDTarget?
    private(set) var minimapPoint: SIMD2<Float>?
    private(set) var pointerText: String?

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

    func setHUDHover(_ target: HUDTarget, place: SIMD2<Float>?) {
        guard target != hudHover || place != minimapPoint else { return }
        hudHover = target
        minimapPoint = place
        invalidate()
    }

    func setPressed(_ target: HUDTarget?) {
        guard target != pressed else { return }
        pressed = target
        invalidate()
    }

    func setPointerText(_ text: String?) {
        guard text != pointerText else { return }
        pointerText = text
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
/// The view lives here, not in the renderer: the renderer is given its camera before each draw. Everything is placed at the minimum
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
                if display.scrollZooms != oldValue.scrollZooms { defaults.set(display.scrollZooms, forKey: Self.scrollZoomsKey) }
                if display.minimap != oldValue.minimap { defaults.set(display.minimap, forKey: Self.minimapKey) }
                if display.globe != oldValue.globe {
                    defaults.set(display.globe, forKey: Self.globeKey)
                    setGlobe(display.globe)
                }
                if display.heightColour != oldValue.heightColour, !applyingColourDefault {
                    defaults.set(display.heightColour, forKey: Self.heightColourKey)
                }
                pushDisplay()
                if display.minimap != oldValue.minimap || display.map != oldValue.map { pushInset() }
                overlay.invalidate()  // (the legend, the minimap's points and the scale bar follow the layers)
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
    /// The one-line hint of the controls is on screen (the first World window, and View > Controls…); any input takes it away.
    @Published var hintVisible = false

    /// The part of the window the map is drawn into (the window less the dock), in the window's coordinates: the camera's
    /// aspect and every pixel rule are its.
    private(set) var freeRect = CGRect.zero
    /// Where the window looks (DESIGN §36.5): the centre, the heading, the tilt and the distance. The camera is derived from it.
    var pose = WorldPose(x: 0.5, y: 0.5, distance: 1)
    /// The camera of `pose` on this world, for the renderer, the projector and the picker.
    var camera: Camera { pose.camera(size: worldSize, height: height) }
    /// While the camera is where the window put it, a resize of the free rectangle fits the world to it again.
    var cameraIsHome = true
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
    /// The trackpad's two-finger scroll zooms instead of panning (View > Scroll zooms), and the minimap is shown (`i`).
    static let scrollZoomsKey = "world.scrollZooms"
    static let minimapKey = "world.minimap"
    static let globeKey = "world.globe"
    /// The player's colour choice: true is the height colour, false the world's own; absent is the default of a 3D world.
    static let heightColourKey = "world.heightColour"

    var followState = FollowState()
    var glide = CameraGlide()
    /// The scroll gesture in progress (DESIGN §36.5), and the floor point the left button grabbed.
    var scrollLaw = ScrollLaw()
    var grabAnchor: WorldVantage.Anchor?
    /// The world opened and no frame has been seen: the Home view is still to face the populated side.
    var homeAwaitsFrame = false
    /// A fly-to on the van Wijk path (not while following: a followed organism is not a path), and a north-up ease.
    var flight: Flight?
    var headingEase: HeadingEase?
    /// A press of a zoom button easing the distance (0.25 s).
    var zoomEase: ZoomEase?
    /// An ease to a standard view, or of the tilt alone (the 3D button): the tilt, the heading, the height and the distance (`ViewEase`).
    var viewEase: ViewEase?
    /// The distance the player asked for, while the limit of the tilt holds the view nearer (`WorldPose.maxDistance`): the view goes
    /// back out to it as the tilt is relaxed, so tilting to the horizon and back does not leave the view zoomed in. Nil while no
    /// limit holds the view.
    var distanceWish: Float?
    /// The tilt the 3D button goes back to: the last angled view (Oblique at first).
    var lastTilt = ViewPreset.oblique.tilt
    /// The distance the player had before the Side view put the view at a distance of its own (`before`), and the distance it put it at
    /// (`at`): the next standard view takes `before` for its wish if the zoom is still at `at`.
    var sideReturn: (before: Float, at: Float)?
    /// While the compass is turned like a dial: the angle of the pointer about its centre and the heading when it went down.
    var compassTurn: (angle: Float?, heading: Float)?
    /// While the tilt handle is dragged: how far the pointer is from the thumb's centre (so that the thumb does not jump).
    var tiltGrab: CGFloat = 0
    /// The colour default is being applied (not the player's choice, so not kept).
    var applyingColourDefault = false
    /// Whether the system asks for less motion: a flight is then a straight 0.25 s ease. Tests inject theirs.
    var reduceMotion: () -> Bool = { NSWorkspace.shared.accessibilityDisplayShouldReduceMotion }
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
        if let stored = defaults.object(forKey: Self.scrollZoomsKey) as? Bool { display.scrollZooms = stored }
        if let stored = defaults.object(forKey: Self.minimapKey) as? Bool { display.minimap = stored }
        if let stored = defaults.object(forKey: Self.globeKey) as? Bool {
            display.globe = stored
            globeEnabled = stored
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
        pushInset()
        refitToShape(aspect: aspect)  // (the drawable follows, and refits the same way; whichever comes second finds nothing to do)
    }

    /// The window changed shape: the view is fitted to the new aspect. At Home the world is shown whole in the new shape; elsewhere the
    /// pose is clamped again (the distance to the new footprint's limit, then the tilt), so that the picture is never wider than the
    /// one-world patch and the seam copies are planned for the shape that is drawn. Before the world's header there is no view.
    private func refitToShape(aspect: Float) {
        guard play.store.header != nil, aspect.isFinite, aspect > 0 else { return }
        distanceWish = nil
        sideReturn = nil
        if cameraIsHome {
            pose.distance = globeEnabled ? Morph.globeDistance(L: worldSize, fovY: pose.fovY, aspect: aspect)
                : pose.distanceLimit(size: worldSize, aspect: aspect)
        } else {
            pose.clamp(aspect: aspect, size: worldSize, height: height, globe: globeEnabled)
        }
        pushCamera()
    }

    /// The view's width over its height (1 before there is one).
    var aspect: Float { DockLayout.aspect(of: freeRect) }

    /// Whether the globe overview exists for this world (DESIGN §36.10): the distance leads on to the globe, Home is the globe, the
    /// projector and the grab go through the morph. Off until the renderer draws it (PR B step 7); then the view is the flat map.
    var globeEnabled = false
    var worldSize: Float { Float(play.store.header?.size ?? 1) }

    /// Whether a weather map is in view: the planet level (no organisms, the map is the picture) exists only then.
    var mapShown: Bool { display.map && play.store.map != nil }

    /// The farthest the camera may be in the window as it is now, at its heading: the picture stays within one world of its centre
    /// (`WorldPose.maxDistance`; the Home view is at this distance).
    var distanceLimit: Float { pose.distanceLimit(size: worldSize, aspect: aspect, globe: globeEnabled) }
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
            applyColourDefault()
            if store.header != nil { showHintOnce() }
        }
        if changes.contains(.clock) { handleClock() }
        if changes.contains(.frame), let index = store.frame?.colorIndex, index != menuColourIndex {
            menuColourIndex = index
            viewMenu.refresh()
        }
        if changes.contains(.header) || changes.contains(.frame) { refreshSizePolicy(force: changes.contains(.header)) }
        // The Home view faces the side of the world the organisms are on, which the header does not say: the first frame of a world that
        // has not been touched since it opened turns it that way (once).
        if changes.contains(.header) {
            homeAwaitsFrame = store.frame == nil
            if store.header != nil { goHome() }  // (the renderer's turn below does it again when a view attaches later)
        }
        if homeAwaitsFrame, changes.contains(.frame), store.frame != nil {
            homeAwaitsFrame = false
            if cameraIsHome { goHome() }
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
            let first = lastMapSerial < 0
            lastMapSerial = store.mapSerial
            if first {  // the first map makes the planet level possible, and the minimap
                pushCamera()
                pushInset()
            }
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
        pushInset()
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
            let shape = Float(drawableSize.width / drawableSize.height)
            let vantage = self.vantage()
            let rows = vantage.isGlobe
                ? OrganismSizePolicy.rowsInView(frame: frame, vantage: vantage)
                : OrganismSizePolicy.rowsInView(frame: frame, camera: camera, aspect: shape, size: worldSize)
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

    /// The Home view of this world in the window as it is now: the flat overview, north up, facing the populated side.
    func homePose(header: StreamHeader) -> WorldPose {
        WorldPose.home(for: header, aspect: aspect, populatedX: play.store.frame.flatMap { WorldPose.populatedX(frame: $0, size: Float(header.size)) },
                       fovY: pose.fovY, globe: globeEnabled)
    }

    /// The Home view at once (a new world; a test). The `h` and `r` keys and the toolbar fly there (`home()`).
    func goHome() {
        guard let header = play.store.header else { return }
        pose = homePose(header: header)
        distanceWish = nil
        sideReturn = nil
        viewEase = nil
        cameraIsHome = true
        pushCamera()
    }

    /// The drawable changed size (the free rectangle, or the window): fit the view to the new shape (`refitToShape`).
    func drawableResized(_ size: CGSize) {
        guard size.height >= 1 else { return }
        drawableSize = size
        pixelsPerPoint = freeRect.height >= 1 ? Float(size.height / freeRect.height) : 2
        renderer?.worldDraw.pixelsPerPoint = pixelsPerPoint
        refitToShape(aspect: Float(size.width / size.height))
        refreshSizePolicy(force: true)  // (after the refit: the count is of the view that is drawn)
    }

    /// Edits the pose as a `Camera` (the camera's own vocabulary: the target, the distance, the yaw and the pitch) and applies the
    /// clamps. The gestures edit the `WorldPose` (`editPose`); this is for the glide and the tests.
    func edit(_ change: (inout Camera) -> Void) {
        var edited = camera
        change(&edited)
        editPose { $0.apply(edited) }
    }

    /// Edits the view and applies the clamps of every edit: the world's `x` and `y`, the height, the distance between a few radii and
    /// the footprint's limit, the tilt (`WorldPose.clamp`).
    ///
    /// A change that leaves the distance alone gets the distance the player asked for (`distanceWish`) if the limit of the tilt had
    /// held it nearer, and the limit holds it again if it still must: tilting the view to the horizon and back leaves the zoom where
    /// it was.
    /// The view of the pose as the globe's (or the flat map's, `globeEnabled` off): the type that grabs, zooms and pans through the morph.
    func vantage() -> WorldVantage {
        WorldVantage(pose: pose, size: worldSize, height: height, aspect: aspect, globe: globeEnabled && mapShown)
    }

    /// `editPose` for a gesture of the vantage (grab, zoom, pan): the regime solver of DESIGN §36.10 at the globe, the plane's exact one
    /// on the flat map.
    func editPose(through change: (inout WorldVantage) -> Void) {
        editPose { pose in
            var v = WorldVantage(pose: pose, size: self.worldSize, height: self.height, aspect: self.aspect,
                                 globe: self.globeEnabled && self.mapShown)
            change(&v)
            pose = v.pose
        }
    }

    func editPose(_ change: (inout WorldPose) -> Void) {
        let before = pose.distance
        let wish = distanceWish ?? before
        change(&pose)
        if pose.distance == before { pose.distance = wish }
        pose.followsHeight = Self.followsHeight(followState.target)
        let requested = pose.distance
        pose.clamp(aspect: aspect, size: worldSize, height: height, globe: globeEnabled)
        distanceWish = pose.distance < requested * (1 - 1e-5) ? requested : nil
        cameraIsHome = false
        pushCamera()
    }

    /// The size of the free rectangle in points (the view's, before the first layout).
    var viewSize: CGSize {
        if freeRect.width >= 1, freeRect.height >= 1 { return freeRect.size }
        return view?.bounds.size ?? CGSize(width: 1, height: 1)
    }

    /// A drag by `dx`, `dy` points with no anchor: a pan (the map follows the pointer: middle button) or, with `orbit`, a turn of the
    /// heading and the tilt about the target. The left button grabs instead (`beginGrab`, `grab(to:)`).
    func drag(dx: Float, dy: Float, orbit: Bool) {
        dismissHint()
        cancelCameraMotion()  // a manual drag cancels the follow (the selection stays)
        headingEase = nil
        let size = viewSize
        if orbit { editPose { $0.turn(dx: dx, dy: dy) } } else { editPose(through: { $0.pan(dx: dx, dy: dy, view: size) }) }
    }

    // MARK: The grab

    /// The left button went down at `point` (points of the free rectangle, origin at the top left): the floor point under it is what
    /// the drag keeps under the pointer.
    func beginGrab(at point: CGPoint) {
        dismissHint()
        grabAnchor = vantage().anchor(at: WorldPose.ndc(of: point, in: viewSize))
    }

    /// The pointer is at `point` with the left button down: the point grabbed is under it, exactly (`WorldPose.grab`).
    func grab(to point: CGPoint) {
        // (a press that took hold of nothing, in the sky or the haze of a tilted view, moves nothing and stops nothing: a follow goes on)
        guard let anchor = grabAnchor else { return }
        cancelCameraMotion()
        headingEase = nil
        let ndc = WorldPose.ndc(of: point, in: viewSize)
        editPose(through: { $0.grab(anchor: anchor, at: ndc) })
    }

    func endGrab() { grabAnchor = nil }

    // MARK: Zoom and scroll

    /// A scroll event at `point` (nil: the pointer is not known; the zoom is about the centre): the law of `ScrollLaw`.
    func scroll(_ event: ScrollEvent, at point: CGPoint?) {
        dismissHint()
        // Over the minimap the pointer is off the map's surface: a zoom is about the centre of the view.
        let onMap = point.map { pressTarget(at: $0) == .map } ?? true
        switch scrollLaw.handle(event, scrollZooms: display.scrollZooms) {
        case .none: return
        case .pan(let dx, let dy):
            cancelCameraMotion()
            headingEase = nil
            let size = viewSize
            editPose(through: { $0.pan(dx: dx, dy: dy, view: size) })
        case .zoom(let factor):
            zoom(by: factor, at: onMap ? point : nil)
        }
    }

    /// A pinch of `magnification` about `point`.
    func magnify(_ magnification: Float, at point: CGPoint? = nil) {
        dismissHint()
        let onMap = point.map { pressTarget(at: $0) == .map } ?? true
        zoom(by: ViewerInput.dollyFactor(magnification: magnification), at: onMap ? point : nil)
    }

    /// Multiplies the distance by `factor` about the pointer (the surface point under it stays under it): the view's own zoom. A
    /// flight that was under way stops where it is; a follow goes on (the user's zoom wins over the distance a fly was heading for).
    func zoom(by factor: Float, at point: CGPoint? = nil) {
        stopFlight()
        viewEase = nil
        distanceWish = nil
        let ndc = point.map { WorldPose.ndc(of: $0, in: viewSize) }
        editPose(through: { $0.zoom(by: factor, at: ndc) })
    }

    /// `n`, a click on the compass: turn the map north up, the heading easing to 0 over 0.3 s.
    func northUp() {
        guard pose.heading != 0, flight?.home != true else { return }  // (a flight to Home is turning the view north up already)
        viewEase = nil  // (an ease to a view would turn the heading back to its own every frame: any other input ends it where it is)
        headingEase = HeadingEase(from: pose.heading)
        startAnimating()
    }

    /// The globe overview switched on or off (View > Globe overview): the view is fitted to the new regime and flies Home.
    private func setGlobe(_ on: Bool) {
        guard globeEnabled != on else { return }
        globeEnabled = on
        guard play.store.header != nil else { return }
        pose.clamp(aspect: aspect, size: worldSize, height: height, globe: on)
        pushCamera()
        home()
    }

    func pushCamera() {
        let camera = self.camera
        let vantage = self.vantage()
        // (with the globe the derived level: organism under 0.12 L, planet from s = 0.5, else region)
        let newLevel = globeEnabled
            ? projector(size: freeRect.size).level
            : WorldLevels.level(distance: camera.distance, fovY: camera.fovY, aspect: aspect, size: worldSize, mapShown: mapShown)
        if newLevel != level { level = newLevel }
        overlay.invalidate()
        refreshSizePolicy()
        guard let renderer else { return }
        renderer.camera = camera
        renderer.worldDraw.level = newLevel
        // The globe path when the world has the globe (it draws the flat close-up too, as the same morph at s = 0).
        let footprint = pose.footprint(aspect: aspect, size: worldSize, height: height)
        renderer.worldDraw.vantage = vantage.isGlobe ? vantage : nil
        renderer.worldDraw.wideTiles = footprint.map { $0.x > 0.4 * worldSize } ?? true
        renderer.worldDraw.copies = SeamCopies.plan(y: pose.y, extentY: footprint?.y, size: worldSize, s: vantage.s)
        view?.needsDisplay = true
    }

    /// The projection the overlay and the picker use: the camera on the free rectangle (`size` in points).
    func projector(size: CGSize) -> WorldProjector {
        WorldProjector(pose: pose, size: worldSize, height: height, view: size, mapShown: mapShown, globe: globeEnabled)
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

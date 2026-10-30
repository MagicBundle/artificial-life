import AlifeCore
import AppKit
import CoreGraphics
import Foundation
import simd

/// What the World window does for a planet (DESIGN §46.8): the camera and its controls, picking and hover, the selection's camera parts, the
/// flights, and what the world sends to the sphere renderer. The decisions are the pure types of `AlifeCore` (`SphereCamera`, `SpherePicker`,
/// `SphereFollow`, `SphereScene`, `Mollweide`, `WorldHUD`); this applies them to the window. The window's other parts (the dock, the
/// chapters, the alerts, the powers) are the `WorldModel`'s and are the box world's as they are: this is created by the model, and the
/// model hands the camera work to it when the world in view is a planet.
@MainActor
final class SphereController {
    unowned let model: WorldModel
    var play: PlayController { model.play }
    var overlay: OverlayState { model.overlay }

    // MARK: State

    private(set) var camera = SphereCamera(pose: SpherePose(up: SIMD3(1, 0, 0), heading: SIMD3(0, 0, 1), distance: 100), planetRadius: 1, lidRadius: 2)
    private(set) var level = SphereLevel.planet
    weak var renderer: SphereRenderer?
    weak var view: SphereMTKView?
    private(set) var drawableSize = CGSize.zero
    private(set) var pixelsPerPoint: Float = 2
    /// While the camera is where the window put it, a resize of the view fits the planet to it again.
    private(set) var cameraIsHome = true
    /// The field pixel the player clicked (a cell is a place, not an entity): the shader marks it, the HUD describes it.
    private(set) var selectedCell: Int?
    /// What the pointer is over, with the words for its tooltip.
    private(set) var hover: (point: CGPoint, pick: SpherePick, text: String)?
    /// The floor point under the pointer on the planet, and its words ("12.3N 45.6E").
    private(set) var pointerText: String?
    /// The Mollweide image of the planet for the minimap, and its centre longitude (radians).
    private(set) var minimap: CGImage?
    private(set) var minimapSerial = 0

    private var scene: SphereScene?
    private var sceneDisplay = WorldDisplay()
    private var configuredFor: StreamHeader?
    private var follow = SphereFollow()
    private var flight: (path: SphereFlight, elapsed: Float, home: Bool)?
    private var flightGoalDistance: Float?
    /// Where the follow-style fly (a selection that moves, or an organism that died) goes while there is nothing to follow.
    var flyGoal: SIMD3<Float>?
    private var grabAnchor: SIMD3<Float>?
    private var grabMoved = false
    private var motion = GlobeMotion()
    private var scrollLaw = ScrollLaw()
    private var northEase: (from: SIMD3<Float>, elapsed: Float)?
    private var tiltEase: (from: Float, to: Float, elapsed: Float)?
    private var zoomEase: ZoomEase?
    private var compassTurn: (angle: Float?, heading: SIMD3<Float>)?
    private var tiltGrab: CGFloat = 0
    private var lastAdvance: Double?
    private var lastHoverPick: Double = -1
    private var lastDynamic = -1
    private var lastTilt = ViewPreset.oblique.tilt

    init(model: WorldModel) { self.model = model }

    var sphere: SphereHeader? { play.store.header?.sphere }
    var isActive: Bool { sphere != nil }
    private var planetRadius: Float { Float(sphere?.planetRadius ?? 1) }
    private var lidRadius: Float { Float(sphere?.lidRadius ?? 2) }
    /// The free rectangle the planet is drawn into, in points.
    var viewSize: CGSize {
        let free = model.freeRect
        if free.width >= 1, free.height >= 1 { return free.size }
        return view?.bounds.size ?? CGSize(width: 1, height: 1)
    }
    var aspect: Float { Float(viewSize.width / max(viewSize.height, 1)) }
    var now: () -> Double { model.now }

    // MARK: The world arrives

    /// The renderer and the view of a newly made `SphereMTKView`: they are given everything the store holds.
    func attach(renderer: SphereRenderer, view: SphereMTKView) {
        self.renderer = renderer
        self.view = view
        renderer.onResize = { [weak self] size in self?.drawableResized(size) }
        configuredFor = nil
        var all: WorldStore.Changes = [.frame, .territory, .map, .storms, .environment, .material, .soil]
        if play.store.header != nil { all.formUnion(.header) }
        deliver(all)
    }

    func detach() {
        renderer = nil
        view = nil
    }

    /// Hands what the world sent to the renderer: `changes` is what the delivery changed.
    func deliver(_ changes: WorldStore.Changes) {
        let store = play.store
        guard let sphere = store.header?.sphere else { return }
        if changes.contains(.header) || configuredFor != store.header {
            configuredFor = store.header
            follow.set(.none)
            flight = nil
            selectedCell = nil
            hover = nil
            scene = nil
            let home = SphereScene.home(store: store, aspect: aspect) ?? SpherePose(up: SIMD3(1, 0, 0), heading: SIMD3(0, 0, 1), distance: 100)
            camera = SphereCamera(pose: home, planetRadius: Float(sphere.planetRadius), lidRadius: Float(sphere.lidRadius))
            cameraIsHome = true
            renderer?.configure(sphere: sphere, bodyRadius: Float(store.header?.radius ?? 1), camera: camera)
            composeScene()
        }
        if changes.contains(.frame), store.frame != nil, cameraIsHome, store.frameSerial <= 2 {
            goHome()  // the first frames: face the side the organisms are on
        }
        let sceneChanged: WorldStore.Changes = [.map, .material, .soil, .territory, .storms, .environment]
        if !changes.isDisjoint(with: sceneChanged) { composeScene() }
        if changes.contains(.frame) || changes.contains(.header) { renderer?.show(frame: store.frame) }
        if changes.contains(.frame) || changes.contains(.header) || changes.contains(.identities) { updateDynamic() }
        if changes.contains(.frame), !follow.target.isNone { startAnimating() }
        pushDraw()
    }

    /// The colours, the lines and the minimap for the records the store holds and the layers the player has on.
    func composeScene() {
        let store = play.store
        guard store.header?.sphere != nil else { return }
        var options = SphereScene.Options()
        options.display = model.display
        if case .storm(let id) = model.selection { options.selectedStorm = id }
        if case .pool(let index) = model.selection { options.selectedPool = index }
        if case .kin(let id) = model.selection { options.selectedKin = id }
        guard let composed = SphereScene.compose(store: store, options: options) else { return }
        scene = composed
        sceneDisplay = model.display
        renderer?.setFloor(composed.floor)
        renderer?.setCloud(composed.cloud)
        renderer?.setTerritory(composed.territory)
        renderer?.setOverlay(composed.overlay)
        rebuildMinimap()
        overlay.invalidate()
        view?.needsDisplay = true
    }

    /// The display changed (the View menu): the layers are composed again, the rest is the draw's.
    func displayChanged() {
        composeScene()
        updateDynamic()
        pushDraw()
    }

    /// The selection changed: the highlights follow it.
    func selectionChanged() {
        composeScene()
        updateDynamic()
        pushDraw()
    }

    private func rebuildMinimap() {
        guard let sphere, let scene, model.display.minimap else {
            minimap = nil
            return
        }
        let centres = SphereCentres.shared.centres(nside: sphere.nsideField)
        let tint: Float = model.display.territory == .tint ? 0.3 : 0
        let colours = SphereColours.chart(floor: scene.floor, cloud: scene.cloud, territory: scene.territory,
                                          territoryNside: play.store.territory?.nside ?? sphere.nsideTerritory, fieldNside: sphere.nsideField, centres: centres,
                                          territoryAlpha: tint)
        let width = 360, height = 180
        let pixels = Mollweide.image(width: width, height: height, centre: 0, nside: sphere.nsideField, colours: colours)
        pixels.withUnsafeBytes { raw in
            guard let provider = CGDataProvider(data: Data(raw) as CFData) else { return }
            minimap = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4 * width,
                              space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue),
                              provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        }
        minimapSerial += 1
    }

    /// The bonds of the newest frame and the marker round the selected organism: the lines that change with each frame.
    func updateDynamic() {
        guard let renderer, let frame = play.store.frame, let sphere else { return }
        var lines: [SphereLine] = []
        if model.display.bonds, frame.nBonds > 0 { lines += SphereOverlay.bonds(frame, flow: model.display.bondFlow) }
        if let uid = model.selection.organismUID, let index = play.store.listedIndex(ofUID: uid), true {
            let p = frame.position(index)
            let eye = camera.eye
            let distance = simd_length(eye - p)
            let radius = max(Float(play.store.header?.radius ?? 1) * 2.4, distance * 0.012)
            lines += SphereOverlay.marker(at: p, eye: eye, radius: radius)
        }
        _ = sphere
        renderer.setDynamic(lines)
    }

    // MARK: Pushing the camera

    func pushDraw() {
        guard let renderer else { return }
        var draw = SphereDraw(camera: camera)
        draw.display = model.display
        draw.selectedPixel = selectedCell
        draw.cappedSize = model.display.sizeMode == .capped
        draw.pixelsPerPoint = pixelsPerPoint
        renderer.setDraw(draw)
        view?.needsDisplay = true
    }

    func pushCamera() {
        let altitudeLevel = SphereLevel.level(distance: camera.pose.distance, planetRadius: planetRadius, lidRadius: lidRadius)
        if altitudeLevel != level { level = altitudeLevel }
        overlay.invalidate()
        pushDraw()
        if !follow.target.isNone || model.selection.organismUID != nil { updateDynamic() }
    }

    /// The drawable changed size (the free rectangle, or the window): the planet is fitted to the new shape.
    func drawableResized(_ size: CGSize) {
        guard size.height >= 1 else { return }
        drawableSize = size
        let free = model.freeRect
        pixelsPerPoint = free.height >= 1 ? Float(size.height / free.height) : 2
        refit()
    }

    /// At Home the planet is shown whole in the new shape; elsewhere the distance is held inside its limits.
    func refit() {
        guard isActive else { return }
        if cameraIsHome, let home = SphereScene.home(store: play.store, aspect: aspect, fovY: camera.pose.fovY) {
            camera.pose = SpherePose(up: camera.pose.up, heading: camera.pose.heading, tilt: 0, distance: home.distance, fovY: home.fovY)
        } else {
            camera.clamp(aspect: aspect)
        }
        pushCamera()
    }

    /// The free rectangle changed (the window, the dock).
    func freeRectChanged() { refit() }

    /// Edits the camera and keeps it inside its limits; any edit leaves Home.
    func edit(keepsHome: Bool = false, _ change: (inout SphereCamera) -> Void) {
        var next = camera
        change(&next)
        next.clamp(aspect: aspect)
        camera = next
        if !keepsHome { cameraIsHome = false }
        pushCamera()
    }

    // MARK: Home and the views

    func homePose() -> SpherePose? { SphereScene.home(store: play.store, aspect: aspect, fovY: camera.pose.fovY) }

    /// The Home view at once (a new world).
    func goHome() {
        guard let pose = homePose() else { return }
        camera.pose = pose
        cameraIsHome = true
        flight = nil
        follow.set(.none)
        pushCamera()
    }

    /// `h`, `r`, the toolbar's Home: whatever follows or flies stops (the selection stays) and the camera flies to the Home view.
    func home() {
        cancelMotion(keepSelection: true)
        guard let pose = homePose() else { return }
        startFlight(to: pose, home: true)
    }

    /// `n`: the heading eases to north (0.3 s).
    func northUp() {
        let north = SphereFrame.basis(at: camera.pose.up).north
        guard simd_length(camera.pose.heading - north) > 1e-3 else { return }
        stopFlight()
        northEase = (camera.pose.heading, 0)
        startAnimating()
    }

    /// A standard view: its tilt, eased (0.5 s, a quarter second under reduced motion).
    func viewPreset(_ preset: ViewPreset) {
        stopFlight()
        setTiltEase(to: min(preset.tilt, SpherePose.maxTilt))
    }

    /// The 3D button: from straight down to the angle last used, and back.
    func toggle3D() {
        stopFlight()
        let angled = camera.pose.tilt > 0.05
        if angled { lastTilt = camera.pose.tilt }
        setTiltEase(to: angled ? 0 : min(max(lastTilt, ViewPreset.oblique.tilt * 0.5), ViewPreset.low.tilt))
    }

    private func setTiltEase(to goal: Float) {
        if camera.pose.tilt > 0.05 { lastTilt = camera.pose.tilt }
        if model.reduceMotion() {
            edit { $0.pose.tilt = goal }
        } else {
            tiltEase = (camera.pose.tilt, goal, 0)
            startAnimating()
        }
    }

    func setTilt(_ tilt: Float) {
        stopFlight()
        tiltEase = nil
        edit { $0.pose.tilt = min(max(tilt, 0), SpherePose.maxTilt) }
        if camera.pose.tilt > 0.05 { lastTilt = camera.pose.tilt }
    }

    func setHeading(bearing: Float) {
        stopFlight()
        northEase = nil
        edit { camera in
            let b = SphereFrame.basis(at: camera.pose.up)
            camera.pose.heading = simd_normalize(cos(bearing) * b.north + sin(bearing) * b.east)
        }
    }

    // MARK: Flights

    /// Flies to look at the direction `place`, ending at `distance` (nil: the distance it has), keeping its heading and tilt.
    func flyTo(_ place: SIMD3<Float>, distance: Float? = nil) {
        guard isActive, simd_length(place) > 0 else { return }
        let target = simd_normalize(place)
        var goal = camera.pose
        let q = SphereFrame.rotation(from: camera.pose.up, to: target)
        goal = goal.rotated(by: q)
        if let distance { goal.distance = min(max(distance, SpherePose.minDistance), camera.maxDistance(aspect: aspect)) }
        startFlight(to: goal, home: false)
    }

    private func startFlight(to goal: SpherePose, home: Bool) {
        stopCoasting()
        zoomEase = nil
        tiltEase = nil
        northEase = nil
        if model.reduceMotion() {
            camera.pose = goal
            cameraIsHome = home
            pushCamera()
            return
        }
        flight = (SphereFlight(from: camera.pose, to: goal, planetRadius: planetRadius), 0, home)
        startAnimating()
    }

    /// An input of the player ends a flight where it is (a follow goes on).
    func stopFlight() {
        flight = nil
        flightGoalDistance = nil
        stopCoasting()
        zoomEase = nil
    }

    private func stopCoasting() { motion.stopCoasting(time: now()) }

    /// A manual drag, `Esc`, a reset: whatever moved the camera by itself stops.
    func cancelMotion(keepSelection: Bool) {
        stopFlight()
        tiltEase = nil
        northEase = nil
        setFollow(.none)
    }

    // MARK: Following

    /// The clicked cell is cleared (a selection of anything else, `Esc`, a click on nothing).
    func clearCell() {
        guard selectedCell != nil else { return }
        selectedCell = nil
        pushDraw()
        overlay.invalidate()
    }

    func setFollow(_ target: FollowTarget) {
        follow.set(target)
        if model.followTarget != target { model.followTarget = target }
        if target.isNone { flightGoalDistance = nil } else { startAnimating() }
    }

    /// The camera goes to the place of a selection: on a storm or a lineage, at the region distance.
    func flyDistanceRegion() -> Float {
        min(camera.pose.distance, max(0.45 * (lidRadius - planetRadius) + 20, 30))
    }

    /// The distance an organism is looked at from.
    func flyDistanceOrganism() -> Float { 10 }

    func fly(toFollowing distance: Float?) {
        flightGoalDistance = distance
        flight = nil
        startAnimating()
    }

    // MARK: Animation

    var isAnimating: Bool {
        flight != nil || !follow.target.isNone || northEase != nil || tiltEase != nil || zoomEase != nil || motion.isCoasting || spinWanted
            || flightGoalDistance != nil
    }

    var spinWanted: Bool {
        guard model.display.spin, !model.reduceMotion(), level == .planet else { return false }
        if let clock = play.store.clock, clock.rate == 0 { return false }
        return true
    }

    func startAnimating() { view?.startAnimating() }

    /// One tick of the display link, `dt` from the last. False when nothing animates any more.
    func advance() -> Bool {
        let t = now()
        let dt = lastAdvance.map { Float(t - $0) } ?? 1.0 / 60
        lastAdvance = t
        let moving = advanceCamera(dt: dt)
        if !moving { lastAdvance = nil }
        view?.needsDisplay = true
        return moving
    }

    private func advanceCamera(dt: Float) -> Bool {
        var easing = false
        let step = min(dt, 0.1)
        if var ease = northEase {
            ease.elapsed += step
            let t = min(ease.elapsed / HeadingEase.duration, 1)
            let north = SphereFrame.basis(at: camera.pose.up).north
            let e = t * t * (3 - 2 * t)
            let from = ease.from - simd_dot(ease.from, camera.pose.up) * camera.pose.up
            let blended = simd_normalize(simd_mix(simd_length(from) > 1e-6 ? simd_normalize(from) : north, north, SIMD3(repeating: e)))
            edit(keepsHome: true) { $0.pose.heading = simd_length(blended) > 1e-6 ? blended : north }
            northEase = t >= 1 ? nil : ease
            easing = t < 1
        }
        if var ease = tiltEase {
            ease.elapsed += step
            let t = min(ease.elapsed / ViewEase.duration, 1)
            let e = t * t * (3 - 2 * t)
            let tilt = ease.from + (ease.to - ease.from) * e
            edit { $0.pose.tilt = tilt }
            tiltEase = t >= 1 ? nil : ease
            easing = easing || t < 1
        }
        if var ease = zoomEase {
            let s = ease.step(dt: step)
            zoomEase = s.done ? nil : ease
            edit { $0.pose.distance = s.distance }
            easing = easing || !s.done
        }
        if motion.isCoasting, let d = motion.coast(dt: step) {
            let size = SIMD2(Float(viewSize.width), Float(viewSize.height))
            edit { $0.pan(dx: d.x, dy: d.y, size: size) }
            easing = true
        }
        if spinWanted {
            let busy = !follow.target.isNone || flight != nil || easing || grabAnchor != nil || model.powers.isAiming
            if motion.spins(enabled: true, time: now(), busy: busy) {
                let angle = GlobeMotion.spinDegrees * Float.pi / 180 * step
                edit(keepsHome: true) { $0.pose = $0.pose.rotated(by: simd_quatf(angle: -angle, axis: SIMD3(0, 0, 1))) }
            }
            easing = true
        }
        if var f = flight {
            f.elapsed += step
            let pose = f.path.pose(at: f.elapsed)
            camera.pose = pose
            pushCamera()
            let done = f.elapsed >= f.path.duration
            flight = done ? nil : f
            if done { cameraIsHome = f.home }
            return !done || easing
        }
        if !follow.target.isNone || flightGoalDistance != nil {
            var pose = camera.pose
            var moving = false
            if let goal = follow.goal(store: play.store) ?? flyGoal {
                moving = SphereFollow.approach(&pose, toward: goal, distance: flightGoalDistance, dt: dt,
                                               constant: flightGoalDistance != nil ? 0.35 : 0.6)
            } else if let distance = flightGoalDistance {
                moving = SphereFollow.approach(&pose, toward: pose.up, distance: distance, dt: dt, constant: 0.35)
            }
            camera.pose = pose
            camera.clamp(aspect: aspect)
            cameraIsHome = false
            pushCamera()
            if !moving { flightGoalDistance = nil; flyGoal = nil }
            return !follow.target.isNone || moving || easing
        }
        return easing
    }

    // MARK: Input: grab, drag, scroll, pinch

    func beginGrab(at point: CGPoint) {
        model.dismissHint()
        let ndc = SpherePicker.ndc(of: point, in: viewSize)
        // a press that is off the planet and its sky grabs the limb: the direction is continuous with the silhouette
        grabAnchor = camera.grabDirection(ndc: ndc, aspect: aspect)
        motion.begin(at: point, time: now())
        grabMoved = false
    }

    func grab(to point: CGPoint) {
        guard let anchor = grabAnchor else { return }
        cancelMotion(keepSelection: true)
        motion.move(to: point, time: now())
        grabMoved = true
        let ndc = SpherePicker.ndc(of: point, in: viewSize)
        edit { $0.grab(anchor: anchor, at: ndc, aspect: self.aspect) }
    }

    func endGrab() {
        defer {
            grabAnchor = nil
            grabMoved = false
        }
        guard grabAnchor != nil else { return }
        let flicked = motion.end(time: now(), allowed: level != .organism && !model.reduceMotion())
        if grabMoved, model.display.keepNorthUp, camera.pose.tilt < 0.4, !camera.pose.isNorthUp {
            northEase = (camera.pose.heading, 0)
            startAnimating()
        }
        if flicked { startAnimating() }
    }

    /// A drag with no anchor: a turn of the heading and the tilt about the point looked at (right or Option drag), or a pan (middle button).
    func drag(dx: Float, dy: Float, orbit: Bool) {
        model.dismissHint()
        motion.stop(time: now())
        cancelMotion(keepSelection: true)
        if orbit {
            edit { $0.turn(dx: dx, dy: dy) }
        } else {
            let size = SIMD2(Float(viewSize.width), Float(viewSize.height))
            edit { $0.pan(dx: dx, dy: dy, size: size) }
        }
    }

    func scroll(_ event: ScrollEvent, at point: CGPoint?) {
        model.dismissHint()
        let onMap = point.map { pressTarget(at: $0) == .map } ?? true
        switch scrollLaw.handle(event, scrollZooms: model.display.scrollZooms) {
        case .none: return
        case .pan(let dx, let dy):
            cancelMotion(keepSelection: true)
            let size = SIMD2(Float(viewSize.width), Float(viewSize.height))
            edit { $0.pan(dx: dx, dy: dy, size: size) }
        case .zoom(let factor):
            zoom(by: factor, at: onMap ? point : nil)
        }
    }

    func magnify(_ magnification: Float, at point: CGPoint?) {
        model.dismissHint()
        let onMap = point.map { pressTarget(at: $0) == .map } ?? true
        zoom(by: ViewerInput.dollyFactor(magnification: magnification), at: onMap ? point : nil)
    }

    /// Multiplies the distance by `factor` about the pointer: the floor point under it stays under it.
    func zoom(by factor: Float, at point: CGPoint? = nil) {
        stopFlight()
        tiltEase = nil
        let ndc = point.map { SpherePicker.ndc(of: $0, in: viewSize) }
        edit { $0.zoom(by: factor, at: ndc, aspect: self.aspect) }
    }

    /// `+` and `-`: the distance goes to `e^(-+0.35)` times what it is, about the centre, over 0.25 s.
    func zoomStep(closer: Bool) {
        let base = zoomEase?.to ?? camera.pose.distance
        stopFlight()
        let factor = exp(WorldHUD.zoomStep * (closer ? -1 : 1))
        let goal = min(max(base * factor, SpherePose.minDistance), camera.maxDistance(aspect: aspect))
        zoomEase = ZoomEase(from: camera.pose.distance, to: goal)
        startAnimating()
    }

    // MARK: Input: clicks and hover

    /// The controls over the planet: the layout of the box's (`WorldHUD`), with the minimap's rectangle turned into the 2:1 chart of the
    /// Mollweide projection (the square the layout reserves, cut to its lower half).
    var hud: WorldHUD {
        var hud = WorldHUD(free: viewSize, minimap: model.display.minimap && minimap != nil, compass: true)
        if let square = hud.minimap {
            let chart = CGRect(x: square.minX, y: square.maxY - square.width / 2, width: square.width, height: square.width / 2)
            hud.minimap = chart
            hud.home = CGRect(x: chart.minX + 3, y: chart.minY + 3, width: WorldHUD.homeSide, height: WorldHUD.homeSide)
        }
        return hud
    }

    func pressTarget(at point: CGPoint) -> HUDTarget { hud.target(at: point) }

    /// The most an organism is drawn with now, in points of the free rectangle (infinite at the physical size): the picker's radius.
    var drawnRadiusCap: Float {
        model.display.sizeMode == .capped ? 16 : .infinity
    }

    func pick(at point: CGPoint) -> SpherePick {
        SpherePicker.pick(at: point, view: viewSize, camera: camera, store: play.store, display: model.display, level: level,
                          maxRadius: drawnRadiusCap)
    }

    func click(at point: CGPoint, shift: Bool, clicks: Int) {
        if model.powers.isAiming {
            if clicks == 1 { aimClick(at: point) }
            return
        }
        switch pick(at: point) {
        case .organism(let uid, _):
            selectedCell = nil
            model.selectOrganism(uid, follow: !shift, camera: !shift, fly: clicks >= 2)
        case .storm(let id):
            selectedCell = nil
            model.selectStorm(id, fly: true, follow: clicks >= 2)
        case .pool(let index):
            selectedCell = nil
            model.selectPool(index)
        case .territory(_, let kin):
            selectedCell = nil
            model.selectKin(id: kin, fly: false, follow: false)
            if clicks >= 2 { dive(at: point) }
        case .cell(let pixel, let direction):
            if clicks == 1 {
                model.clearSelection()
                selectedCell = pixel
                pushDraw()
                overlay.invalidate()
            } else {
                selectedCell = pixel
                flyTo(direction, distance: flyDistanceRegion())
            }
        case .none:
            if clicks == 1 { model.clearSelection() }
        }
    }

    /// A double click on the territory or the floor: fly there, and in to the region distance, unless the camera is closer already.
    func dive(at point: CGPoint) {
        guard let g = camera.floorDirection(ndc: SpherePicker.ndc(of: point, in: viewSize), aspect: aspect) else { return }
        flyTo(g, distance: flyDistanceRegion())
    }

    /// A click while a power is being aimed names its target (the longitude and latitude of the floor under it).
    func aimClick(at point: CGPoint) {
        guard let g = camera.floorDirection(ndc: SpherePicker.ndc(of: point, in: viewSize), aspect: aspect) else {
            model.tell("That point is not on the planet: aim at the floor")
            return
        }
        let (lat, lon) = SphereFrame.latLon(of: g)
        let patches = play.store.patches
        if let command = model.powers.aim(lon: lon, lat: lat, patches: patches) {
            play.send(command)
        } else if model.powers.offer?.scope == .patch, patches.isEmpty {
            model.tell("This world has no food patch to aim at")
        }
    }

    func hover(at point: CGPoint?) {
        if let point, viewSize.width >= 1 {
            let target = pressTarget(at: point)
            overlay.setHUDHover(target, place: nil)
            if target != .map {
                setHover(nil)
                overlay.setPointerText(nil)
                return
            }
        } else {
            overlay.setHUDHover(.map, place: nil)
        }
        guard let point else {
            setHover(nil)
            overlay.setPointerText(nil)
            return
        }
        if model.powers.isAiming {
            let g = camera.floorDirection(ndc: SpherePicker.ndc(of: point, in: viewSize), aspect: aspect)
            overlay.setPointerText(g.map { SphereFrame.placeText($0) })
            setHover(nil)
            overlay.setAim(point: point, world: g.map { SIMD2(Float(SphereFrame.latLon(of: $0).lon), Float(SphereFrame.latLon(of: $0).lat)) })
            return
        }
        overlay.setAim(point: nil, world: nil)
        let t = now()
        if lastHoverPick >= 0, t - lastHoverPick < WorldModel.hoverPeriod, t >= lastHoverPick { return }
        lastHoverPick = t
        let pick = self.pick(at: point)
        if let text = SpherePicker.describe(pick, store: play.store), case .cell = pick {
            setHover(nil)
            overlay.setPointerText(text)
        } else if let text = SpherePicker.describe(pick, store: play.store) {
            setHover((point, pick, text))
            overlay.setPointerText(pointerPlace(at: point, over: pick))
        } else {
            setHover(nil)
            overlay.setPointerText(nil)
        }
    }

    private func pointerPlace(at point: CGPoint, over pick: SpherePick) -> String? {
        if case .organism(let uid, let hint) = pick, let frame = play.store.frame, let i = play.store.listedIndex(ofUID: uid, hint: hint) {
            return "organism " + SphereFrame.placeText(frame.position(i), planetRadius: Double(planetRadius))
        }
        return camera.floorDirection(ndc: SpherePicker.ndc(of: point, in: viewSize), aspect: aspect).map { SphereFrame.placeText($0) }
    }

    private func setHover(_ new: (point: CGPoint, pick: SpherePick, text: String)?) {
        let same = hover?.pick == new?.pick && hover?.point == new?.point
        guard !same || (hover == nil) != (new == nil) else { return }
        hover = new
        overlay.invalidate()
    }

    // MARK: Input: the controls

    func hudPress(_ target: HUDTarget, at point: CGPoint) {
        model.dismissHint()
        hover = nil
        overlay.setPressed(target)
        switch target {
        case .compass:
            if let compass = hud.compass {
                compassTurn = (Compass.dialAngle(of: point, around: CGPoint(x: compass.midX, y: compass.midY)), camera.pose.heading)
            }
        case .tilt:
            guard let track = hud.tiltTrack else { return }
            let thumb = WorldHUD.thumbCentre(tilt: camera.pose.tilt * WorldPose.maxTilt / SpherePose.maxTilt, track: track)
            tiltGrab = abs(point.y - thumb.y) <= WorldHUD.thumbDiameter / 2 ? point.y - thumb.y : 0
            setTilt(hudTilt(atY: point.y, track: track))
        default: break
        }
    }

    private func hudTilt(atY y: CGFloat, track: CGRect) -> Float {
        WorldHUD.tilt(atY: y, track: track, grab: tiltGrab) * SpherePose.maxTilt / WorldPose.maxTilt
    }

    func hudDrag(_ target: HUDTarget, to point: CGPoint) {
        switch target {
        case .minimap: dragMinimap(to: point)
        case .compass:
            guard let turn = compassTurn, let compass = hud.compass else { return }
            guard let nowAngle = Compass.dialAngle(of: point, around: CGPoint(x: compass.midX, y: compass.midY)) else { return }
            guard let from = turn.angle else {
                compassTurn = (nowAngle, camera.pose.heading)
                return
            }
            // the dial turns the screen's up direction about the point looked at by the angle the pointer turned (clockwise on the
            // screen is a turn of the heading to the east)
            let q = simd_quatf(angle: -(nowAngle - from), axis: camera.pose.up)
            stopFlight()
            northEase = nil
            edit { $0.pose.heading = simd_normalize(q.act(turn.heading)) }
        case .tilt:
            guard let track = hud.tiltTrack else { return }
            setTilt(hudTilt(atY: point.y, track: track))
        default: return
        }
    }

    private func minimapCentre() -> Double { 0 }

    private func dragMinimap(to point: CGPoint) {
        guard let box = hud.minimap, let direction = Mollweide.direction(at: CGPoint(x: min(max(point.x, box.minX + 0.5), box.maxX - 0.5) - box.minX, y: min(max(point.y, box.minY + 0.5), box.maxY - 0.5) - box.minY), centre: minimapCentre(), in: box.size) else { return }
        cancelMotion(keepSelection: true)
        edit { camera in
            camera.pose = camera.pose.rotated(by: SphereFrame.rotation(from: camera.pose.up, to: direction))
        }
    }

    func hudRelease(_ target: HUDTarget, at point: CGPoint, clicks: Int, dragged: Bool) {
        overlay.setPressed(nil)
        compassTurn = nil
        let rects = hud
        func inside(_ rect: CGRect?) -> Bool { rect?.contains(point) ?? false }
        switch target {
        case .map, .readout: return
        case .minimap:
            guard !dragged, let box = rects.minimap,
                  let place = Mollweide.direction(at: CGPoint(x: point.x - box.minX, y: point.y - box.minY), centre: minimapCentre(), in: box.size)
            else { return }
            cancelMotion(keepSelection: true)
            flyTo(place, distance: clicks >= 2 ? flyDistanceRegion() : nil)
        case .home:
            if inside(rects.home) { home() }
        case .compass:
            guard !dragged, inside(rects.compass) else { return }
            if clicks >= 2 { home() } else { northUp() }
        case .zoomIn:
            if inside(rects.zoomIn) { zoomStep(closer: true) }
        case .zoomOut:
            if inside(rects.zoomOut) { zoomStep(closer: false) }
        case .toggle3D:
            if inside(rects.toggle3D) { toggle3D() }
        case .tilt:
            tiltGrab = 0
        case .preset(let preset):
            if inside(rects.presets.first { $0.preset == preset }?.rect) { viewPreset(preset) }
        }
    }

    // MARK: What the readouts say

    /// "Northern hemisphere  39N  235E   70 units up": where the view is.
    var readout: String {
        let (lat, lon) = SphereFrame.latLon(of: camera.pose.up)
        let hemisphere = abs(lat) < 5 ? "Equator" : (lat > 0 ? "Northern hemisphere" : "Southern hemisphere")
        let height = max(camera.altitude, 0)
        return "\(hemisphere)  " + SphereFrame.placeText(camera.pose.up) + String(format: "   %.0f units up", height)
    }

    var viewText: String {
        let tilt = Int((camera.pose.tilt * 180 / .pi).rounded())
        let bearing = Int((camera.pose.bearing * 180 / .pi).rounded())
        return "Tilt \(tilt)° · bearing \((bearing + 360) % 360)°"
    }

    /// The words of the cell the player clicked, from the records.
    var cellText: String? {
        guard let selectedCell, let sphere else { return nil }
        return SphereReadout.cellText(pixel: selectedCell, sphere: sphere, store: play.store)
    }

    /// The place and the tilt for the minimap: the floor point under the view's centre and the cap the eye can see.
    func footprint(points: Int = 72) -> [SIMD3<Float>] {
        let r = simd_length(camera.eye)
        guard r > planetRadius else { return [] }
        let alpha = acos(min(planetRadius / r, 1))
        let eyeDir = camera.eye / r
        return SphereOverlay.circle(centre: eyeDir, angle: alpha, radius: 1, count: points)
    }
}

import AlifeCore
import CoreGraphics
import Foundation
import simd

/// The orientation aids over the map (DESIGN §36.5, the spec's 4 and 5.3): the minimap, the compass, the zoom buttons, the
/// readout, the scale bar and the one-time hint; and the height colour mode. The model hit-tests the controls (`pressTarget`), so
/// the Metal view gets every press and no `NSView` lies over the map; the pure geometry is `WorldHUD`, `WorldInset` and
/// `WorldReadout` in AlifeCore.
extension WorldModel {
    // MARK: Where the controls are

    /// The controls over the map for the free rectangle as it is now.
    var hud: WorldHUD {
        WorldHUD(free: freeRect.size, minimap: display.minimap && mapShown, compass: !pose.faceDown)
    }

    /// The minimap's geometry; nil when it is off, hidden by the window's size, or the world has no map.
    var inset: WorldInset? {
        hud.minimap.map { WorldInset(rect: $0, size: worldSize) }
    }

    /// What a press at `point` (points of the free rectangle, origin at the top left) lands on.
    func pressTarget(at point: CGPoint) -> HUDTarget { hud.target(at: point) }

    /// The minimap's rectangle as fractions of the free rectangle (what the renderer is given), nil when there is none.
    func pushInset() {
        var unit: CGRect?
        if let box = hud.minimap, freeRect.width >= 1, freeRect.height >= 1 {
            unit = CGRect(x: box.minX / freeRect.width, y: box.minY / freeRect.height, width: box.width / freeRect.width,
                          height: box.height / freeRect.height)
        }
        renderer?.worldDraw.inset = unit
        overlay.invalidate()
        view?.needsDisplay = true
    }

    // MARK: Pressing a control

    /// The left button went down on a control.
    func hudPress(_ target: HUDTarget) {
        dismissHint()
        overlay.setHover(nil)
        overlay.setPressed(target)
    }

    /// The pointer moved with the button down on `target`. Only the minimap drags: the centre of the view follows the pointer live
    /// (a grab on the minimap's own plane), the pointer held inside the box.
    func hudDrag(_ target: HUDTarget, to point: CGPoint) {
        guard target == .minimap, let inset else { return }
        cancelCameraMotion()
        headingEase = nil
        let inside = CGPoint(x: min(max(point.x, inset.rect.minX + 0.5), inset.rect.maxX - 0.5),
                             y: min(max(point.y, inset.rect.minY + 0.5), inset.rect.maxY - 0.5))
        guard let place = inset.torus(atPoint: inside) else { return }
        // The marker in the box is the floor point under the centre of the picture, which is the pose's own `x`, `y` only when the view
        // is straight down at height 0: the pose goes where the floor under its centre is the place under the pointer.
        let aspect = self.aspect
        editPose {
            let off = $0.offset(at: .zero, aspect: aspect, planeZ: 0) ?? .zero
            $0.x = place.x - off.x
            $0.y = place.y - off.y
        }
    }

    /// The button came up at `point` after a press on `target`. `dragged` is whether the pointer moved far enough to be a drag
    /// (the minimap has already followed it). `clicks` is the click count of the sequence.
    func hudRelease(_ target: HUDTarget, at point: CGPoint, clicks: Int, dragged: Bool) {
        overlay.setPressed(nil)
        let rects = hud
        func inside(_ rect: CGRect?) -> Bool { rect?.contains(point) ?? false }
        switch target {
        case .map: return
        case .minimap:
            guard !dragged, let inset, let place = inset.torus(atPoint: point) else { return }
            cancelCameraMotion()
            headingEase = nil
            // A click flies there keeping the distance; the second click of a double click dives to the region distance.
            let distance = clicks >= 2 ? min(pose.distance, flyDistance(.region)) : pose.distance
            flyTo(SIMD3(place.x, place.y, 0), distance: distance)
        case .home:
            if inside(rects.home) { home() }
        case .compass:
            guard inside(rects.compass) else { return }
            if clicks >= 2 { home() } else { northUp() }
        case .zoomIn:
            if inside(rects.zoomIn) { zoomStep(closer: true) }
        case .zoomOut:
            if inside(rects.zoomOut) { zoomStep(closer: false) }
        }
    }

    /// A press of `+` or `-`: the distance goes to `e^(-+0.35)` times what it is (or is heading for, if a press is still easing),
    /// about the centre, over 0.25 s.
    func zoomStep(closer: Bool) {
        let base = zoomEase?.to ?? pose.distance  // (before `stopFlight`, which ends an ease: presses compose)
        stopFlight()
        let factor = exp(WorldHUD.zoomStep * (closer ? -1 : 1))
        zoomEase = ZoomEase(from: pose.distance, to: clampedDistance(base * factor))
        startAnimating()
    }

    // MARK: Hover over the controls

    /// The pointer is over `target`; for the minimap, `place` is the world point under it.
    func hoverHUD(_ target: HUDTarget, at point: CGPoint) {
        var place: SIMD2<Float>?
        if target == .minimap { place = inset?.torus(atPoint: point) }
        overlay.setHUDHover(target, place: place)
    }

    /// The readout of the pointer on the map, nil off the map or with no weather map: "floor 270 E  28 N" for the floor under the
    /// pointer, or, with the pointer over an organism (`pick`), the organism's own place and height, "organism 300 E  28 N, height 80":
    /// in a 3D world the floor under an organism is not where it is (30 units east of the centre at the ceiling, from a distance of 128
    /// the floor under the pointer was 56 degrees of longitude away).
    func pointerReadout(at point: CGPoint?, over pick: WorldPick? = nil) -> String? {
        guard mapShown, let point, freeRect.width >= 1 else { return nil }
        if case .organism(let uid, let hint)? = pick, let frame = play.store.frame,
           let index = play.store.listedIndex(ofUID: uid, hint: hint) {
            return WorldReadout.organismText(frame.position(index), size: worldSize)
        }
        guard let place = projector(size: freeRect.size).world(at: point, height: 0) else { return nil }
        return WorldReadout.floorText(x: place.x, y: place.y, size: worldSize)
    }

    // MARK: The scale bar and the organism reference

    /// Points of the free rectangle per world unit at the point looked at; nil when it is not on screen or has no scale.
    var pointsPerUnit: Float? {
        guard freeRect.width >= 1, freeRect.height >= 1 else { return nil }
        let at = SIMD3<Float>(pose.x, pose.y, pose.z)
        return projector(size: freeRect.size).points(ofUnits: 1, at: at).map { Float($0) }
    }

    /// The scale bar for the view as it is: a round number of world units, the plane it is true on (the height of the point looked
    /// at: the floor unless an organism is followed) and what an organism is at this scale.
    var scaleBar: (bar: ScaleBar, organism: ScaleBar.OrganismReference?, plane: String?)? {
        guard let scale = pointsPerUnit, let bar = ScaleBar.choose(pointsPerUnit: scale) else { return nil }
        let plane = ScaleBar.planeText(z: pose.z, height: height)
        guard let header = play.store.header else { return (bar, nil, plane) }
        let radius = Float(header.radius)
        let physicalPixels = radius * scale * pixelsPerPoint
        var drawn = physicalPixels
        if display.sizeMode == .capped, drawableSize.height >= 1 {
            let limits = OrganismSizePolicy.limits(
                mode: .capped, cover: coverRadius, distance: camera.distance, size: worldSize, agentRadius: radius,
                viewHeight: Float(drawableSize.height), fovY: camera.fovY, pixelsPerPoint: pixelsPerPoint)
            drawn = min(max(physicalPixels, limits.minimum), max(limits.minimum, limits.cap))
        }
        return (bar, ScaleBar.organism(radius: radius, pointsPerUnit: scale, drawnRadiusPoints: drawn / max(pixelsPerPoint, 0.1)), plane)
    }

    // MARK: The hint

    /// The hint line of the controls, shown once on the first World window and again from View > Controls. A Magic Mouse reports
    /// precise deltas like a trackpad (so its scroll pans, and the system cannot tell the two apart): it zooms with the switch.
    static let hintText = "Drag moves the map. Two-finger scroll pans; pinch or a mouse wheel zooms (Magic Mouse: View > Scroll zooms). H home, N north up, I minimap, C colour. Option-drag tilts."
    static let hintKey = "world.hintShown"

    /// The first time a world opens the hint is shown, until any input.
    func showHintOnce() {
        guard !defaults.bool(forKey: Self.hintKey) else { return }
        hintVisible = true
    }

    /// View > Controls…: the hint again (it goes at the next input).
    func showHint() { hintVisible = true }

    /// Any input dismisses the hint, and it is not shown again by itself.
    func dismissHint() {
        guard hintVisible else { return }
        hintVisible = false
        defaults.set(true, forKey: Self.hintKey)
    }

    // MARK: The colour mode

    /// The colour modes the player can choose: the world's own, then the height, in a world that has some.
    var colourModes: [String] {
        let engine = play.store.header?.colorModes ?? []
        return heightColourAvailable ? engine + [HeightColour.modeName] : engine
    }

    var heightColourAvailable: Bool {
        guard let header = play.store.header else { return false }
        return HeightColour.isAvailable(dims: header.dims, height: Float(header.height))
    }

    /// The colour mode in view: the height, or the engine's.
    var currentColourMode: String {
        if display.heightColour { return HeightColour.modeName }
        let store = play.store
        return store.frame.flatMap { store.header?.colorMode(index: $0.colorIndex) } ?? ""
    }

    /// Chooses a colour mode: the height (computed here) or one of the world's (a command to the world). The choice is kept.
    func setColourMode(_ mode: String) {
        if mode == HeightColour.modeName {
            guard heightColourAvailable else { return }
            display.heightColour = true
        } else {
            display.heightColour = false
            play.setColor(mode)
        }
    }

    /// `c`: the next mode of the list, the height included.
    func cycleColour() {
        if let next = WorldKeys.nextColorMode(after: currentColourMode, in: colourModes) { setColourMode(next) }
    }

    /// A world opens: the height colour is the first colour of a 3D world until the player picks another (`HeightColour.isDefaultIn3D`;
    /// the choice is kept in `world.heightColour`), and does not exist in a 2D one.
    func applyColourDefault() {
        applyingColourDefault = true
        defer { applyingColourDefault = false }
        guard heightColourAvailable else {
            if display.heightColour { display.heightColour = false }  // (a flat world has no height; the stored choice is left alone)
            return
        }
        let stored = defaults.object(forKey: Self.heightColourKey) as? Bool
        display.heightColour = stored ?? HeightColour.isDefaultIn3D
    }
}

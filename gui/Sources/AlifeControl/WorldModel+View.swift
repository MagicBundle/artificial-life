import AlifeCore
import CoreGraphics
import Foundation
import simd

/// The camera controls of a 3D world (DESIGN §36.5, camera controls): the standard views, the 3D button, the tilt handle and the
/// compass dial. The geometry is pure and in AlifeCore (`ViewPreset`, `ViewEase`, `WorldPose`, `WorldHUD`); this applies it to the
/// pose. The tilt reaches edge-on, the distance is held by the tilt's limit (`WorldPose.maxDistance`) and given back when the tilt
/// is relaxed (`distanceWish`), and the far floor is hidden by the haze (`WorldFog`).
extension WorldModel {
    /// Whether the world has a height to tilt over: the view controls and the standard views exist for a 3D world only (a 2D world is
    /// seen from below, with no north up and no horizon).
    var hasTiltableView: Bool { !pose.faceDown && play.store.header?.dims == 3 }

    /// Eases to a standard view (`t`, `o`, `l`, `s`, the strip, the View menu): the tilt, the heading, the height of the point looked at
    /// and the distance go there over half a second (a quarter under reduced motion). A follow goes on (the view is then of the
    /// organism, at its height), a flight under way stops where it is, and any other input ends the ease where it is.
    func viewPreset(_ preset: ViewPreset) {
        guard hasTiltableView, play.store.header != nil else { return }
        dismissHint()
        let following = !followState.target.isNone
        stopFlight()
        headingEase = nil
        if pose.tilt > 0.05 { lastTilt = pose.tilt }
        viewEase = ViewEase(preset: preset, from: pose, wish: distanceWish, aspect: aspect, size: worldSize, height: height,
                            keepHeight: following, duration: reduceMotion() ? ViewEase.reducedDuration : ViewEase.duration)
        startAnimating()
    }

    /// The 3D button: from a view that is angled to straight down (the heading is kept), from the Top view to the angle last used
    /// (Oblique at first), eased.
    func toggle3D() {
        guard hasTiltableView, play.store.header != nil else { return }
        dismissHint()
        stopFlight()
        headingEase = nil
        let angled = pose.tilt > 0.05
        if angled { lastTilt = pose.tilt }
        let goal = angled ? 0 : max(lastTilt, ViewPreset.oblique.tilt * 0.5)
        viewEase = ViewEase(tiltTo: goal, from: pose, wish: distanceWish, aspect: aspect, size: worldSize,
                            duration: reduceMotion() ? ViewEase.reducedDuration : ViewEase.duration)
        startAnimating()
    }

    /// Sets the tilt at once (the handle): `tilt` is clamped to `[0, maxTilt]`; the distance is held by the limit of the new tilt.
    func setTilt(_ tilt: Float) {
        dismissHint()
        stopFlight()
        headingEase = nil
        editPose { $0.tilt = tilt }
        if pose.tilt > 0.05 { lastTilt = pose.tilt }
    }

    /// Sets the heading at once (the compass dial).
    func setHeading(_ heading: Float) {
        dismissHint()
        stopFlight()
        headingEase = nil
        editPose { $0.heading = heading }
    }
}

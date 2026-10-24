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
    ///
    /// The Side view has a distance of its own, so the one the player had is kept (`sideReturn`) and is the wish of the next view
    /// chosen from it, if the zoom has not been touched since: Top after Side goes back to the overview and not to a third of the world.
    func viewPreset(_ preset: ViewPreset) {
        guard hasTiltableView, play.store.header != nil else { return }
        dismissHint()
        let following = Self.followsHeight(followState.target)
        stopFlight()
        headingEase = nil
        if pose.tilt > 0.05 { lastTilt = pose.tilt }
        var wish = distanceWish
        if preset != .side { wish = distanceToReturnTo() ?? wish }
        let (before, tiltBefore) = (distanceWish ?? pose.distance, pose.tilt)
        let ease = ViewEase(preset: preset, from: pose, wish: wish, aspect: aspect, size: worldSize, height: height, keepHeight: following,
                            duration: reduceMotion() ? ViewEase.reducedDuration : ViewEase.duration)
        if preset == .side, tiltBefore < WorldPose.gripTilt { sideReturn = (before: before, at: ease.toDistance) }
        viewEase = ease
        startAnimating()
    }

    /// The distance the player had before the Side view, while the zoom is still where the Side view put it; nil otherwise. The memory
    /// goes once it is used or found stale.
    private func distanceToReturnTo() -> Float? {
        guard let kept = sideReturn else { return nil }
        sideReturn = nil
        guard abs(pose.distance - kept.at) <= 0.02 * kept.at, pose.tilt >= WorldPose.gripTilt else { return nil }
        return kept.before
    }

    /// The 3D button: from a view that is angled to straight down (the heading is kept), from the Top view to the angle last used (the
    /// Oblique one at first, and never steeper than the Low view: the edge-on view is the Side button's and the handle's), eased.
    func toggle3D() {
        guard hasTiltableView, play.store.header != nil else { return }
        dismissHint()
        let following = Self.followsHeight(followState.target)
        stopFlight()
        headingEase = nil
        let angled = pose.tilt > 0.05
        if angled { lastTilt = pose.tilt }
        let goal = angled ? 0 : min(max(lastTilt, ViewPreset.oblique.tilt * 0.5), ViewPreset.low.tilt)
        viewEase = ViewEase(tiltTo: goal, from: pose, wish: distanceToReturnTo() ?? distanceWish, aspect: aspect, size: worldSize,
                            keepHeight: following, duration: reduceMotion() ? ViewEase.reducedDuration : ViewEase.duration)
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

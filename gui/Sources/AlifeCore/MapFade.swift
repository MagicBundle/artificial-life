import Foundation

/// The wall-clock cross-fade between two deliveries of the weather map (DESIGN §33.4.6). The maps change in steps, once per
/// tick; the fade runs for `duration` seconds from the moment a `MAP1` is delivered, so it does not depend on the rate and
/// at a pause nothing moves (a fade that has finished stays finished). The clock is injected: `now` is seconds from any
/// monotonic origin.
package struct MapFade: Equatable, Sendable {
    package static let duration: Double = 0.4

    /// Which image the fade starts from when a map is delivered during one.
    package enum Start: Equatable, Sendable {
        /// The map that was current: the fade had got halfway or further, or none was running.
        case current
        /// The map before it: the fade was less than halfway, and the image on screen is still nearer to that one.
        case previous
    }

    private var startedAt: Double?

    package init() {}

    /// Whether a fade is running at `now` (the display link runs while it is).
    package func isAnimating(at now: Double) -> Bool {
        guard let startedAt else { return false }
        return now - startedAt < Self.duration
    }

    /// The weight of the current map against the previous one: 0 at the delivery, 1 when the fade has run.
    package func blend(at now: Double) -> Float {
        guard let startedAt else { return 1 }
        let t = (now - startedAt) / Self.duration
        guard t < 1 else { return 1 }
        guard t > 0 else { return 0 }
        return Float(t * t * (3 - 2 * t))  // smoothstep
    }

    /// A map was delivered: the fade starts again from the image nearest what is on screen. A single texture pair cannot
    /// hold a mixture, so a delivery during a fade picks whichever end of it the picture is closer to.
    package mutating func deliver(at now: Double) -> Start {
        let start: Start = blend(at: now) >= 0.5 ? .current : .previous
        startedAt = now
        return start
    }

    /// A new world: no fade, the current map is shown whole.
    package mutating func reset() { startedAt = nil }
}

/// What a buffer set has to do to hold the maps the renderer wants to show: which of its two texture sets to swap, and
/// which to upload (DESIGN §33.4.6). A set remembers the serials of the maps its `cur` and `prev` textures hold, so a
/// delivery that moves the old `cur` to `prev` costs one upload, not two. Pure, so the discipline is tested without a GPU.
package struct MapUploadPlan: Equatable, Sendable {
    package var swap: Bool
    package var uploadCur: Bool
    package var uploadPrev: Bool

    package var isEmpty: Bool { !swap && !uploadCur && !uploadPrev }

    package static func make(have: (cur: Int, prev: Int), want: (cur: Int, prev: Int)) -> MapUploadPlan {
        let direct = (have.cur != want.cur ? 1 : 0) + (have.prev != want.prev ? 1 : 0)
        let swapped = (have.prev != want.cur ? 1 : 0) + (have.cur != want.prev ? 1 : 0)
        if swapped < direct {
            return MapUploadPlan(swap: true, uploadCur: have.prev != want.cur, uploadPrev: have.cur != want.prev)
        }
        return MapUploadPlan(swap: false, uploadCur: have.cur != want.cur, uploadPrev: have.prev != want.prev)
    }
}

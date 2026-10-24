import Foundation

/// The standard views of a 3D world (DESIGN §36.5, camera controls): four angles on the same place, each one key, one button of the
/// strip over the map and one item of the View menu, reached by an ease of the tilt, the heading, the height of the point looked at
/// and the distance (`ViewEase`).
///
/// The tilt is measured from straight down (`WorldPose.tilt`): the Top view is 0, the Oblique 45 degrees, the Low 70, the Side 90 (the
/// view axis along the floor, edge-on). The Side view looks at the middle of the height because the pose lifts the point looked at
/// there from a tilt of 70 degrees (`WorldPose.lift`), whatever way the tilt was reached (a preset, the handle, an Option drag).
package enum ViewPreset: String, CaseIterable, Sendable {
    /// Straight down, north up: the map as on paper.
    case top
    /// 45 degrees: the usual angle of a 3D map; heights show.
    case oblique
    /// 70 degrees: the altitude layers stand out, the horizon is in the picture.
    case low
    /// Edge-on, looking at the floor's mid-height: the vertical distribution of the life, 0 to H, at a glance.
    case side

    package var title: String {
        switch self {
        case .top: return "Top"
        case .oblique: return "Oblique"
        case .low: return "Low"
        case .side: return "Side"
        }
    }

    /// The key (`WorldKeys`): none of them is used by another World key.
    package var key: String {
        switch self {
        case .top: return "t"
        case .oblique: return "o"
        case .low: return "l"
        case .side: return "s"
        }
    }

    /// The tilt the view ends at, radians from straight down.
    package var tilt: Float {
        switch self {
        case .top: return 0
        case .oblique: return 45 * .pi / 180
        case .low: return 70 * .pi / 180
        case .side: return .pi / 2
        }
    }

    /// A line for the tooltip and the menu.
    package var help: String {
        switch self {
        case .top: return "Straight down, north up"
        case .oblique: return "45 degrees: the usual 3D angle"
        case .low: return "70 degrees: the altitude layers stand out"
        case .side: return "Edge-on from the middle of the height: the life from 0 to H at a glance"
        }
    }
}

/// The ease to a standard view, or of the tilt alone (the 3D button): the tilt, the heading (the short way round), the height of the
/// point looked at and the distance (in its logarithm) go from where they are to where they end over `duration` seconds, smoothly.
/// Pure: the window gives it the time that passed.
package struct ViewEase: Equatable, Sendable {
    package static let duration: Float = 0.5
    /// The duration of a straight ease (reduce motion).
    package static let reducedDuration: Float = 0.25
    /// The Side view's height of the band of life in the picture, over the picture's height: 0.8, so that the layer from 0 to H
    /// fills four fifths of it at the distance of the point looked at.
    package static let sideFill: Float = 0.8

    package var fromTilt: Float
    package var toTilt: Float
    package var fromHeading: Float
    package var toHeading: Float
    package var fromZ: Float
    package var toZ: Float
    package var fromDistance: Float
    package var toDistance: Float
    package var length: Float
    /// The distance the player's wish is once the ease is over, when the limit of the angle it ends at holds the view nearer than the
    /// wish (the zoom is then given back as the tilt is relaxed again: Top after Low goes back out); nil when nothing holds it, and for
    /// the Side view, which has a distance of its own.
    package var endWish: Float?
    package private(set) var elapsed: Float = 0

    package init(fromTilt: Float, toTilt: Float, fromHeading: Float, toHeading: Float, fromZ: Float, toZ: Float,
                 fromDistance: Float, toDistance: Float, duration: Float = ViewEase.duration) {
        self.fromTilt = fromTilt
        self.toTilt = toTilt
        self.fromHeading = fromHeading
        self.toHeading = toHeading
        self.fromZ = fromZ
        self.toZ = toZ
        self.fromDistance = max(fromDistance, 1e-6)
        self.toDistance = max(toDistance, 1e-6)
        self.length = max(duration, 1e-3)
    }

    /// The ease from `pose` to the standard view `preset`.
    ///
    /// * **Top** goes straight down and turns north up; **Oblique** and **Low** tilt to their angle and keep the heading.
    /// * **Side** looks along the heading from the middle of the height (the pose lifts the point looked at there at edge-on), at the
    ///   distance at which the layer from 0 to `H` fills `sideFill` of the picture (`H / (2 tan(fov/2)) / sideFill`), held to the limit of
    ///   that tilt (the eye over the clear box).
    /// * The height `z` of the point looked at is 0 (the floor) for every view, and unchanged while `keepHeight` is set (the camera
    ///   follows an organism, whose height it is).
    /// * The distance is `wish` (what the player asked for before a tilt held it nearer; the pose's own if there is none), held to the
    ///   limit of the angle it ends at: tilting to the Low view from the Home view comes in, and Top goes back out (`endWish`).
    /// With `globe` (the globe exists for this world) the distance limit is the globe's, and the Side view keeps the distance it is
    /// at once the view is morphing (`s > 0`): there the standard views are angles about the planet, the height fit is for the flat
    /// close-up.
    package init(preset: ViewPreset, from pose: WorldPose, wish: Float?, aspect: Float, size: Float, height: Float, keepHeight: Bool,
                 duration: Float = ViewEase.duration, globe: Bool = false) {
        let tilt = preset.tilt
        let heading = preset == .top ? 0 : pose.heading
        let z = keepHeight ? pose.z : 0
        var wanted = wish ?? pose.distance
        let morphing = globe && !pose.faceDown
            && Morph.s(distance: wish ?? pose.distance, L: size, fovY: pose.fovY, aspect: aspect) > 0
        if preset == .side, height > 0, !morphing {
            wanted = height / (2 * tan(pose.fovY / 2)) / Self.sideFill
        }
        let limit = WorldPose.maxDistance(size: size, fovY: pose.fovY, aspect: aspect, heading: heading, z: z, tilt: tilt,
                                          globe: globe && !pose.faceDown)
        wanted = max(wanted, WorldLevels.minDistance)
        self.init(fromTilt: pose.tilt, toTilt: tilt, fromHeading: pose.heading, toHeading: heading, fromZ: pose.z, toZ: z,
                  fromDistance: pose.distance, toDistance: min(wanted, limit), duration: duration)
        if preset != .side, wanted > limit * (1 + 1e-5) { endWish = wanted }
    }

    /// The ease of the tilt alone (the 3D button): the heading is what it was, the distance is `wish` (held by the limit of the tilt it
    /// ends at, and given back by `endWish`), and the height of the point looked at is what it was, except that it goes to the floor at
    /// the Top view unless `keepHeight` is set (the camera follows an organism): the Top view the Top preset has, not one
    /// of the plane an organism was on.
    package init(tiltTo tilt: Float, from pose: WorldPose, wish: Float?, aspect: Float, size: Float, keepHeight: Bool = true,
                 duration: Float = ViewEase.duration, globe: Bool = false) {
        let z = tilt == 0 && !keepHeight ? 0 : pose.z
        let limit = WorldPose.maxDistance(size: size, fovY: pose.fovY, aspect: aspect, heading: pose.heading, z: z, tilt: tilt,
                                          globe: globe && !pose.faceDown)
        let wanted = max(wish ?? pose.distance, WorldLevels.minDistance)
        self.init(fromTilt: pose.tilt, toTilt: tilt, fromHeading: pose.heading, toHeading: pose.heading, fromZ: pose.z, toZ: z,
                  fromDistance: pose.distance, toDistance: min(wanted, limit), duration: duration)
        if wanted > limit * (1 + 1e-5) { endWish = wanted }
    }

    /// The values `dt` seconds on, and whether the ease is over (it then ends exactly at its end).
    package mutating func step(dt: Float) -> (tilt: Float, heading: Float, z: Float, distance: Float, done: Bool) {
        elapsed += max(dt, 0)
        let t = min(max(elapsed / length, 0), 1)
        if t >= 1 { return (toTilt, toHeading, toZ, toDistance, true) }
        let e = t * t * (3 - 2 * t)
        let turn = Torus.wrapDelta(toHeading - fromHeading, size: 2 * .pi)
        return (fromTilt + (toTilt - fromTilt) * e, WorldPose.wrap(fromHeading + turn * e), fromZ + (toZ - fromZ) * e,
                fromDistance * pow(toDistance / fromDistance, e), false)
    }
}

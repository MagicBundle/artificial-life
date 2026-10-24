import CoreGraphics
import Foundation
import simd

/// The words and numbers of the orientation aids over the map (DESIGN §36.5, the spec's 4.1 and 4.2): the longitude and latitude
/// of a point, the hemisphere, the readout line, the scale bar and the compass's arrow. Pure.
///
/// **Longitude** is `360 x / L` east, from 0 to 359. **Latitude** is the Gudermannian of `2 pi ybar / L`, `atan(sinh(.))`, which is
/// what PR B's globe will draw: the equator is `y = 0` (0 degrees, where the weather is), a warm pool at `y = 36` is 37.5 N, and the
/// wrap at `y = L/2` is 85.05 degrees (the pole rows). On the flat map it is a name for the row, not a scale: a degree of
/// latitude is longer towards the poles.
package enum WorldReadout {
    /// The latitude of a row `ybar` (signed, from the equator) of a world of `size`, in degrees (positive in the hemisphere with f > 0).
    package static func latitude(ybar: Float, size: Float) -> Float {
        guard size > 0 else { return 0 }
        let t = Double(2 * Float.pi * ybar / size)
        return Float(atan(sinh(t)) * 180 / .pi)
    }

    /// The longitude of `x`, in degrees east from 0 to under 360.
    package static func longitude(x: Float, size: Float) -> Float {
        guard size > 0 else { return 0 }
        return Torus.wrap(x, size: size) / size * 360
    }

    /// "264 E  28 N": the longitude and the latitude of a world point, to the degree. The latitude is "0" on the equator.
    package static func coordinates(x: Float, y: Float, size: Float) -> String {
        var lon = Int(longitude(x: x, size: size).rounded())
        if lon >= 360 { lon = 0 }
        let lat = Int(latitude(ybar: Torus.wrapDelta(y, size: size), size: size).rounded())
        let latitude = lat == 0 ? "0" : "\(abs(lat)) \(lat > 0 ? "N" : "S")"
        return "\(lon) E  \(latitude)"
    }

    /// The name of the hemisphere of row `y`: the one with f > 0 ("northern": 0 < y < L/2) or the other; "Equator" within 0.5% of
    /// `L` of the line `y = 0` between them. It follows the sign of the signed distance from the equator, the same number the latitude
    /// is made from, so the two never disagree (at exactly `y = L/2` the name used to say Southern over a latitude of 85 N).
    package static func hemisphereName(y: Float, size: Float) -> String {
        let ybar = Torus.wrapDelta(y, size: size)
        if abs(ybar) < 0.005 * size { return "Equator" }
        return ybar > 0 ? "Northern hemisphere" : "Southern hemisphere"
    }

    /// How high the eye is, as the readout says it: its height over the floor in world units ("205 units up"), or, in a world seen
    /// from below (2D), how far it is from the plane. Not `L`: no other word on the screen is in `L`.
    package static func eyeText(_ eye: Float, awayFromPlane: Bool = false) -> String {
        guard eye.isFinite else { return "" }
        let number = abs(eye) < 10 ? String(format: "%.1f", eye) : String(format: "%.0f", eye)
        return "\(number) units \(awayFromPlane ? "away" : "up")"
    }

    /// The line at the top left of the map: "Northern hemisphere  39 N  235 E   205 units up" for a world with a weather map (it has
    /// hemispheres), "x 120  y 80   205 units up" for one without. The point is the centre of the view, and the height is the eye's
    /// over the floor (`WorldPose.eyeHeight`), not the distance from the point looked at.
    package static func line(pose: WorldPose, size: Float, height: Float = 0, mapShown: Bool) -> String {
        let up = eyeText(pose.eyeHeight(size: size, height: height), awayFromPlane: pose.faceDown)
        if mapShown {
            return "\(hemisphereName(y: pose.y, size: size))  \(coordinates(x: pose.x, y: pose.y, size: size))   \(up)"
        }
        return String(format: "x %.0f  y %.0f   ", pose.x, pose.y) + up
    }

    /// The readout of the angle, over the view controls: "Tilt 45° · heading 270°". The tilt is from straight down (0) to edge-on
    /// (90); the heading is the compass bearing the view looks towards (0 is north up, 90 east, 270 west).
    package static func viewText(pose: WorldPose) -> String {
        "Tilt \(pose.tiltDegrees)° · heading \(pose.bearingDegrees)°"
    }

    /// What the readout of the pointer says about the floor under it: "floor 319 E  28 N". It is only the floor: in a 3D world an
    /// organism above it is somewhere else (`organismText`).
    package static func floorText(x: Float, y: Float, size: Float) -> String {
        "floor " + coordinates(x: x, y: y, size: size)
    }

    /// What it says with the pointer over an organism: where the organism is, not the floor behind it: "organism 300 E  28 N, height 80".
    package static func organismText(_ position: SIMD3<Float>, size: Float) -> String {
        "organism " + coordinates(x: position.x, y: position.y, size: size) + String(format: ", height %.0f", position.z)
    }
}

/// The compass's arrow (the spec's 4.2): where world north, `+y`, points on the screen.
package enum Compass {
    /// The angle of `point` about `centre` on the screen, clockwise from up, in `(-pi, pi]`: what a drag of the compass turns by.
    package static func angle(of point: CGPoint, around centre: CGPoint) -> Float {
        Float(atan2(point.x - centre.x, -(point.y - centre.y)))
    }

    /// The dial's dead zone: a pointer this close to the centre of the compass (in points) has no angle worth turning by.
    package static let deadZone: CGFloat = 6

    /// `angle(of:around:)`, nil within the dead zone about the centre (the angle of a vector of a few points is the noise of the
    /// pointer's position, and a drag that passes through the centre would jump the heading).
    package static func dialAngle(of point: CGPoint, around centre: CGPoint) -> Float? {
        hypot(point.x - centre.x, point.y - centre.y) < deadZone ? nil : angle(of: point, around: centre)
    }

    /// The screen angle of north, clockwise from up, in radians in `(-pi, pi]`: 0 when the map is north up. The exact projection of
    /// the north vector on the screen; near edge-on it swings sideways whatever the heading and at edge-on it is the noise of
    /// `north . right` (`needleAngle` is what the compass draws).
    package static func northAngle(camera: Camera) -> Float {
        let north = SIMD3<Float>(0, 1, 0)
        return atan2(simd_dot(north, camera.right), simd_dot(north, camera.up))
    }

    /// The tilts between which the needle goes from the projection to the heading alone.
    package static let needleBlend: ClosedRange<Float> = (60 * Float.pi / 180)...(85 * Float.pi / 180)

    /// The angle of the compass's needle, clockwise from up, in radians in `(-pi, pi]`: north's projection on the screen
    /// (`atan2(sin h, cos h cos tilt)` for the heading `h`) for a view that looks down, going over smoothly to the heading alone (the
    /// angle of the Top view) from 60 degrees to 85, so that a view looking along the floor shows north where the heading says it is
    /// (up when the view faces north) and not a sideways arrow that is float noise at exactly edge-on.
    package static func needleAngle(heading: Float, tilt: Float) -> Float {
        let range = needleBlend
        let t = min(max((tilt - range.lowerBound) / (range.upperBound - range.lowerBound), 0), 1)
        let w = t * t * (3 - 2 * t)
        let up = cos(heading) * (cos(tilt) * (1 - w) + w)
        return atan2(sin(heading), up)
    }
}

/// A scale bar in world units (the spec's 4.2, and the owner's request after the play of 2026-10-24): a bar of a round number of
/// units (1, 2 or 5 times a power of ten) that is between `minPoints` and `maxPoints` long, true at the point looked at, with the
/// size of an organism for reference.
package struct ScaleBar: Equatable, Sendable {
    package static let minPoints: CGFloat = 60
    package static let maxPoints: CGFloat = 140

    /// The length of the bar in world units, and in points.
    package var units: Float
    package var points: CGFloat

    /// The largest round number of units whose bar is at most `maxPoints` long at `pointsPerUnit`; nil for a scale that is not a
    /// number or is so small that the bar would be under a point.
    package static func choose(pointsPerUnit: Float, maxPoints: CGFloat = ScaleBar.maxPoints) -> ScaleBar? {
        guard pointsPerUnit.isFinite, pointsPerUnit > 1e-6 else { return nil }
        let limit = Float(maxPoints) / pointsPerUnit  // the units that fit
        guard limit.isFinite, limit > 0 else { return nil }
        var decade = pow(10, floor(log10(limit)))
        if !decade.isFinite || decade <= 0 { return nil }
        // 5, 2 or 1 times the decade, whichever is the largest that fits.
        for factor: Float in [5, 2, 1] where factor * decade <= limit * 1.0000001 {
            decade *= factor
            return ScaleBar(units: decade, points: CGFloat(decade * pointsPerUnit))
        }
        return nil
    }

    /// "20 units", "1 unit", "0.5 units".
    package var label: String {
        let text = units >= 1 ? String(format: "%.0f", units) : String(format: "%g", units)
        return units == 1 ? "1 unit" : "\(text) units"
    }

    /// The plane the bar is true on. A bar is the length of a horizontal segment at one height: in a 3D world a segment higher up is
    /// nearer the eye and longer on the screen (an organism at the ceiling is up to 2.6 times larger than the floor's bar says at the
    /// region view), so the bar says which plane it is: "on the floor" for a view of the floor (the height of the point looked at is
    /// 0, as it is unless an organism is followed), "at height 40" for one at 40. Nil for a world with no height (2D).
    package static func planeText(z: Float, height: Float) -> String? {
        guard height > 0, z.isFinite else { return nil }
        return z < 0.5 ? "on the floor" : String(format: "at height %.0f", z)
    }

    /// The bar's label with the plane it is true on: "20 units on the floor".
    package func label(on plane: String?) -> String { plane.map { "\(label) \($0)" } ?? label }

    /// What an organism is, next to the bar: its diameter in world units and in points at this scale, and how many times larger than
    /// that it is drawn (a symbol, §36.7); the drawn size is the size policy's for an organism at the point looked at.
    package struct OrganismReference: Equatable, Sendable {
        package var diameterUnits: Float
        package var physicalPoints: CGFloat
        /// The drawn diameter over the physical one (1 at the physical size).
        package var drawnRatio: Float

        /// The organism as a segment on the bar's own scale: as long as its diameter is at `pointsPerUnit` (never clamped to a size the
        /// plate has no room for, as a ring's radius was), at least `minimum` points so that a distant organism is still a mark, and
        /// at most `room`, with `clipped` set when it is longer (a view zoomed in so far that the organism is wider than the plate;
        /// the plate draws a break in the segment).
        package func segment(minimum: CGFloat = 2, room: CGFloat) -> (points: CGFloat, clipped: Bool) {
            guard physicalPoints.isFinite, physicalPoints <= room else { return (room, true) }
            return (max(physicalPoints, minimum), false)
        }

        /// "organism 2 units", and ", drawn x3.2" where a symbol is drawn more than 25% larger than the organism, or ", drawn x0.6"
        /// where it is more than 20% smaller (the size policy caps it far from the organisms, §36.7).
        package var label: String {
            var text = "organism \(diameterUnits == diameterUnits.rounded() ? String(format: "%.0f", diameterUnits) : String(format: "%g", diameterUnits)) units"
            if drawnRatio >= 1.25 || drawnRatio <= 0.8 { text += String(format: ", drawn x%.1f", drawnRatio) }
            return text
        }
    }

    /// The reference for an organism of `radius` units, at `pointsPerUnit`, drawn with a radius of `drawnRadiusPoints` (the size
    /// policy's, in points of the free rectangle; infinite or 0 for the physical size).
    package static func organism(radius: Float, pointsPerUnit: Float, drawnRadiusPoints: Float) -> OrganismReference? {
        guard radius > 0, pointsPerUnit.isFinite, pointsPerUnit > 0 else { return nil }
        let physical = radius * pointsPerUnit
        var ratio: Float = 1
        if drawnRadiusPoints.isFinite, drawnRadiusPoints > 0, physical > 0 { ratio = drawnRadiusPoints / physical }
        return OrganismReference(diameterUnits: 2 * radius, physicalPoints: CGFloat(2 * physical), drawnRatio: ratio)
    }
}

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
    /// `L` of the line `y = 0` between them.
    package static func hemisphereName(y: Float, size: Float) -> String {
        if abs(Torus.wrapDelta(y, size: size)) < 0.005 * size { return "Equator" }
        return WorldPose.hemisphere(y: y, size: size) == 0 ? "Northern hemisphere" : "Southern hemisphere"
    }

    /// The camera's distance as the readout says it: a fraction of `L`, "0.34 L up".
    package static func distanceText(_ distance: Float, size: Float) -> String {
        guard size > 0 else { return "" }
        return String(format: "%.2f L up", distance / size)
    }

    /// The line at the top left of the map: "Northern hemisphere  39 N  235 E   0.34 L up" for a world with a weather map (it has
    /// hemispheres), "x 120  y 80   0.34 L up" for one without. The point is the centre of the view.
    package static func line(pose: WorldPose, size: Float, mapShown: Bool) -> String {
        let up = distanceText(pose.distance, size: size)
        if mapShown {
            return "\(hemisphereName(y: pose.y, size: size))  \(coordinates(x: pose.x, y: pose.y, size: size))   \(up)"
        }
        return String(format: "x %.0f  y %.0f   ", pose.x, pose.y) + up
    }
}

/// The compass's arrow (the spec's 4.2): where world north, `+y`, points on the screen.
package enum Compass {
    /// The screen angle of north, clockwise from up, in radians in `(-pi, pi]`: 0 when the map is north up.
    package static func northAngle(camera: Camera) -> Float {
        let north = SIMD3<Float>(0, 1, 0)
        return atan2(simd_dot(north, camera.right), simd_dot(north, camera.up))
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

    /// What an organism is, next to the bar: its diameter in world units and in points at this scale, and how many times larger than
    /// that it is drawn (a symbol, §36.7); the drawn size is the size policy's for an organism at the point looked at.
    package struct OrganismReference: Equatable, Sendable {
        package var diameterUnits: Float
        package var physicalPoints: CGFloat
        /// The drawn diameter over the physical one (1 at the physical size).
        package var drawnRatio: Float

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

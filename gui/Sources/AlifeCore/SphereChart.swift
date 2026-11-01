import CoreGraphics
import Foundation
import simd

/// The flat chart of the sphere world (DESIGN §46.8): an equal-area Mollweide chart of the HEALPix cells, recentrable in longitude, for the
/// overview and the minimap. It replaces the Mercator globe for a sphere world (an equal-area chart keeps the cells' areas, which is what
/// the territory and the rain are about). Pure: the chart is a function from a point of the plane to a direction, so the picture is made
/// by looking each pixel up in the cells, with no mesh.
///
/// The chart is normalised to the rectangle `x in [-1, 1]`, `y in [-1/2, 1/2]` (an ellipse of semi-axes 1 and 1/2, aspect 2:1), with
/// north up and east to the right of the centre longitude.
package enum Mollweide {
    /// `(x, y)` of longitude `lon` and latitude `lat`, radians, on a chart centred on `centre` (radians).
    package static func project(lon: Double, lat: Double, centre: Double = 0) -> SIMD2<Double> {
        var dl = lon - centre
        dl -= 2 * .pi * (dl / (2 * .pi)).rounded(.toNearestOrEven)  // [-pi, pi]; the antimeridian is +pi for +pi
        // 2 theta + sin 2 theta = pi sin lat, by Newton's method (it converges in a few steps except at the poles, where theta = +-pi/2)
        let target = Double.pi * sin(lat)
        var theta = lat
        if abs(abs(lat) - .pi / 2) < 1e-9 {
            theta = lat
        } else {
            for _ in 0..<30 {
                let f = 2 * theta + sin(2 * theta) - target
                let d = 2 + 2 * cos(2 * theta)
                if abs(d) < 1e-12 { break }
                let step = f / d
                theta -= step
                if abs(step) < 1e-14 { break }
            }
        }
        let k = 2 * 2.0.squareRoot()
        return SIMD2((2 * 2.0.squareRoot() / .pi) * dl * cos(theta) / k, 2.0.squareRoot() * sin(theta) / k)
    }

    /// The longitude and latitude (radians) of the chart point `(x, y)`; nil outside the ellipse.
    package static func invert(x: Double, y: Double, centre: Double = 0) -> (lon: Double, lat: Double)? {
        let k = 2 * 2.0.squareRoot()
        let s = y * k / 2.0.squareRoot()  // sin theta
        guard abs(s) <= 1 else { return nil }
        let theta = asin(s)
        let c = cos(theta)
        let sinLat = (2 * theta + sin(2 * theta)) / .pi
        let lat = asin(min(max(sinLat, -1), 1))
        guard c > 1e-12 else { return abs(x) < 1e-9 ? (centre, lat) : nil }
        let dl = (x * k * .pi) / (2 * 2.0.squareRoot() * c)
        guard abs(dl) <= .pi + 1e-12 else { return nil }
        var lon = centre + dl
        lon -= 2 * .pi * (lon / (2 * .pi)).rounded()
        return (lon, lat)
    }

    /// The point of the chart of a planet-frame direction.
    package static func project(_ direction: SIMD3<Float>, centre: Double = 0) -> SIMD2<Double> {
        let (lat, lon) = SphereFrame.latLon(of: direction)
        return project(lon: lon * .pi / 180, lat: lat * .pi / 180, centre: centre)
    }

    /// The direction under the chart point, nil outside the ellipse.
    package static func direction(x: Double, y: Double, centre: Double = 0) -> SIMD3<Float>? {
        guard let (lon, lat) = invert(x: x, y: y, centre: centre) else { return nil }
        return SIMD3(Float(cos(lat) * cos(lon)), Float(cos(lat) * sin(lon)), Float(sin(lat)))
    }

    /// The pixels `0xAARRGGBB` of a `width` x `height` image of the chart (pixel rows top to bottom): each pixel inside the ellipse takes
    /// the colour of the field cell its direction is in (`colours`, ring order, of `12 nside^2` cells); outside it is clear (0). A cell's
    /// `colours` entry is `0xAARRGGBB`.
    package static func image(width: Int, height: Int, centre: Double, nside: Int, colours: [UInt32]) -> [UInt32] {
        precondition(colours.count == Healpix.npix(nside), "one colour per cell")
        var out = [UInt32](repeating: 0, count: width * height)
        for py in 0..<height {
            let y = 0.5 - (Double(py) + 0.5) / Double(height)
            for px in 0..<width {
                let x = (Double(px) + 0.5) / Double(width) * 2 - 1
                guard let d = direction(x: x, y: y, centre: centre) else { continue }
                out[py * width + px] = colours[Healpix.pixelFloat(of: d, nside: nside)]
            }
        }
        return out
    }

    /// The view point (in points of a chart rectangle `size`) of a direction.
    package static func point(_ direction: SIMD3<Float>, centre: Double, in size: CGSize) -> CGPoint {
        let p = project(direction, centre: centre)
        return CGPoint(x: (p.x * 0.5 + 0.5) * size.width, y: (0.5 - p.y) * size.height)
    }

    /// The direction under a point of a chart rectangle; nil outside the ellipse.
    package static func direction(at point: CGPoint, centre: Double, in size: CGSize) -> SIMD3<Float>? {
        guard size.width >= 1, size.height >= 1 else { return nil }
        return direction(x: Double(point.x / size.width) * 2 - 1, y: 0.5 - Double(point.y / size.height), centre: centre)
    }
}

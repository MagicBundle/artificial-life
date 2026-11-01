import Foundation
import simd

// The sphere world's geometry as the client reads it (DESIGN §46.7, §46.8; the stream text is `stream protocol v2`).
// Pure values: no UI, no Metal.

/// What a protocol 2 header says about the planet. A box world has none (`StreamHeader.sphere == nil`).
package struct SphereHeader: Equatable, Sendable {
    /// `planet_radius`: the floor is the sphere of this radius.
    package var planetRadius: Double
    /// `height`: the shell thickness `H`; the lid is the sphere of radius `planetRadius + height`.
    package var shellHeight: Double
    /// `shell_edges`: the `FZ + 1` radii of the equal-volume layers, ascending from `planetRadius` to the lid.
    package var shellEdges: [Double]
    /// `nside_field`: the Nside of the field cells, of `STRC`, `FSOL` and of the `MAPH` planes.
    package var nsideField: Int
    /// `nside_weather`: 0 without weather.
    package var nsideWeather: Int
    /// `nside_territory`: the Nside of `TERR`.
    package var nsideTerritory: Int
    package var f0: Double
    /// `weather_depth`: `H1 + H2`, 0 without weather.
    package var weatherDepth: Double

    package init(planetRadius: Double, shellHeight: Double, shellEdges: [Double], nsideField: Int, nsideWeather: Int,
                 nsideTerritory: Int, f0: Double, weatherDepth: Double) {
        self.planetRadius = planetRadius
        self.shellHeight = shellHeight
        self.shellEdges = shellEdges
        self.nsideField = nsideField
        self.nsideWeather = nsideWeather
        self.nsideTerritory = nsideTerritory
        self.f0 = f0
        self.weatherDepth = weatherDepth
    }

    /// The lid's radius.
    package var lidRadius: Double { planetRadius + shellHeight }
    /// `FZ`.
    package var layers: Int { shellEdges.count - 1 }
    /// Cells of one layer of the field (`12 Nside^2`).
    package var fieldCells: Int { Healpix.npix(nsideField) }
    package var territoryCells: Int { Healpix.npix(nsideTerritory) }
    package var hasWeather: Bool { nsideWeather > 0 }

    /// The layer a radius is in (`geom.layer_of`): the equal-volume shells `r_k <= r < r_{k+1}`, clamped to the shell.
    package func layer(ofRadius r: Double) -> Int {
        guard layers > 0 else { return 0 }
        let dv = (pow(lidRadius, 3) - pow(planetRadius, 3)) / Double(layers)
        let k = Int(((r * r * r - pow(planetRadius, 3)) / dv).rounded(.down))
        return min(max(k, 0), layers - 1)
    }

    /// The floor point (radius `planetRadius`) under a direction.
    package func floorPoint(_ direction: SIMD3<Float>) -> SIMD3<Float> { direction * Float(planetRadius) }
}

/// The local frame of the planet at a position, and the places of the sphere in the words of a player (DESIGN §46.7, section 3 of
/// the stream text): `up = p / |p|`, `east = normalise(z x up)` (`(1, 0, 0)` at a pole), `north = up x east`.
package enum SphereFrame {
    /// A direction closer to a pole than this has no east of its own: `|z x up| < 1e-6`.
    package static let poleEpsilon: Float = 1e-6

    package struct Basis: Equatable, Sendable {
        package var up: SIMD3<Float>
        package var east: SIMD3<Float>
        package var north: SIMD3<Float>
    }

    /// The basis at the direction of `position` (any non-zero vector). The centre of the planet has no up: `(0, 0, 1)` is returned.
    package static func basis(at position: SIMD3<Float>) -> Basis {
        let length = simd_length(position)
        let up = length > 0 ? position / length : SIMD3<Float>(0, 0, 1)
        let ze = SIMD3<Float>(-up.y, up.x, 0)
        let zn = simd_length(ze)
        let east = zn < poleEpsilon ? SIMD3<Float>(1, 0, 0) : ze / zn
        return Basis(up: up, east: east, north: simd_cross(up, east))
    }

    /// The heading of an agent at `position` with the transmitted bearing `yaw` (from local north, towards east) and elevation
    /// `pitch`: `t = cos(yaw) north + sin(yaw) east`, `forward = cos(pitch) t + sin(pitch) up`.
    package static func heading(at position: SIMD3<Float>, yaw: Float, pitch: Float) -> SIMD3<Float> {
        let b = basis(at: position)
        let t = cos(yaw) * b.north + sin(yaw) * b.east
        return cos(pitch) * t + sin(pitch) * b.up
    }

    /// The inverse of `heading`: `(yaw, pitch)` of a heading at a position, the yaw in [-pi, pi) and 0 for a vertical heading.
    package static func bearing(at position: SIMD3<Float>, heading: SIMD3<Float>) -> (yaw: Float, pitch: Float) {
        let b = basis(at: position)
        let radial = simd_dot(heading, b.up)
        let t = heading - radial * b.up
        let tn = simd_length(t)
        let pitch = atan2(radial, tn)
        guard tn > 1e-9 * max(simd_length(heading), 1e-30) else { return (0, pitch) }
        var yaw = atan2(simd_dot(t, b.east), simd_dot(t, b.north))
        if yaw >= Float.pi { yaw -= 2 * Float.pi }
        return (yaw, pitch)
    }

    /// Latitude and longitude, degrees, of a direction: `lat = asin(z)` in [-90, 90], `lon = atan2(y, x)` in [-180, 180).
    package static func latLon(of v: SIMD3<Float>) -> (lat: Double, lon: Double) {
        latLon(of: SIMD3<Double>(Double(v.x), Double(v.y), Double(v.z)))
    }

    package static func latLon(of v: SIMD3<Double>) -> (lat: Double, lon: Double) {
        let n = simd_length(v)
        guard n > 0 else { return (0, 0) }
        let lat = asin(min(max(v.z / n, -1), 1)) * 180 / .pi
        var lon = atan2(v.y, v.x) * 180 / .pi
        if lon >= 180 { lon -= 360 }
        return (lat, lon)
    }

    /// The unit vector at latitude `lat` and longitude `lon`, degrees.
    package static func direction(lat: Double, lon: Double) -> SIMD3<Float> {
        let la = lat * .pi / 180, lo = lon * .pi / 180
        return SIMD3(Float(cos(la) * cos(lo)), Float(cos(la) * sin(lo)), Float(sin(la)))
    }

    /// "12.3N 45.6E" (one decimal), the text of every place of the sphere game layer (`geom.place_text`), with " h4.2" for a height.
    package static func placeText(_ v: SIMD3<Float>, planetRadius: Double? = nil) -> String {
        let (lat, lon) = latLon(of: v)
        let text = String(format: "%.1f%@ %.1f%@", abs(lat), lat >= 0 ? "N" : "S", abs(lon), lon >= 0 ? "E" : "W")
        guard let planetRadius else { return text }
        return text + String(format: " h%.1f", Double(simd_length(v)) - planetRadius)
    }

    /// The great-circle angle, radians, between two directions (`atan2(|a x b|, a . b)`: exact at small and antipodal angles).
    package static func angle(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        atan2(simd_length(simd_cross(a, b)), simd_dot(a, b))
    }

    /// The point `s` of the way along the great circle from direction `a` to direction `b` (unit vectors); `a` for coincident
    /// directions, and for antipodal ones the path through the pole side of the plane of `a` and the z axis.
    package static func slerp(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ s: Float) -> SIMD3<Float> {
        let theta = angle(a, b)
        if theta < 1e-6 { return a }
        var axisPart = b - simd_dot(a, b) * a
        if simd_length(axisPart) < 1e-6 {  // antipodal: any perpendicular will do; take the one towards the north pole
            axisPart = simd_cross(simd_cross(a, SIMD3<Float>(0, 0, 1)), a)
            if simd_length(axisPart) < 1e-6 { axisPart = SIMD3<Float>(1, 0, 0) }
        }
        let e = simd_normalize(axisPart)
        let phi = theta * s
        return simd_normalize(cos(phi) * a + sin(phi) * e)
    }

    /// The rotation that takes unit vector `from` to unit vector `to` about the axis perpendicular to both (the identity for
    /// coincident vectors; for antipodal ones a half turn about a perpendicular).
    package static func rotation(from a: SIMD3<Float>, to b: SIMD3<Float>) -> simd_quatf {
        let d = simd_dot(a, b)
        let c = simd_cross(a, b)
        let s = simd_length(c)
        // (the angle comes from the cross product, which keeps its precision where the cosine has none: a turn of 1e-4 is not nothing)
        if s < 1e-9 {
            if d > 0 { return simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1)) }
            var axis = simd_cross(a, SIMD3<Float>(0, 0, 1))
            if simd_length(axis) < 1e-3 { axis = simd_cross(a, SIMD3<Float>(1, 0, 0)) }
            return simd_quatf(angle: .pi, axis: simd_normalize(axis))
        }
        return simd_quatf(angle: atan2(s, d), axis: c / s)
    }
}

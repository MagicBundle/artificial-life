import Foundation
import simd

/// The mapping of the torus onto a globe and back (DESIGN §36.10, the spec's 2.1). Pure.
///
/// The torus is `x, y` in `[0, L)` periodic and `z` in `[0, H]`. With `ybar = Torus.wrapDelta(y)` the signed distance from the equator
/// `y = 0` (north is +y), the globe is the Web-Mercator wrapping of the torus onto a sphere:
///
/// ```text
/// lambda = 2 pi x / L                          longitude: east = +x, the prime meridian is x = 0, seamless
/// phi    = gd(2 pi ybar / L) = atan(sinh(..))  latitude, |phi| <= phiMax = gd(pi) = 85.05 degrees
/// P      = (R + kz z) (cos phi cos lambda, cos phi sin lambda, sin phi)         R = 0.45 L,  kz = 0.06 R / H
/// ```
///
/// The pole of the sphere is +Z. The torus wraps in y at `ybar = +-L/2`, which is `phi = +-phiMax`: the two edges of the mesh, each a
/// pole row closed by a cap. The map is conformal, with `scale(latitude) = 2 pi R cos(phi) / L` sphere units per torus unit.
package enum GlobeMap {
    /// `R / L`.
    package static let radiusFactor: Float = 0.45
    /// The height of the lid (`z = H`) as a fraction of `R`.
    package static let lidFraction: Float = 0.06
    /// `gd(pi)`: the latitude of the pole rows, 85.05 degrees.
    package static let phiMax: Float = 1.4844223

    /// The radius of the surface (z = 0).
    package static func radius(L: Float) -> Float { radiusFactor * L }

    /// `kz`: sphere units of radius per unit of height, so that the lid is a shell of `lidFraction` of `R`.
    package static func shellScale(L: Float, height: Float) -> Float { lidFraction * radius(L: L) / max(height, 1e-6) }

    /// The latitude of the row `ybar`, signed distance from the equator, not wrapped (a seam copy's `ybar + kL` keeps going to the pole).
    package static func latitude(ybar: Float, L: Float) -> Float { atan(sinh(2 * .pi * ybar / L)) }

    /// The latitude of the row `y` (any real y, wrapped), in `[-phiMax, phiMax]`.
    package static func latitude(y: Float, L: Float) -> Float { latitude(ybar: Torus.wrapDelta(y, size: L), L: L) }

    /// The row of a latitude, in `[0, L)`; the seam row `L/2` for `|latitude| >= phiMax` (both poles are the same row).
    package static func y(latitude: Float, L: Float) -> Float {
        guard abs(latitude) < phiMax else { return L / 2 }
        return Torus.wrap(ybar(latitude: latitude, L: L), size: L)
    }

    /// The signed distance from the equator of a latitude (`asinh(tan phi) L / 2 pi`), with `|latitude| < phiMax`.
    package static func ybar(latitude: Float, L: Float) -> Float { asinh(tan(latitude)) * L / (2 * .pi) }

    /// The longitude of the column `x`: `2 pi x / L`, not wrapped.
    package static func longitude(x: Float, L: Float) -> Float { 2 * .pi * x / L }

    /// The unit vector of the point `(x, y)` in world-fixed axes (the pole is +Z).
    package static func direction(x: Float, y: Float, L: Float) -> SIMD3<Float> {
        direction(longitude: longitude(x: x, L: L), latitude: latitude(y: y, L: L))
    }

    package static func direction(longitude: Float, latitude: Float) -> SIMD3<Float> {
        let c = cos(latitude)
        return SIMD3(c * cos(longitude), c * sin(longitude), sin(latitude))
    }

    /// The point `(x, y, z)` of the world on the globe: the surface at radius `R + kz z`.
    package static func point(x: Float, y: Float, z: Float, L: Float, height: Float) -> SIMD3<Float> {
        (radius(L: L) + shellScale(L: L, height: height) * z) * direction(x: x, y: y, L: L)
    }

    /// The torus point of a point on or above the globe. `seam` is true at `|phi| >= phiMax` (inside a cap), where `y` is the seam row
    /// `L/2` and `x` the longitude's column.
    package static func torus(from p: SIMD3<Float>, L: Float, height: Float) -> (x: Float, y: Float, z: Float, seam: Bool) {
        let r = simd_length(p)
        let h = hypot(p.x, p.y)
        let x = Torus.wrap(L * atan2(p.y, p.x) / (2 * .pi), size: L)
        let z = (r - radius(L: L)) / shellScale(L: L, height: height)
        let ybar = asinh(p.z / max(h, 1e-30)) * L / (2 * .pi)  // tan(phi) = z / h, better conditioned than asin
        if abs(ybar) >= L / 2 { return (x, L / 2, z, true) }
        return (x, Torus.wrap(ybar, size: L), z, false)
    }

    /// `(x, y)` of a direction (a unit vector, or any vector: only its direction counts), with the seam flag, on the unit sphere.
    package static func torus(direction d: SIMD3<Float>, L: Float) -> (x: Float, y: Float, seam: Bool) {
        let t = torus(from: d * radius(L: L), L: L, height: 1)
        return (t.x, t.y, t.seam)
    }

    /// `m(phi)`: sphere units per torus unit at a latitude (`2.83 cos phi`).
    package static func scale(latitude: Float, L: Float) -> Float { 2 * .pi * radius(L: L) * cos(latitude) / L }

    /// The engine's hemisphere of a row: 0 for `0 < y < L/2` (f > 0, the northern, `phi > 0`), else 1.
    package static func hemisphere(y: Float, L: Float) -> Int { WorldPose.hemisphere(y: y, size: L) }

    // MARK: Double path (the tests' reference)

    /// `point` in `Double`.
    package static func pointDouble(x: Double, y: Double, z: Double, L: Double, height: Double) -> SIMD3<Double> {
        let R = Double(radiusFactor) * L, kz = Double(lidFraction) * R / height
        let yb = y - L * (y / L).rounded(.toNearestOrEven)
        let lam = 2 * Double.pi * x / L, phi = atan(sinh(2 * Double.pi * yb / L))
        return (R + kz * z) * SIMD3(cos(phi) * cos(lam), cos(phi) * sin(lam), sin(phi))
    }

    /// `torus(from:)` in `Double` (the seam case returns `y = L/2`).
    package static func torusDouble(from p: SIMD3<Double>, L: Double, height: Double) -> (x: Double, y: Double, z: Double, seam: Bool) {
        let R = Double(radiusFactor) * L, kz = Double(lidFraction) * R / height
        let r = (p * p).sum().squareRoot(), h = hypot(p.x, p.y)
        var x = L * atan2(p.y, p.x) / (2 * Double.pi)
        x -= L * (x / L).rounded(.down)
        let yb = asinh(p.z / h) * L / (2 * Double.pi)
        if abs(yb) >= L / 2 { return (x, L / 2, (r - R) / kz, true) }
        return (x, yb - L * (yb / L).rounded(.down), (r - R) / kz, false)
    }
}

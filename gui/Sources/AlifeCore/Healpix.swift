import Foundation
import simd

/// HEALPix in the RING scheme for any integer Nside, the Swift side of the sphere world's cells (DESIGN §46.1, §46.8).
///
/// Three things live here, all tested against the engine's tables (`alife.geom`, written to a fixture by
/// `tests/make_sphere_gui_fixture.py`):
///
/// * `pixel(of:nside:)`, the lookup of a direction in `Double`, the closed form of `geom.ang2pix`;
/// * `pixelFloat(of:nside:)`, the same in `Float` with the operations in the order of the Metal function `healpix_ring`
///   of `ShadersSphere` (the shader is generated from the same constants; a test compares the two and the engine);
/// * `centre(of:nside:)`, `centres(nside:)` and `corners(of:nside:)`, the pixel centres (`geom.pix2vec`) and the four
///   corners of a pixel (the map's cell borders).
///
/// A direction is a unit vector in the planet frame: `z` is north, `x` is longitude 0, `y` is longitude +90 degrees east
/// (`lon = atan2(y, x)`, `lat = asin(z)`).
package enum Healpix {
    package static func npix(_ nside: Int) -> Int { 12 * nside * nside }

    /// The solid angle of one pixel, steradians.
    package static func pixelArea(nside: Int) -> Double { 4 * Double.pi / Double(npix(nside)) }

    // MARK: Direction to pixel

    /// The RING pixel of the direction of `v` (not necessarily unit), in `Double`. The closed form of Gorski et al. (2005) as
    /// `geom.ang2pix` writes it: the equatorial belt for |z| <= 2/3, the polar caps otherwise.
    package static func pixel(of v: SIMD3<Double>, nside n: Int) -> Int {
        let r = (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
        guard r > 0 else { return 0 }
        let z = min(max(v.z / r, -1), 1)
        var phi = atan2(v.y, v.x)
        if phi < 0 { phi += 2 * Double.pi }
        let tt = phi / (0.5 * Double.pi)  // [0, 4]
        let za = abs(z)
        let nn = Double(n)
        let tot = 12 * n * n, ncap = 2 * n * (n - 1)
        if za <= 2.0 / 3.0 {
            let t1 = nn * (0.5 + tt), t2 = nn * z * 0.75
            let jp = Int((t1 - t2).rounded(.down)), jm = Int((t1 + t2).rounded(.down))
            let ir = n + 1 + jp - jm
            let ks = 1 - (ir & 1)
            let ip = floorMod((jp + jm - n + ks + 1) >> 1, 4 * n)
            return ncap + (ir - 1) * 4 * n + ip
        }
        let tp = tt - tt.rounded(.down)
        let tmp = nn * (3 * (1 - za)).squareRoot()
        let jp = Int((tp * tmp).rounded(.down)), jm = Int(((1 - tp) * tmp).rounded(.down))
        let ir = min(max(jp + jm + 1, 1), n)
        let ip = floorMod(Int((tt * Double(ir)).rounded(.down)), 4 * ir)
        return z > 0 ? 2 * ir * (ir - 1) + ip : tot - 2 * ir * (ir + 1) + ip
    }

    /// The same in `Float`, the arithmetic of the shader's `healpix_ring` line by line (a unit vector is expected, as the
    /// shader's input is a normalised position). Adjacent pixels may be returned for a direction within float rounding of a
    /// pixel border; never a pixel further than a neighbour.
    package static func pixelFloat(of v: SIMD3<Float>, nside n: Int) -> Int {
        let z = min(max(v.z, -1), 1)
        var phi = atan2(v.y, v.x)
        if phi < 0 { phi += 2 * Float.pi }
        let tt = phi * (2 / Float.pi)  // [0, 4]
        let za = abs(z)
        let nn = Float(n)
        let tot = 12 * n * n, ncap = 2 * n * (n - 1)
        if za <= 2.0 / 3.0 {
            let t1 = nn * (0.5 + tt), t2 = nn * z * 0.75
            let jp = Int((t1 - t2).rounded(.down)), jm = Int((t1 + t2).rounded(.down))
            let ir = n + 1 + jp - jm
            let ks = 1 - (ir & 1)
            let ip = floorMod((jp + jm - n + ks + 1) >> 1, 4 * n)
            return ncap + (ir - 1) * 4 * n + ip
        }
        let tp = tt - tt.rounded(.down)
        let tmp = nn * (3 * (1 - za)).squareRoot()
        let jp = Int((tp * tmp).rounded(.down)), jm = Int(((1 - tp) * tmp).rounded(.down))
        let ir = min(max(jp + jm + 1, 1), n)
        let ip = floorMod(Int((tt * Float(ir)).rounded(.down)), 4 * ir)
        return z > 0 ? 2 * ir * (ir - 1) + ip : tot - 2 * ir * (ir + 1) + ip
    }

    private static func floorMod(_ a: Int, _ m: Int) -> Int {
        let r = a % m
        return r < 0 ? r + m : r
    }

    // MARK: Pixel to direction

    /// Integer square root, exact.
    private static func isqrt(_ a: Int) -> Int {
        var r = Int(Double(a).squareRoot())
        while r * r > a { r -= 1 }
        while (r + 1) * (r + 1) <= a { r += 1 }
        return r
    }

    /// `(z, phi)` of the centre of RING pixel `p`: `cos(colatitude)` and the longitude in [0, 2 pi).
    package static func zphi(of p: Int, nside n: Int) -> (z: Double, phi: Double) {
        let nn = Double(n)
        let ncap = 2 * n * (n - 1), tot = 12 * n * n
        if p < ncap {  // north cap: ring ir has 4 ir pixels
            let ir = (1 + isqrt(1 + 2 * p)) >> 1
            let ip = p + 1 - 2 * ir * (ir - 1)
            return (1 - Double(ir * ir) / (3 * nn * nn), (Double(ip) - 0.5) * Double.pi / (2 * Double(ir)))
        }
        if p < tot - ncap {  // equatorial belt: 4 n pixels per ring
            let q = p - ncap
            let ir = q / (4 * n) + n
            let ip = q % (4 * n) + 1
            let shift: Double = (ir + n) & 1 == 1 ? 1 : 0.5
            return ((2 * nn - Double(ir)) * 2 / (3 * nn), (Double(ip) - shift) * Double.pi / (2 * nn))
        }
        let q = tot - p  // south cap
        let ir = (1 + isqrt(2 * q - 1)) >> 1
        let ip = 4 * ir + 1 - (q - 2 * ir * (ir - 1))
        return (-1 + Double(ir * ir) / (3 * nn * nn), (Double(ip) - 0.5) * Double.pi / (2 * Double(ir)))
    }

    /// The centre of RING pixel `p` as a unit vector (`geom.pix2vec`).
    package static func centre(of p: Int, nside n: Int) -> SIMD3<Double> {
        let (z, phi) = zphi(of: p, nside: n)
        let s = max(0, 1 - z * z).squareRoot()
        return SIMD3(s * cos(phi), s * sin(phi), z)
    }

    /// All centres of an Nside, as `Float` vectors (cached by the caller).
    package static func centres(nside n: Int) -> [SIMD3<Float>] {
        (0..<npix(n)).map { p in
            let c = centre(of: p, nside: n)
            return SIMD3(Float(c.x), Float(c.y), Float(c.z))
        }
    }

    // MARK: Corners

    /// The four corners of pixel `p` (the west, north, east and south points of the diamond in the RING picture), as unit
    /// vectors: each is the centre of the (virtual) pixel of a refined grid, i.e. found by the HEALPix map of the facet
    /// coordinates. They bound the pixel for drawing a border; they are not used for any lookup.
    package static func corners(of p: Int, nside n: Int) -> [SIMD3<Double>] {
        let (ix, iy, face) = xyf(of: p, nside: n)
        let step = 1 / Double(n)
        let fx = Double(ix) * step, fy = Double(iy) * step
        // the corners of a facet cell in order (0,0), (1,0), (1,1), (0,1) (`geom.CORNERS`)
        return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)].map { dx, dy in
            vector(fx: fx + dx * step, fy: fy + dy * step, face: face)
        }
    }

    private static let jrll = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    private static let jpll = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7]

    /// RING pixel to `(ix, iy, face)`: the pixel's column and row inside its base facet (`geom.pix2xyf`).
    package static func xyf(of pix: Int, nside n: Int) -> (ix: Int, iy: Int, face: Int) {
        let ncap = 2 * n * (n - 1), tot = 12 * n * n
        var iring: Int, iphi: Int, kshift: Int, nr: Int, face: Int
        if pix < ncap {
            iring = (1 + isqrt(1 + 2 * pix)) >> 1
            iphi = pix + 1 - 2 * iring * (iring - 1)
            kshift = 0
            nr = iring
            face = (iphi - 1) / max(iring, 1)
        } else if pix >= tot - ncap {
            let ps = tot - pix
            let nrs = (1 + isqrt(2 * ps - 1)) >> 1
            iphi = 4 * nrs + 1 - (ps - 2 * nrs * (nrs - 1))
            iring = 4 * n - nrs
            kshift = 0
            nr = nrs
            face = 8 + (iphi - 1) / max(nrs, 1)
        } else {
            let q = pix - ncap
            let t = q / (4 * n)
            iring = t + n
            iphi = q - t * 4 * n + 1
            kshift = (iring + n) & 1
            nr = n
            let ire = t + 1, irm = 2 * n + 2 - ire
            let ifm = (iphi - (ire >> 1) + n - 1) / n
            let ifp = (iphi - (irm >> 1) + n - 1) / n
            face = ifp == ifm ? (ifp | 4) : (ifp < ifm ? ifp : ifm + 8)
        }
        let irt = iring - jrll[face] * n + 1
        var ipt = 2 * iphi - jpll[face] * nr - kshift - 1
        if ipt >= 2 * n { ipt -= 8 * n }
        return ((ipt - irt) >> 1, (-(ipt + irt)) >> 1, face)
    }

    /// Continuous facet coordinates in [0, 1]^2 to a unit vector (`geom.xyf2vec`).
    package static func vector(fx: Double, fy: Double, face: Int) -> SIMD3<Double> {
        let jr = Double(jrll[face]) - fx - fy
        let nr = jr < 1 ? jr : (jr > 3 ? 4 - jr : 1)
        let z = jr < 1 ? 1 - nr * nr / 3 : (jr > 3 ? nr * nr / 3 - 1 : (2 - jr) * (2.0 / 3.0))
        let safe = nr > 1e-15
        var t = (Double(jpll[face]) * nr + fx - fy) / (safe ? nr : 1)
        if t < 0 { t += 8 } else if t >= 8 { t -= 8 }
        let phi = safe ? 0.25 * Double.pi * t : 0
        let st = max(0, 1 - z * z).squareRoot()
        return SIMD3(st * cos(phi), st * sin(phi), z)
    }
}

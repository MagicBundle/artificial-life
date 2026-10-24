import Foundation
import simd

/// The wrap of the world in `y`, and how the flat map draws what is near it (DESIGN §36.5, the spec's 2.2 and 3.4).
///
/// The weather map is one patch of one world (`Patch`): `L` wide around the view in x, where the texture repeats without a seam,
/// and `L` tall in y from `kL - L/2` to `kL + L/2`, the cell of the world the view's centre is in. The world wraps in y at
/// `y = L/2` (the second line where the Coriolis parameter changes sign, with no pool and no storm on it); across it the true
/// neighbour is **a copy** of the patch, shifted by `L`, drawn only while the picture reaches within `0.1 L` of the edge.
///
/// **One seam rule for every layer.** What a copy draws, it draws at *that copy's image* of the point and with *that copy's
/// alpha*: the map's quad, the organisms, the bonds, the storm rings and tracks and the picker's answer. On the flat map the copies
/// are all opaque and the images are the minimum image about the view's centre, which is where the copy that covers the point puts
/// it (`image(of:center:size:)` and its tests), so there is no seam to see; the alpha is there for PR B's globe, where the
/// copies fade out over `s` in `[0.1, 0.3]`.
package enum SeamCopies {
    /// The picture must reach this fraction of `L` from the edge of a patch before the neighbouring copy is drawn.
    package static let marginFraction: Float = 0.1
    /// The morph parameter above which no copy is drawn, and the one from which it fades (PR B; 0 on the flat map).
    package static let fadeStart: Float = 0.1
    package static let fadeEnd: Float = 0.3

    /// A copy of the patch: its index `k` (the cell `y` in `[kL - L/2, kL + L/2]`) and its alpha.
    package struct Copy: Equatable, Sendable {
        package var k: Int
        package var alpha: Float

        package init(k: Int, alpha: Float) {
            self.k = k
            self.alpha = alpha
        }
    }

    /// The copy the view's centre is in: the cell `y` is nearest.
    package static func base(y: Float, size: Float) -> Int {
        size > 0 ? Int((y / size).rounded(.toNearestOrEven)) : 0
    }

    /// The alpha of a copy at morph parameter `s`: 1 on the flat map, fading to nothing by `fadeEnd`.
    package static func alpha(s: Float) -> Float {
        let t = min(max((s - fadeStart) / (fadeEnd - fadeStart), 0), 1)
        return 1 - t * t * (3 - 2 * t)
    }

    /// The copies to draw for a view centred on `y` that reaches `extentY` from it along y: the base cell, and its neighbour on a
    /// side when the picture reaches within `marginFraction * L` of that side's edge. Nil extent (the horizon in view) draws both.
    package static func plan(y: Float, extentY: Float?, size: Float, s: Float = 0) -> [Copy] {
        let k = base(y: y, size: size)
        let center = y - Float(k) * size  // in [-L/2, L/2]
        let near = size * (0.5 - marginFraction)
        let a = alpha(s: s)
        var copies = [Copy(k: k, alpha: 1)]
        guard a > 0 else { return copies }
        if extentY == nil || center + (extentY ?? 0) > near { copies.append(Copy(k: k + 1, alpha: a)) }
        if extentY == nil || center - (extentY ?? 0) < -near { copies.append(Copy(k: k - 1, alpha: a)) }
        return copies.sorted { $0.k < $1.k }
    }

    /// The `y` range of the cell of copy `k`.
    package static func range(of copy: Copy, size: Float) -> ClosedRange<Float> {
        let center = Float(copy.k) * size
        return (center - size / 2)...(center + size / 2)
    }

    /// The image of `point` that is drawn: the one nearest the view's centre. It lies in the cell of exactly one copy (or on the
    /// boundary of two), and that copy is the one that draws it.
    package static func image(of point: SIMD2<Float>, center: SIMD2<Float>, size: Float) -> SIMD2<Float> {
        Torus.nearestImage(of: point, to: center, size: size)
    }

    /// The copy of `plan` whose cell holds the image of `point`; nil if none does (a point outside the picture's reach).
    package static func copy(containing point: SIMD2<Float>, center: SIMD2<Float>, plan: [Copy], size: Float) -> Copy? {
        let image = image(of: point, center: center, size: size)
        return plan.filter { $0.alpha > 0 && range(of: $0, size: size).contains(image.y) }.max { $0.alpha < $1.alpha }
    }
}

import CoreGraphics
import Foundation
import simd

/// The minimap's geometry (DESIGN §36.5, the spec's 4.1): a square map of the whole torus, east to the right, **north up**, the
/// equator through the middle and the wrap in `y` (at `L/2`) along its top and bottom edges. A point `(x, y)` of the world is at
/// `u = x / L`, `v = 1/2 - ybar / L` of the box (`ybar` the signed distance from the equator, `Torus.wrapDelta(y)`).
///
/// The view's footprint is the projection of its border onto the floor, drawn in the same box. It is unwrapped about the view's
/// centre, so it is one closed curve that may leave the box on a side; the box is a torus, so the pieces that leave it come back
/// on the opposite side (`pieces(of:)`: a footprint across `x = 0` is two pieces, one across the wrap in `y` is two, and one across
/// a corner is four).
package struct WorldInset: Equatable, Sendable {
    /// The box in the points of the free rectangle.
    package var rect: CGRect
    /// `L`.
    package var size: Float

    package init(rect: CGRect, size: Float) {
        self.rect = rect
        self.size = size
    }

    /// The point of the box for `x` and the signed `ybar`, unwrapped (it may lie outside the box).
    package func point(x: Float, ybar: Float) -> CGPoint {
        let side = rect.width
        return CGPoint(x: rect.minX + CGFloat(x / size) * side, y: rect.minY + CGFloat(0.5 - ybar / size) * side)
    }

    /// The point of the box for a world point: `x` wrapped into `[0, L)` and `y` to the signed distance from the equator.
    package func point(forTorus x: Float, y: Float) -> CGPoint {
        point(x: Torus.wrap(x, size: size), ybar: Torus.wrapDelta(y, size: size))
    }

    /// The world point `(x, y)` in `[0, L)^2` under `p`, nil outside the box. The inverse of `point(forTorus:y:)`.
    package func torus(atPoint p: CGPoint) -> SIMD2<Float>? {
        guard rect.width >= 1, rect.contains(p) else { return nil }
        let u = Float((p.x - rect.minX) / rect.width)
        let v = Float((p.y - rect.minY) / rect.height)
        return Torus.wrap(SIMD2(u * size, (0.5 - v) * size), size: size)
    }

    /// The box's point for the world point under the screen's centre on the floor (the centre of the picture; the view's own `x`,
    /// `y` when the centre ray misses the floor), unwrapped like the footprint.
    package func centre(of pose: WorldPose, aspect: Float) -> CGPoint {
        let at = pose.offset(at: .zero, aspect: aspect, planeZ: 0) ?? .zero
        return unwrapped(pose, offset: at)
    }

    /// The border of the picture on the floor: `perSide` samples along each of the four sides, in order, as points of the box
    /// unwrapped about the centre. Nil if a ray of the border does not meet the floor in front of the eye (the horizon is in
    /// view: the tilt limit keeps this from happening, but the minimap does not rely on it).
    package func footprint(of pose: WorldPose, aspect: Float, perSide: Int = 16) -> [CGPoint]? {
        guard perSide >= 1 else { return nil }
        var out: [CGPoint] = []
        let n = Float(perSide)
        for side in 0..<4 {
            for i in 0..<perSide {
                let t = -1 + 2 * Float(i) / n
                let ndc: SIMD2<Float>
                switch side {
                case 0: ndc = SIMD2(t, 1)  // the top, left to right
                case 1: ndc = SIMD2(1, -t)  // the right side, top to bottom
                case 2: ndc = SIMD2(-t, -1)  // the bottom, right to left
                default: ndc = SIMD2(-1, t)  // the left side, bottom to top
                }
                guard let at = pose.offset(at: ndc, aspect: aspect, planeZ: 0) else { return nil }
                out.append(unwrapped(pose, offset: at))
            }
        }
        return out
    }

    private func unwrapped(_ pose: WorldPose, offset: SIMD2<Float>) -> CGPoint {
        point(x: pose.x + offset.x, ybar: pose.ybar(size: size) + offset.y)
    }

    /// The copies of `polygon` (points of the box, unwrapped) shifted by whole boxes that touch the box, so that what leaves it on
    /// one side shows on the other: a footprint across the wrap in `x`, in `y` or at a corner is two or four pieces. The piece
    /// that is already in the box comes first.
    package func pieces(of polygon: [CGPoint]) -> [[CGPoint]] {
        guard let first = polygon.first else { return [] }
        var minX = first.x, maxX = first.x, minY = first.y, maxY = first.y
        for p in polygon {
            minX = min(minX, p.x)
            maxX = max(maxX, p.x)
            minY = min(minY, p.y)
            maxY = max(maxY, p.y)
        }
        let side = rect.width
        var out: [[CGPoint]] = []
        for dy in [0, -1, 1] as [CGFloat] {
            for dx in [0, -1, 1] as [CGFloat] {
                let shifted = CGRect(x: minX + dx * side, y: minY + dy * side, width: maxX - minX, height: maxY - minY)
                if shifted.intersects(rect) {
                    out.append(polygon.map { CGPoint(x: $0.x + dx * side, y: $0.y + dy * side) })
                }
            }
        }
        return out
    }
}

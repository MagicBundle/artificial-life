import CoreGraphics
import Foundation
import simd

/// A storm as the overlay draws it, in the view's points (origin at the top left).
package struct StormGlyph: Equatable, Sendable {
    package var id: Int
    package var name: String?
    package var named: Bool
    package var hemisphere: Int
    package var center: CGPoint
    /// The ring's radius: the storm's `r` in points, never under 5.
    package var radius: CGFloat
    /// The track, continuous across the seam, ending at `center`.
    package var track: [CGPoint]
    package var zeta: Double
    package var coasting: Bool
}

/// World to view for the `Canvas` overlay and the picker: the camera's projection onto the free rectangle, with every point
/// at the minimum image of the camera target (DESIGN §33.4.6). Pure, in the view's points: the view's size is the free
/// rectangle, so the overlay, the Metal view and the picker agree on one rectangle.
package struct WorldProjector: Equatable, Sendable {
    package var camera: Camera
    /// `L`.
    package var size: Float
    /// The free rectangle's size, in points.
    package var view: CGSize

    package init(camera: Camera, size: Float, view: CGSize) {
        self.camera = camera
        self.size = size
        self.view = view
    }

    package var aspect: Float { view.height >= 1 ? Float(view.width / view.height) : 1 }

    package var level: WorldLevel {
        WorldLevels.level(distance: camera.distance, fovY: camera.fovY, aspect: aspect, size: size)
    }

    /// The image of a world point nearest the camera target.
    package func image(of point: SIMD3<Float>) -> SIMD3<Float> {
        Torus.nearestImage(of: point, to: camera.target, size: size)
    }

    /// The point in the view where `world` (at its minimum image) is drawn; nil behind the eye.
    package func screen(_ world: SIMD3<Float>) -> CGPoint? {
        guard let ndc = camera.ndc(of: image(of: world), aspect: aspect) else { return nil }
        return CGPoint(x: CGFloat(ndc.x * 0.5 + 0.5) * view.width, y: CGFloat(0.5 - ndc.y * 0.5) * view.height)
    }

    /// The length in points of `units` world units across the view at `world`.
    package func points(ofUnits units: Float, at world: SIMD3<Float>) -> CGFloat? {
        let at = image(of: world)
        guard let a = screen(at), let b = screen(at + camera.right * units) else { return nil }
        return hypot(a.x - b.x, a.y - b.y)
    }

    /// The storms of a `STRM` record as glyphs. A storm behind the eye is left out.
    package func glyphs(of storms: PlayStorms) -> [StormGlyph] {
        storms.storms.compactMap { storm in
            let floor = SIMD3<Float>(Float(storm.x), Float(storm.y), 0)
            guard let center = screen(floor) else { return nil }
            let radius = max(points(ofUnits: Float(storm.r), at: floor) ?? 0, 5)
            return StormGlyph(id: storm.id, name: storm.name, named: storm.named, hemisphere: storm.hemi, center: center,
                              radius: radius, track: track(storm), zeta: storm.zeta, coasting: storm.state == "coasting")
        }
    }

    /// The storm's track as points: unwrapped, so continuous across the seam; it is shifted whole by the multiple of `L`
    /// that brings the storm's current position to its minimum image, then wrapped by nothing.
    package func track(_ storm: PlayStorms.Storm) -> [CGPoint] {
        guard let last = storm.track.last, last.count >= 3 else { return [] }
        let here = Torus.nearestImage(of: SIMD2(Float(storm.x), Float(storm.y)),
                                      to: SIMD2(camera.target.x, camera.target.y), size: size)
        let shift = here - SIMD2(Float(last[1]), Float(last[2]))
        return storm.track.compactMap { entry in
            guard entry.count >= 3 else { return nil }
            return screenOfImage(SIMD3(Float(entry[1]) + shift.x, Float(entry[2]) + shift.y, 0))
        }
    }

    /// Like `screen`, for a point that is already at the image to draw (no further wrapping).
    package func screenOfImage(_ world: SIMD3<Float>) -> CGPoint? {
        guard let ndc = camera.ndc(of: world, aspect: aspect) else { return nil }
        return CGPoint(x: CGFloat(ndc.x * 0.5 + 0.5) * view.width, y: CGFloat(0.5 - ndc.y * 0.5) * view.height)
    }
}

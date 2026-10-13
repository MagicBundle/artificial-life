import Foundation
import simd

/// The viewer's camera: one pose that an orbit control and a fly control both edit.
///
/// World axes are the engine's: x and y span the periodic floor plane, z is up (gravity points to -z). The pose is the
/// point looked at (`target`), the `distance` from it back to the eye, and the direction of the view as a `yaw`
/// (heading of its horizontal part, from +x towards +y) and a `pitch` (its elevation; negative looks down). Orbiting
/// turns the direction about the target, so the eye moves; looking (the fly control) turns it about the eye, so the
/// target moves. Panning, moving and dollying move the pose without turning it.
///
/// The screen basis comes from `yaw` alone (`right`) and the direction (`up = right × forward`), so a view straight up
/// or down (`pitch` = ±π/2) is well defined: there `yaw` spins the picture about the view axis.
///
/// Matrices are right handed with the depth range [0, 1] of Metal's clip space; the eye looks down -z in view space.
package struct Camera: Equatable, Sendable {
    package var target: SIMD3<Float>
    package var distance: Float
    package var yaw: Float
    package var pitch: Float
    /// The vertical field of view, in radians.
    package var fovY: Float
    package var near: Float
    package var far: Float

    package static let defaultFovY: Float = 50 * .pi / 180
    package static let maxPitch: Float = .pi / 2
    package static let minDistance: Float = 0.25

    package init(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float,
                 fovY: Float = Camera.defaultFovY, near: Float = 0.1, far: Float = 10_000) {
        self.target = target
        self.distance = max(distance, Self.minDistance)
        self.yaw = Self.wrap(yaw)
        self.pitch = min(max(pitch, -Self.maxPitch), Self.maxPitch)
        self.fovY = fovY
        self.near = near
        self.far = far
    }

    // MARK: Basis

    /// The unit direction the camera looks along.
    package var forward: SIMD3<Float> {
        SIMD3(cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), sin(pitch))
    }

    /// The unit direction of the screen's +x, horizontal: it depends on `yaw` only.
    package var right: SIMD3<Float> { SIMD3(sin(yaw), -cos(yaw), 0) }

    /// The unit direction of the screen's +y.
    package var up: SIMD3<Float> { cross(right, forward) }

    /// The eye.
    package var position: SIMD3<Float> { target - forward * distance }

    // MARK: Matrices

    /// World to view: the eye at the origin, +x right, +y up, the view along -z.
    package func viewMatrix() -> simd_float4x4 {
        let r = right, u = up, f = forward, p = position
        return simd_float4x4(columns: (
            SIMD4(r.x, u.x, -f.x, 0),
            SIMD4(r.y, u.y, -f.y, 0),
            SIMD4(r.z, u.z, -f.z, 0),
            SIMD4(-dot(r, p), -dot(u, p), dot(f, p), 1)))
    }

    /// View to clip: a perspective projection for a viewport of width / height `aspect`, depth [0, 1] from `near` to
    /// `far`.
    package func projectionMatrix(aspect: Float) -> simd_float4x4 {
        let fy = 1 / tan(fovY / 2)
        let range = near - far
        return simd_float4x4(columns: (
            SIMD4(fy / aspect, 0, 0, 0),
            SIMD4(0, fy, 0, 0),
            SIMD4(0, 0, far / range, -1),
            SIMD4(0, 0, near * far / range, 0)))
    }

    package func viewProjectionMatrix(aspect: Float) -> simd_float4x4 {
        projectionMatrix(aspect: aspect) * viewMatrix()
    }

    /// The normalised device coordinates of a world point: x and y in [-1, 1] and depth in [0, 1] inside the view.
    /// Nil for a point at or behind the eye's plane.
    package func ndc(of point: SIMD3<Float>, aspect: Float) -> SIMD3<Float>? {
        let clip = viewProjectionMatrix(aspect: aspect) * SIMD4(point, 1)
        guard clip.w > 0 else { return nil }
        return SIMD3(clip.x, clip.y, clip.z) / clip.w
    }

    // MARK: Controls

    /// Turns the view about the target (the eye moves). Angles in radians; `pitch` stops at straight up or down.
    package mutating func orbit(deltaYaw: Float, deltaPitch: Float) {
        yaw = Self.wrap(yaw + deltaYaw)
        pitch = min(max(pitch + deltaPitch, -Self.maxPitch), Self.maxPitch)
    }

    /// Turns the view about the eye (the target moves, the eye does not).
    package mutating func look(deltaYaw: Float, deltaPitch: Float) {
        let eye = position
        orbit(deltaYaw: deltaYaw, deltaPitch: deltaPitch)
        target = eye + forward * distance
    }

    /// Slides the pose in the screen plane by a drag of `dx`, `dy` pixels (+y down, as the window reports it) in a
    /// viewport `viewportHeight` pixels high, so the point under the cursor follows it at the target's depth.
    package mutating func pan(dx: Float, dy: Float, viewportHeight: Float) {
        guard viewportHeight > 0 else { return }
        let perPixel = 2 * distance * tan(fovY / 2) / viewportHeight
        target += (-dx * right + dy * up) * perPixel
    }

    /// Moves the eye (and the target with it) along the view, the screen's right and the screen's up, by world units.
    package mutating func move(forward f: Float = 0, right r: Float = 0, up u: Float = 0) {
        target += forward * f + right * r + up * u
    }

    /// Scales the distance to the target by `factor` (below 1 comes closer).
    package mutating func dolly(by factor: Float) {
        guard factor > 0, factor.isFinite else { return }
        distance = max(distance * factor, Self.minDistance)
    }

    // MARK: Starting poses

    /// The 3D world seen from outside its south side (-y), above the floor, with x to the right: the whole box in view.
    package static func overview(size: Float, height: Float, aspect: Float = 1,
                                 fovY: Float = Camera.defaultFovY) -> Camera {
        let radius = 0.5 * (2 * size * size + height * height).squareRoot()
        return Camera(target: SIMD3(size / 2, size / 2, height / 2),
                      distance: fitDistance(radius: radius, aspect: aspect, fovY: fovY),
                      yaw: .pi / 2, pitch: -25 * .pi / 180, fovY: fovY, near: 0.1, far: 8 * max(size, height))
    }

    /// The 2D world as the 2D renderer's PNGs and the pygame viewer show it: the plane of the world seen face on with
    /// its column index growing with x to the right and its row index growing with y downwards (`resource[y, x]`). That
    /// is the view from the -z side, looking along +z with y down, so the eye sits below the plane z = 0. A 2D stream
    /// has z = 0 for every row, so nothing is hidden by the choice.
    package static func topDown(size: Float, aspect: Float = 1, fovY: Float = Camera.defaultFovY) -> Camera {
        let distance = fitDistance(radius: size / 2 * 1.02, aspect: aspect, fovY: fovY, face: true)
        return Camera(target: SIMD3(size / 2, size / 2, 0), distance: distance, yaw: .pi / 2, pitch: Self.maxPitch,
                      fovY: fovY, near: 0.1, far: 8 * size)
    }

    /// The starting pose for a stream: the overview of a 3D world, the face-on view of a 2D one.
    package static func home(for header: StreamHeader, aspect: Float = 1) -> Camera {
        header.dims == 3
            ? overview(size: Float(header.size), height: Float(header.height), aspect: aspect)
            : topDown(size: Float(header.size), aspect: aspect)
    }

    /// The distance at which a sphere of `radius` fits the smaller field of view (or, for `face`, a flat square of
    /// half-side `radius` fills it).
    private static func fitDistance(radius: Float, aspect: Float, fovY: Float, face: Bool = false) -> Float {
        let halfY = atan(min(1, aspect) * tan(fovY / 2))
        return face ? radius / tan(halfY) : radius / sin(halfY)
    }

    /// Wraps an angle into [-π, π).
    private static func wrap(_ angle: Float) -> Float {
        let turn = 2 * Float.pi
        var a = angle.truncatingRemainder(dividingBy: turn)
        if a >= .pi { a -= turn }
        if a < -.pi { a += turn }
        return a
    }
}

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
///
/// The clip planes follow the pose. A camera that knows the world it looks at (`scene`) puts the far plane just behind
/// the farthest point of the world and the near plane in front of the nearest, wherever the eye is, so that zooming or
/// flying away never clips the world and the depth buffer's resolution is spent on the world and not on the empty
/// space around it. A camera without a scene uses the fixed `near` and `far`.
package struct Camera: Equatable, Sendable {
    /// The world as a sphere: what the clip planes are fitted to.
    package struct Scene: Equatable, Sendable {
        package var center: SIMD3<Float>
        package var radius: Float

        package init(center: SIMD3<Float>, radius: Float) {
            self.center = center
            self.radius = radius
        }
    }

    package var target: SIMD3<Float>
    package var distance: Float
    package var yaw: Float
    package var pitch: Float
    /// The vertical field of view, in radians.
    package var fovY: Float
    /// The clip planes of a camera without a `scene`, and the least `near` of one with a scene.
    package var near: Float
    package var far: Float
    package var scene: Scene?

    package static let defaultFovY: Float = 50 * .pi / 180
    package static let maxPitch: Float = .pi / 2
    package static let minDistance: Float = 0.25

    package init(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float,
                 fovY: Float = Camera.defaultFovY, near: Float = 0.1, far: Float = 10_000, scene: Scene? = nil) {
        self.target = target
        self.distance = max(distance, Self.minDistance)
        self.yaw = Self.wrap(yaw)
        self.pitch = min(max(pitch, -Self.maxPitch), Self.maxPitch)
        self.fovY = fovY
        self.near = near
        self.far = far
        self.scene = scene
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

    /// The planes of the projection. With a `scene`: the far plane at the farthest point of the world from the eye, and
    /// the near plane at a quarter of the distance to the nearest (never nearer than `near`). The depth that the planes
    /// bound is measured along the view axis, so a point off to the side of a wide view is nearer than its distance to
    /// the eye, by at most the cosine of the view's half diagonal (0.4 even for a window four times wider than high);
    /// the sphere impostors, and agents that stick out of the world by their radius, need some room too. A quarter
    /// keeps all of that in front of the plane. Without a scene: `near` and `far`.
    package var clipPlanes: (near: Float, far: Float) {
        guard let scene else { return (near, far) }
        let toCenter = simd_length(position - scene.center)
        return (max(near, 0.25 * (toCenter - scene.radius)), toCenter + scene.radius)
    }

    /// View to clip: a perspective projection for a viewport of width / height `aspect`, depth [0, 1] from the near to the
    /// far plane (`clipPlanes`).
    package func projectionMatrix(aspect: Float) -> simd_float4x4 {
        let fy = 1 / tan(fovY / 2)
        let (near, far) = clipPlanes
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
        let center = SIMD3(size / 2, size / 2, height / 2)
        return Camera(target: center, distance: fitDistance(radius: radius, aspect: aspect, fovY: fovY),
                      yaw: .pi / 2, pitch: -25 * .pi / 180, fovY: fovY, near: 0.1, far: 8 * max(size, height),
                      scene: Scene(center: center, radius: radius * sceneMargin))
    }

    /// The 2D world as the 2D renderer's PNGs and the pygame viewer show it: the plane of the world seen face on with
    /// its column index growing with x to the right and its row index growing with y downwards (`resource[y, x]`). That
    /// is the view from the -z side, looking along +z with y down, so the eye sits below the plane z = 0. A 2D stream
    /// has z = 0 for every row, so nothing is hidden by the choice.
    package static func topDown(size: Float, aspect: Float = 1, fovY: Float = Camera.defaultFovY) -> Camera {
        let distance = fitDistance(radius: size / 2 * 1.02, aspect: aspect, fovY: fovY, face: true)
        let center = SIMD3(size / 2, size / 2, 0)
        return Camera(target: center, distance: distance, yaw: .pi / 2, pitch: Self.maxPitch,
                      fovY: fovY, near: 0.1, far: 8 * size,
                      scene: Scene(center: center, radius: size / 2 * Float(2).squareRoot() * sceneMargin))
    }

    /// The starting pose for a stream: the overview of a 3D world, the face-on view of a 2D one.
    package static func home(for header: StreamHeader, aspect: Float = 1) -> Camera {
        header.dims == 3
            ? overview(size: Float(header.size), height: Float(header.height), aspect: aspect)
            : topDown(size: Float(header.size), aspect: aspect)
    }

    /// The bounding sphere of a world, as the clip planes see it, is this much larger than the sphere through its corners
    /// (agents lie at most their radius outside the box, and a billboard is wider than its sphere).
    private static let sceneMargin: Float = 1.05

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

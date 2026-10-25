import CoreGraphics
import Foundation

/// The motion that outlives the input on the globe (DESIGN §36.10, the spec's 2.5): the inertia of a flick, the idle spin and the
/// settling of the heading after a drag. Pure: the model feeds it the pointer's samples and the clock and applies what it returns.
package struct GlobeMotion: Equatable, Sendable {
    /// The angular velocity of a flick decays with this time constant, in seconds.
    package static let timeConstant: Float = 0.25
    /// A drag shorter than this has no inertia (a click, a nudge), in seconds.
    package static let minimumDrag: Double = 0.1
    /// The pointer's speed over the last `sampleWindow` seconds that counts as a flick, in points a second.
    package static let minimumFlick: Float = 120
    package static let sampleWindow: Double = 0.08
    /// The inertia ends under this speed, in points a second.
    package static let stopSpeed: Float = 8
    /// The idle spin: degrees of longitude a second, and the seconds of rest after an input before it goes on.
    package static let spinDegrees: Float = 0.5
    package static let spinRest: Double = 4
    /// Keep north up: the heading eases to 0 over this many seconds after a drag.
    package static let northUpDuration: Float = 0.4

    package struct Sample: Equatable, Sendable {
        package var time: Double
        package var point: CGPoint
    }

    private var samples: [Sample] = []
    private var pressedAt: Double?
    /// The velocity of the flick in progress, in points a second (zero: none).
    package private(set) var velocity = SIMD2<Float>.zero
    /// When the last input was, for the spin's rest.
    package var lastInput: Double = -.infinity

    package init() {}

    package var isCoasting: Bool { velocity != .zero }

    // MARK: Inertia

    package mutating func begin(at point: CGPoint, time: Double) {
        samples = [Sample(time: time, point: point)]
        pressedAt = time
        velocity = .zero
        lastInput = time
    }

    package mutating func move(to point: CGPoint, time: Double) {
        guard pressedAt != nil else { return }
        samples.append(Sample(time: time, point: point))
        if samples.count > 12 { samples.removeFirst(samples.count - 12) }
        lastInput = time
    }

    /// The drag ended at `time`: a flick starts the inertia when `allowed` (the globe regime, no reduced motion). Returns whether it did.
    @discardableResult
    package mutating func end(time: Double, allowed: Bool) -> Bool {
        defer {
            samples = []
            pressedAt = nil
            lastInput = time
        }
        guard allowed, let pressed = pressedAt, time - pressed > Self.minimumDrag else { return false }
        let recent = samples.filter { time - $0.time <= Self.sampleWindow }
        guard let first = recent.first, let last = recent.last, last.time - first.time > 0.01, time - last.time < 0.06 else { return false }
        let dt = Float(last.time - first.time)
        let v = SIMD2(Float(last.point.x - first.point.x), Float(last.point.y - first.point.y)) / dt
        guard simd_length_squared(v) > Self.minimumFlick * Self.minimumFlick else { return false }
        velocity = v
        return true
    }

    /// The pointer displacement to apply `dt` seconds on (the flick goes on as if the pointer went on, slowing down); nil when over.
    package mutating func coast(dt: Float) -> SIMD2<Float>? {
        guard isCoasting else { return nil }
        let decay = exp(-dt / Self.timeConstant)
        // the distance of a decaying exponential over dt, so the total does not depend on the frame rate
        let moved = velocity * (Self.timeConstant * (1 - decay))
        velocity *= decay
        if simd_length(velocity) < Self.stopSpeed { velocity = .zero }
        return moved
    }

    /// Any input but the drag in progress: the inertia ends where it is.
    package mutating func stopCoasting(time: Double) {
        velocity = .zero
        lastInput = time
    }

    /// A drag without an anchor, a reset: the inertia and the samples of a grab are gone.
    package mutating func stop(time: Double) {
        velocity = .zero
        samples = []
        pressedAt = nil
        lastInput = time
    }

    // MARK: Idle spin

    /// Whether the spin runs at `time`: on, rested, and nothing else moves or is aimed.
    package func spins(enabled: Bool, time: Double, busy: Bool) -> Bool {
        enabled && !busy && !isCoasting && time - lastInput >= Self.spinRest
    }

    /// The change of the centre's column over `dt` seconds (the camera goes west, so the surface turns east, as a planet does).
    package static func spinStep(dt: Float, size: Float) -> Float { -size * spinDegrees / 360 * dt }
}

import simd

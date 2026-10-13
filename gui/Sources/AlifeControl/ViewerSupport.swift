import AlifeCore
import Foundation

// Small pure pieces of the 3D viewer, kept apart from the AppKit and Metal code so that tests can reach them.

// MARK: - What is drawn

/// Which layers of the scene are drawn. The field slice is not here: it is drawn whenever the stream sends one
/// (`StreamController.setFields`).
struct ViewerDisplay: Equatable, Sendable {
    var agents = true
    var bonds = true
    /// The box edges and, in a 3D world, the translucent floor and surface planes.
    var frame = true
}

/// How a drag edits the camera. Both controls edit the same pose (`Camera`): orbit turns it about the target, fly
/// about the eye.
enum CameraMode: String, CaseIterable, Identifiable, Sendable {
    case orbit = "Orbit"
    case fly = "Fly"
    var id: String { rawValue }
}

// MARK: - Keys

/// What a key does. The keys follow the 2D viewer (DESIGN §20.2) where there is an equivalent.
enum ViewerAction: Equatable, Sendable {
    case togglePause, step, faster, slower
    case cycleColor, toggleFields, fieldLayerDown, fieldLayerUp
    case snapshot
    case toggleAgents, toggleBonds, toggleFrame
    case toggleCameraMode, resetCamera
}

enum ViewerKeys {
    /// The action of a key, from `NSEvent.charactersIgnoringModifiers`; nil for a key that does nothing.
    static func action(for characters: String) -> ViewerAction? {
        switch characters.lowercased() {
        case " ": return .togglePause
        case ".": return .step
        case "+", "=": return .faster
        case "-", "_": return .slower
        case "c": return .cycleColor
        case "f", "4": return .toggleFields
        case "[": return .fieldLayerDown
        case "]": return .fieldLayerUp
        case "s", "p": return .snapshot
        case "1": return .toggleAgents
        case "2": return .toggleBonds
        case "3": return .toggleFrame
        case "o": return .toggleCameraMode
        case "r": return .resetCamera
        default: return nil
        }
    }
}

// MARK: - Speed

/// The stream's frame period (`every K`): K steps of the simulation per frame. Doubling and halving within [1, 1024],
/// like the 2D viewer's steps per frame.
enum ViewerSpeed {
    static let range = 1...1024
    static func faster(_ every: Int) -> Int { min(max(every, 1) * 2, range.upperBound) }
    static func slower(_ every: Int) -> Int { max(every / 2, range.lowerBound) }
}

// MARK: - Mouse

/// How pointer movements edit the camera. Pixel deltas are the window's: +x to the right and +y downwards.
enum ViewerInput {
    static let radiansPerPixel: Float = 0.008

    /// The turn for a drag of `dx`, `dy` pixels, as `Camera.orbit(deltaYaw:deltaPitch:)` and `look` take it. Dragging
    /// right turns the view towards the right (the scene seems to turn with the pointer), dragging down looks down.
    static func turn(dx: Float, dy: Float) -> (yaw: Float, pitch: Float) {
        (-dx * radiansPerPixel, -dy * radiansPerPixel)
    }

    /// The factor on the distance for a scroll of `delta` (positive scrolls up, which comes closer). A wheel reports
    /// lines and a trackpad points, so a wheel notch counts for ten.
    static func dollyFactor(scroll delta: Float, precise: Bool) -> Float {
        let exponent = max(min(-delta * (precise ? 0.01 : 0.1), 1.5), -1.5)
        return exp(exponent)
    }

    /// The factor on the distance for a pinch of `magnification` (positive spreads, which comes closer).
    static func dollyFactor(magnification: Float) -> Float { 1 / max(1 + magnification, 0.1) }

    /// The distance a key press moves a flying camera: a fraction of the world, five times more with Shift.
    static func flyStep(size: Float, fast: Bool) -> Float { size / 60 * (fast ? 5 : 1) }
}

// MARK: - The stream process

/// Where a stream comes from.
struct StreamSource: Equatable, Sendable {
    var run: String
    /// `--follow`: the run's newest checkpoints (the faithful view of a director run). Otherwise `alife stream`
    /// re-simulates from the latest checkpoint, without the director.
    var follow: Bool
}

enum StreamLaunch {
    /// The arguments of `alife stream` (after the executable) for `source`, carrying the view the viewer had so that a
    /// restart does not lose it. `--every` applies to a re-simulation only (`--follow` refuses it).
    static func arguments(for source: StreamSource, every: Int, color: String, fieldLayer: Int?) -> [String] {
        var args = ["stream", "runs/\(source.run)"]
        if source.follow {
            args.append("--follow")
        } else if every > 1 {
            args += ["--every", "\(every)"]
        }
        args += ["--color", color]
        if let fieldLayer { args += ["--field-z", "\(fieldLayer)"] }
        return args
    }
}

enum ViewerSnapshot {
    /// `runs/NAME/media/view3d_<step>.png` under `directory` (the repository), next to the 2D viewer's `view_<step>.png`.
    static func url(directory: URL, run: String, step: UInt64) -> URL {
        directory.appendingPathComponent("runs/\(run)/media/view3d_\(step).png")
    }
}

/// What the viewer has asked the stream to do, as far as the viewer knows. The frames confirm it: the colour and the
/// field section of a frame say what was drawn.
struct ViewerControlState: Equatable, Sendable {
    var paused = false
    /// Steps per frame (a re-simulation only).
    var every = 1
    var color = "tag"
    var fields = false
    var fieldLayer = 0
}

// MARK: - Camera edits

/// The keys that move the camera.
enum ArrowKey: Equatable, Sendable {
    case left, right, up, down, pageUp, pageDown
}

extension Camera {
    /// A drag of `dx`, `dy` pixels (+y down): a pan when `pan` (a right or other button, or Option), else a turn about the
    /// target (`orbit`) or the eye (`fly`).
    mutating func drag(dx: Float, dy: Float, mode: CameraMode, pan: Bool, viewportHeight: Float) {
        if pan {
            self.pan(dx: dx, dy: dy, viewportHeight: viewportHeight)
            return
        }
        let turn = ViewerInput.turn(dx: dx, dy: dy)
        switch mode {
        case .orbit: orbit(deltaYaw: turn.yaw, deltaPitch: turn.pitch)
        case .fly: look(deltaYaw: turn.yaw, deltaPitch: turn.pitch)
        }
    }

    /// A zoom by `factor` on the distance (below 1 comes closer): orbiting scales the distance to the target, flying
    /// moves ahead by `reach` times the part of the way that the factor removes.
    mutating func zoom(by factor: Float, mode: CameraMode, reach: Float) {
        switch mode {
        case .orbit: dolly(by: factor)
        case .fly: move(forward: (1 - factor) * reach)
        }
    }

    /// An arrow or page key: orbiting turns about the target (left and right) and dollies (up and down); flying moves
    /// `step` world units ahead, back, sideways or up and down.
    mutating func press(_ key: ArrowKey, mode: CameraMode, step: Float) {
        switch (mode, key) {
        case (.orbit, .left): orbit(deltaYaw: 0.1, deltaPitch: 0)
        case (.orbit, .right): orbit(deltaYaw: -0.1, deltaPitch: 0)
        case (.orbit, .up), (.orbit, .pageUp): dolly(by: 0.9)
        case (.orbit, .down), (.orbit, .pageDown): dolly(by: 1 / 0.9)
        case (.fly, .up): move(forward: step)
        case (.fly, .down): move(forward: -step)
        case (.fly, .left): move(right: -step)
        case (.fly, .right): move(right: step)
        case (.fly, .pageUp): move(up: step)
        case (.fly, .pageDown): move(up: -step)
        }
    }
}

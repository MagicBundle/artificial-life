import AlifeCore
import AppKit
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

// MARK: - Keys that follow the window

/// The observer's keys in the `alife` window reach it whichever view of the window has the focus, as the played world's do (`WorldMTKView`): after
/// a click in the sidebar or in the Run panel the focus is there, and Space, `c` or `s` would otherwise do nothing, or press the focused button. The
/// picture does not take the focus when it appears (showing Watch must not take the keyboard from the sidebar, a click on the picture does).
/// A text being edited keeps its keys, a Command or Control chord is the menu's, and the arrow and Page keys are the focused view's own (the
/// sidebar's list moves with them): they steer the camera only while the picture has the focus.
@MainActor
final class ViewerKeyMonitor {
    private var monitor: Any?

    /// The action a key event asks for; nil when it is not the observer's: a chord, a key that is not in the table, or a text being edited.
    /// `eventWindow` is the window the event was sent to; only the keys of `window` itself are the observer's (the Settings window and the
    /// Help window have their own). It is a parameter, not read off the event, because a window that is not on a screen has no number to find it by.
    static func action(for event: NSEvent, eventWindow: NSWindow?, in window: NSWindow?, firstResponder: NSResponder?) -> ViewerAction? {
        guard let window, eventWindow === window else { return nil }
        if firstResponder is NSText { return nil }
        if event.modifierFlags.contains(.command) || event.modifierFlags.contains(.control) { return nil }
        guard let characters = event.charactersIgnoringModifiers else { return nil }
        return ViewerKeys.action(for: characters)
    }

    func install(in view: NSView, perform: @escaping @MainActor (ViewerAction) -> Void) {
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak view] event in
            nonisolated(unsafe) let key = event
            let handled = MainActor.assumeIsolated { () -> Bool in
                guard let view, let window = view.window,
                      let action = Self.action(for: key, eventWindow: key.window, in: window, firstResponder: window.firstResponder) else { return false }
                perform(action)
                return true
            }
            return handled ? nil : event
        }
    }

    func remove() {
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
    }

    var isInstalled: Bool { monitor != nil }
}

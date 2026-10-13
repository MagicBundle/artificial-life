import AppKit
import MetalKit

/// The viewer's drawing surface: an `MTKView` that takes the mouse and the keyboard. The pointer edits the camera and
/// the keys follow the 2D viewer (`ViewerKeys`); both go through the `ViewerModel`.
///
/// Left drag orbits (or looks, when flying), right drag, middle drag and Option-drag pan, the scroll wheel and the
/// trackpad zoom, and a pinch zooms too. Arrow keys orbit and dolly, or fly (Shift for faster, Page Up and Page Down
/// for up and down).
@MainActor
final class ViewerMTKView: MTKView {
    weak var model: ViewerModel?

    override var acceptsFirstResponder: Bool { true }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)  // the keys work as soon as the window opens
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
    }

    override func mouseDragged(with event: NSEvent) {
        model?.drag(dx: Float(event.deltaX), dy: Float(event.deltaY), pan: event.modifierFlags.contains(.option))
    }

    override func rightMouseDragged(with event: NSEvent) {
        model?.drag(dx: Float(event.deltaX), dy: Float(event.deltaY), pan: true)
    }

    override func otherMouseDragged(with event: NSEvent) {
        model?.drag(dx: Float(event.deltaX), dy: Float(event.deltaY), pan: true)
    }

    override func scrollWheel(with event: NSEvent) {
        model?.scroll(delta: Float(event.scrollingDeltaY), precise: event.hasPreciseScrollingDeltas)
    }

    override func magnify(with event: NSEvent) {
        model?.magnify(Float(event.magnification))
    }

    override func keyDown(with event: NSEvent) {
        if event.modifierFlags.contains(.command) {
            super.keyDown(with: event)  // menu shortcuts
            return
        }
        let fast = event.modifierFlags.contains(.shift)
        switch event.specialKey {
        case .leftArrow?: model?.press(.left, fast: fast)
        case .rightArrow?: model?.press(.right, fast: fast)
        case .upArrow?: model?.press(.up, fast: fast)
        case .downArrow?: model?.press(.down, fast: fast)
        case .pageUp?: model?.press(.pageUp, fast: fast)
        case .pageDown?: model?.press(.pageDown, fast: fast)
        default:
            if let characters = event.charactersIgnoringModifiers, let action = ViewerKeys.action(for: characters) {
                model?.perform(action)
            }
        }
    }
}

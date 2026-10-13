import AppKit
import SwiftUI

/// Runs `onClose` when the window that contains this (invisible) view is about to close. `onDisappear` is not enough to
/// end a process on: SwiftUI decides when a scene's content leaves the hierarchy, and a stream left running keeps the
/// engine busy. With `restorable` false the window is also left out of the windows macOS reopens at the next launch.
struct WindowCloseObserver: NSViewRepresentable {
    var restorable = true
    var onClose: @MainActor () -> Void

    func makeNSView(context: Context) -> ObserverView {
        let view = ObserverView(frame: .zero)
        view.restorable = restorable
        view.onClose = onClose
        return view
    }

    func updateNSView(_ view: ObserverView, context: Context) {
        view.onClose = onClose
    }

    @MainActor
    final class ObserverView: NSView {
        var onClose: @MainActor () -> Void = {}
        var restorable = true
        private var token: (any NSObjectProtocol)?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            if let token {
                NotificationCenter.default.removeObserver(token)
                self.token = nil
            }
            guard let window else { return }
            if !restorable { window.isRestorable = false }
            token = NotificationCenter.default.addObserver(
                forName: NSWindow.willCloseNotification, object: window, queue: .main
            ) { [weak self] _ in
                Task { @MainActor in self?.onClose() }
            }
        }
    }
}

import AppKit
import SwiftUI

/// Which windows of each group are open, so that the last one closing can be told from the others. Windows are held
/// weakly: one that goes away without a close notification does not stay counted.
@MainActor
final class WindowGroups {
    static let shared = WindowGroups()
    private var groups: [String: NSHashTable<AnyObject>] = [:]

    init() {}

    func add(_ window: AnyObject, to group: String) {
        // Not `groups[group, default: ...]`: the table is a class, so adding to the default would not store it.
        let table = groups[group] ?? NSHashTable<AnyObject>.weakObjects()
        table.add(window)
        groups[group] = table
    }

    /// Takes `window` out of `group` and returns how many windows of the group are still open.
    func remove(_ window: AnyObject, from group: String) -> Int {
        guard let table = groups[group] else { return 0 }
        table.remove(window)
        return table.allObjects.count
    }
}

/// Runs `onClose` when the window that contains this (invisible) view is about to close. `onDisappear` is not enough to
/// end a process on: SwiftUI decides when a scene's content leaves the hierarchy, and a stream left running keeps the
/// engine busy. With `restorable` false the window is also left out of the windows macOS reopens at the next launch.
///
/// Windows whose observers name the same `group` count together: `onClose` then runs when the last of them closes, not
/// for each (the main scene is a `WindowGroup`, so there can be several main windows, and closing one of them is not
/// the app ending).
///
/// With `shouldClose` the window is also asked before it closes (`windowShouldClose`, which the red button, Command-W and `performClose`
/// all go through): false keeps it open. SwiftUI owns a window's delegate, so the observer puts a forwarding delegate in front of it
/// (`CloseGuardDelegate`), which hands every other message to the original.
struct WindowCloseObserver: NSViewRepresentable {
    var restorable = true
    var group: String?
    var shouldClose: (@MainActor () -> Bool)?
    var onClose: @MainActor () -> Void

    func makeNSView(context: Context) -> ObserverView {
        let view = ObserverView(frame: .zero)
        view.restorable = restorable
        view.group = group
        view.onClose = onClose
        view.shouldClose = shouldClose
        return view
    }

    func updateNSView(_ view: ObserverView, context: Context) {
        view.onClose = onClose
        view.shouldClose = shouldClose
        view.installCloseGuard()  // (SwiftUI may have put its own delegate back)
    }

    @MainActor
    final class ObserverView: NSView {
        var onClose: @MainActor () -> Void = {}
        var shouldClose: (@MainActor () -> Bool)?
        var restorable = true
        var group: String?
        private var token: (any NSObjectProtocol)?
        private var closeGuard: CloseGuardDelegate?
        /// The window this view is registered with in its group.
        private weak var registered: NSWindow?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            if let token {
                NotificationCenter.default.removeObserver(token)
                self.token = nil
            }
            // With no window the view is being taken down, perhaps by the close that is under way: what is registered
            // stays until `windowWillClose` has counted it.
            guard let window else { return }
            if let registered, registered !== window, let group {
                _ = WindowGroups.shared.remove(registered, from: group)
            }
            if !restorable { window.isRestorable = false }
            installCloseGuard()
            if let group {
                WindowGroups.shared.add(window, to: group)
                registered = window
            }
            token = NotificationCenter.default.addObserver(
                forName: NSWindow.willCloseNotification, object: window, queue: .main
            ) { [weak self] _ in
                // The view is held across the hop to the main actor: if SwiftUI lets go of the closing window's content
                // before the task runs, a weak reference would be nil and the close would go unnoticed.
                guard let self else { return }
                Task { @MainActor in self.windowWillClose() }
            }
        }

        /// Puts the forwarding delegate in front of the window's own, once (and again if the window's delegate was replaced).
        func installCloseGuard() {
            guard shouldClose != nil, let window else { return }
            if let installed = window.delegate as? CloseGuardDelegate, installed === closeGuard { return }
            let guardDelegate = closeGuard ?? CloseGuardDelegate { [weak self] in self?.shouldClose?() ?? true }
            guardDelegate.original = window.delegate
            closeGuard = guardDelegate
            window.delegate = guardDelegate
        }

        private func windowWillClose() {
            var othersOpen = 0
            if let registered, let group { othersOpen = WindowGroups.shared.remove(registered, from: group) }
            registered = nil
            if othersOpen == 0 { onClose() }
        }
    }
}

/// A window delegate that answers `windowShouldClose` and forwards everything else to the delegate the window had (SwiftUI's). A window
/// holds its delegate weakly: the observer view keeps this alive.
@MainActor
final class CloseGuardDelegate: NSObject, NSWindowDelegate {
    weak var original: (any NSWindowDelegate)?
    private let shouldClose: @MainActor () -> Bool

    init(shouldClose: @escaping @MainActor () -> Bool) { self.shouldClose = shouldClose }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if let original, original.responds(to: #selector(NSWindowDelegate.windowShouldClose(_:))), original.windowShouldClose?(sender) == false {
            return false
        }
        return shouldClose()
    }

    override func responds(to aSelector: Selector!) -> Bool {
        super.responds(to: aSelector) || (original as? NSObject)?.responds(to: aSelector) == true
    }

    override func forwardingTarget(for aSelector: Selector!) -> Any? {
        if let original = original as? NSObject, original.responds(to: aSelector) { return original }
        return super.forwardingTarget(for: aSelector)
    }
}

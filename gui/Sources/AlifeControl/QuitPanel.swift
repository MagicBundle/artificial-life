import AlifeCore
import AppKit

/// The panel shown while the app waits for a world or a run to write its save point before it quits (docs/APP.md §5.4).
///
/// It is an `NSPanel` that the app delegate owns, not SwiftUI window content: it exists when the quit came from closing the
/// last window, and no scene is needed. `worksWhenModal` and the common run loop modes are there because `applicationShouldTerminate`
/// has answered `.terminateLater`, which runs the loop in the modal-panel mode; spike S2 (§5.4) checked that the panel draws,
/// updates and shows its button then.
@MainActor
final class QuitPanel {
    private var panel: NSPanel?
    private let titleLabel = NSTextField(labelWithString: "")
    private let detailLabel = NSTextField(labelWithString: "")
    private let warningLabel = NSTextField(wrappingLabelWithString: "")
    private let quitNowButton = NSButton(title: "Quit now without saving", target: nil, action: nil)
    private let spinner = NSProgressIndicator()
    private var shown: QuitPanelModel?

    /// Runs when the user chooses "Quit now without saving".
    var onQuitNow: () -> Void = {}

    /// Where the panel goes when it first appears: the middle of the screen. The tests give a place that is off every screen, so that a test
    /// run never puts a panel on anyone's display (it is the real panel, drawn and updated all the same).
    private let place: (NSPanel) -> Void
    /// AppKit moves a titled window back onto a screen when it is ordered in; a panel that is meant to be off every screen says it is not.
    private let keepsOnScreen: Bool

    /// A process that is running tests never puts a panel on a display, whatever made the panel (`AppDelegate()`'s default one included): the
    /// review of PR-F saw two floating panels inside the display during parallel runs.
    static let runsUnderTests: Bool = {
        let name = ProcessInfo.processInfo.processName
        return name.contains("swiftpm-testing") || name.contains("xctest") || NSClassFromString("XCTestCase") != nil
    }()

    init(place: @escaping (NSPanel) -> Void = QuitPanel.defaultPlace, keepsOnScreen: Bool = !QuitPanel.runsUnderTests) {
        self.place = place
        self.keepsOnScreen = keepsOnScreen
    }

    /// The middle of the screen; under tests, far off every screen and invisible.
    static let defaultPlace: (NSPanel) -> Void = { panel in
        if runsUnderTests {
            panel.alphaValue = 0
            panel.setFrameOrigin(NSPoint(x: -20_000, y: -20_000))
        } else {
            panel.center()
        }
    }

    private final class Panel: NSPanel {
        var keepsOnScreen = true
        override func constrainFrameRect(_ frameRect: NSRect, to screen: NSScreen?) -> NSRect {
            keepsOnScreen ? super.constrainFrameRect(frameRect, to: screen) : frameRect
        }
    }

    var isVisible: Bool { panel?.isVisible == true }
    /// The text on screen, for the tests and the spike.
    var texts: (title: String, detail: String, warning: String, quitNowHidden: Bool) {
        (titleLabel.stringValue, detailLabel.stringValue, warningLabel.stringValue, quitNowButton.isHidden)
    }
    /// The panel's window, for the tests and the spike.
    var window: NSWindow? { panel }

    /// Shows the panel, or updates it when it is already up. Words that did not change are not set again.
    func show(_ model: QuitPanelModel) {
        let panel = self.panel ?? makePanel()
        self.panel = panel
        var needsKey = false
        if shown != model {
            shown = model
            set(titleLabel, model.title)
            set(detailLabel, model.detail)
            set(warningLabel, model.cost)
            warningLabel.isHidden = model.cost.isEmpty
            let hadButton = !quitNowButton.isHidden
            quitNowButton.isHidden = !model.offersQuitNow
            // The button can be reached from the keyboard too: the panel becomes key when it appears (Space presses it) and
            // Option-Command-Q is its key equivalent.
            needsKey = model.offersQuitNow && !hadButton
            // The cost lines wrap: the panel follows the height they need.
            if let content = panel.contentView {
                content.layoutSubtreeIfNeeded()
                panel.setContentSize(NSSize(width: 420, height: content.fittingSize.height))
            }
        }
        if !panel.isVisible {
            place(panel)
            panel.orderFrontRegardless()
            // Ordering a titled window in puts it at AppKit's default place on the display, whatever `constrainFrameRect` says (measured: the
            // panel placed at -20000 came out at (160, 800)). A panel that is meant to stay off every screen is put back at once, in the same
            // turn of the run loop and invisible (the place of the tests sets alpha 0), so that nothing is ever drawn on the display.
            if !keepsOnScreen { place(panel) }
            spinner.startAnimation(nil)
        }
        if needsKey {
            panel.makeFirstResponder(quitNowButton)
            // (Becoming key brings a titled window onto a screen, whatever `constrainFrameRect` says: seen in the parallel test runs, where the
            // "off every screen" panel of the quit tests was drawn on the display at the moment the Quit now line appeared. A panel that is
            // meant to stay off every screen does not take the key.)
            if keepsOnScreen { panel.makeKey() }
        }
        panel.displayIfNeeded()
    }

    func close() {
        spinner.stopAnimation(nil)
        panel?.orderOut(nil)
        panel = nil
        shown = nil
    }

    private func set(_ label: NSTextField, _ text: String) {
        if label.stringValue != text { label.stringValue = text }
    }

    @objc private func quitNowPressed() { onQuitNow() }

    private func makePanel() -> NSPanel {
        let panel = Panel(contentRect: NSRect(x: 0, y: 0, width: 420, height: 150),
                          styleMask: [.titled, .utilityWindow], backing: .buffered, defer: false)
        panel.keepsOnScreen = keepsOnScreen
        // No fade: a window that fades out while the display sleeps keeps a dispatch thread blocked in `NSAnimation` for ever, and the panel is
        // shown and hidden by every quit that waits (the tests' windows leaked enough of them to starve a whole run).
        panel.animationBehavior = .none
        panel.title = "Quitting"
        panel.isReleasedWhenClosed = false
        panel.hidesOnDeactivate = false
        panel.level = .floating
        panel.worksWhenModal = true
        panel.becomesKeyOnlyIfNeeded = true
        panel.initialFirstResponder = quitNowButton
        panel.collectionBehavior = [.moveToActiveSpace, .fullScreenAuxiliary]

        titleLabel.font = .boldSystemFont(ofSize: NSFont.systemFontSize)
        detailLabel.textColor = .secondaryLabelColor
        warningLabel.textColor = .secondaryLabelColor
        warningLabel.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        warningLabel.isHidden = true
        warningLabel.preferredMaxLayoutWidth = 380
        warningLabel.translatesAutoresizingMaskIntoConstraints = false
        warningLabel.widthAnchor.constraint(equalToConstant: 380).isActive = true
        quitNowButton.target = self
        quitNowButton.action = #selector(quitNowPressed)
        quitNowButton.isHidden = true
        quitNowButton.keyEquivalent = "q"
        quitNowButton.keyEquivalentModifierMask = [.command, .option]
        quitNowButton.toolTip = "Quit now without saving (Option-Command-Q)"
        quitNowButton.setAccessibilityHelp("Ends what is still saving and quits. Shortcut: Option-Command-Q.")
        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.isIndeterminate = true
        spinner.translatesAutoresizingMaskIntoConstraints = false
        spinner.widthAnchor.constraint(equalToConstant: 16).isActive = true
        spinner.heightAnchor.constraint(equalToConstant: 16).isActive = true

        let header = NSStackView(views: [spinner, titleLabel])
        header.spacing = 8
        let stack = NSStackView(views: [header, detailLabel, warningLabel, quitNowButton])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 8
        stack.edgeInsets = NSEdgeInsets(top: 16, left: 20, bottom: 16, right: 20)
        stack.translatesAutoresizingMaskIntoConstraints = false

        let content = NSView()
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            stack.topAnchor.constraint(equalTo: content.topAnchor),
            stack.bottomAnchor.constraint(equalTo: content.bottomAnchor),
            content.widthAnchor.constraint(equalToConstant: 420),
        ])
        panel.contentView = content
        content.layoutSubtreeIfNeeded()
        panel.setContentSize(content.fittingSize)
        return panel
    }

    /// The question before a run is stopped by a quit: true to go on. Modal, and shown before the gate, so that the app has not
    /// begun to stop anything when the user cancels.
    static func askToStopRun(_ run: String) -> Bool {
        // A quit from the Dock with another app in front, or with no window left, must not leave the question behind them.
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = QuitWording.confirmTitle(run: run)
        alert.informativeText = QuitWording.confirmDetail
        alert.alertStyle = .warning
        alert.addButton(withTitle: QuitWording.confirmCancel)  // the default button (Return) and Escape
        let stop = alert.addButton(withTitle: QuitWording.confirmStop)
        stop.hasDestructiveAction = true
        return alert.runModal() == .alertSecondButtonReturn
    }

    /// The question when the window of the `alife` app is closed with a run going (docs/APP.md D3): keep it running, stop it and quit, or
    /// cancel. Return is the first button, Keep Running: it loses nothing and does what closing meant. Escape is Cancel. "Remember my choice"
    /// is the alert's own checkbox. Modal, like the quit's question; shown while the window is still up.
    static func askToCloseWindow(_ question: CloseQuestion) -> (answer: CloseAnswer, remember: Bool) {
        NSApp.activate(ignoringOtherApps: true)
        let alert = closeAlert(question)
        let response = alert.runModal()
        let remember = alert.suppressionButton?.state == .on
        if !question.offersKeepRunning {
            // (Cancel, the default and Escape, then Stop and Quit)
            return response == .alertSecondButtonReturn ? (.stopAndQuit, false) : (.cancel, false)
        }
        switch response {
        case .alertFirstButtonReturn: return (.keepRunning, remember)
        case .alertSecondButtonReturn: return (.stopAndQuit, false)
        default: return (.cancel, false)
        }
    }

    /// The alert of `askToCloseWindow`, made but not run (the tests look at it, and draw it).
    static func closeAlert(_ question: CloseQuestion) -> NSAlert {
        let alert = NSAlert()
        alert.messageText = question.title
        alert.informativeText = question.detail
        alert.alertStyle = .informational
        if question.offersKeepRunning {
            alert.addButton(withTitle: question.keepLabel)  // the default button (Return)
            let stop = alert.addButton(withTitle: question.stopLabel)
            stop.hasDestructiveAction = true
            alert.addButton(withTitle: question.cancelLabel)  // Escape
            alert.showsSuppressionButton = true
            alert.suppressionButton?.title = question.rememberLabel
        } else {
            alert.addButton(withTitle: question.cancelLabel)  // the default button and Escape: nothing changes
            let stop = alert.addButton(withTitle: question.stopLabel)
            stop.hasDestructiveAction = true
        }
        return alert
    }
}

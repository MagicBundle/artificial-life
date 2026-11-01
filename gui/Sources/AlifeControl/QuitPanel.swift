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
            panel.center()
            panel.orderFrontRegardless()
            spinner.startAnimation(nil)
        }
        if needsKey {
            panel.makeFirstResponder(quitNowButton)
            panel.makeKey()
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
        let panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 420, height: 150),
                            styleMask: [.titled, .utilityWindow], backing: .buffered, defer: false)
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
    static func askToStopRun(_ run: String, shell: Bool = false) -> Bool {
        // A quit from the Dock with another app in front, or with no window left, must not leave the question behind them.
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = QuitWording.confirmTitle(run: run)
        alert.informativeText = shell ? QuitWording.confirmDetailShell : QuitWording.confirmDetail
        alert.alertStyle = .warning
        alert.addButton(withTitle: QuitWording.confirmCancel)  // the default button (Return) and Escape
        let stop = alert.addButton(withTitle: QuitWording.confirmStop)
        stop.hasDestructiveAction = true
        return alert.runModal() == .alertSecondButtonReturn
    }
}

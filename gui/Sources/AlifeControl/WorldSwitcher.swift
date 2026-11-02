import AlifeCore
import SwiftUI

/// R5 of docs/APP.md: only one world is played at a time. Asking to play another world, or to make a planet, while a world is open puts
/// one sheet in front of the person ("Stop A to play B?"); on "Stop and Play" the open world is saved first (R2) and what was asked for
/// starts only once its process has gone. The words are `SwitchRequest`'s. Nothing starts without the person pressing the button, and
/// nothing starts while the app is quitting.
@MainActor
final class WorldSwitcher: ObservableObject, Identifiable {
    enum Phase: Equatable {
        /// The sheet asks.
        case asking
        /// The open world was asked to stop; the sheet waits for its save.
        case saving
        /// The save did not end well, or took too long: what happened (the other world was not opened).
        case failed(String?)
    }

    /// The question or the wait on the sheet; nil when no sheet is wanted.
    @Published private(set) var request: SwitchRequest?
    @Published private(set) var phase = Phase.asking
    /// Whole seconds the save has taken, for the sheet.
    @Published private(set) var elapsed = 0

    let play: PlayController
    /// `<alife folder>/runs`, for the size of the save point (the estimate).
    var runsDirectory: () -> URL?
    /// The app is quitting: nothing starts then (the quit's own wait is ending the world).
    var isQuitting: @MainActor () -> Bool = { AppDelegate.current?.terminationGate.isWaiting == true }
    /// The main window is shown: nothing starts without one, by any route (a world opened with nobody there runs unseen).
    var windowShown: @MainActor () -> Bool = { AppDelegate.current?.mainWindowIsShown ?? false }
    /// The longest the open world is waited for; its own ladder ends it sooner (30 s, then 10 s). **Past it the other world is not started**:
    /// the open one is still being written, and the sheet says so.
    var maxWait: Duration = .seconds(60)
    var poll: Duration = .milliseconds(100)
    /// What was asked for was dropped because the window was closed: the sentence for the Worlds page.
    var onDropped: (@MainActor (String) -> Void)?

    private var action: (@MainActor () -> Void)?
    private var waiting: Task<Void, Never>?
    /// The request of the switch under way, kept so that the sheet can come back after it was closed.
    private var current: SwitchRequest?

    init(play: PlayController, runsDirectory: @escaping () -> URL? = { nil }) {
        self.play = play
        self.runsDirectory = runsDirectory
    }

    deinit { waiting?.cancel() }

    /// A switch is under way (the sheet asks or waits): a second request is not taken.
    var isBusy: Bool { current != nil }

    /// `goal` needs the open world to stop. Returns true when a sheet takes the request (`then` runs once the world is saved, if the person
    /// confirms); false when nothing needs to stop (no world is live, or the one open is still starting: the caller goes on, or says why not).
    /// While another switch is under way and its sheet was closed, the sheet comes back: the click is answered, not swallowed.
    @discardableResult
    func ask(_ goal: SwitchRequest.Goal, then action: @escaping @MainActor () -> Void) -> Bool {
        if isBusy {
            showAgain()
            return true
        }
        guard play.phase == .live, let open = play.world, !open.isEmpty else { return false }
        let save = runsDirectory().flatMap { SwitchWording.newestSave(of: open, in: $0) }
        let seconds = SwitchWording.estimateSeconds(newestSaveBytes: save?.bytes, compressed: save?.compressed ?? false)
        let made = SwitchRequest(open: open, goal: goal, estimateSeconds: seconds, hasSave: save != nil)
        request = made
        current = made
        phase = .asking
        elapsed = 0
        self.action = action
        return true
    }

    /// The sheet of a switch under way comes back (after "Close Panel").
    func showAgain() {
        if request == nil, let current { request = current }
    }

    /// "Keep Playing": nothing changes.
    func keepPlaying() {
        guard phase == .asking else { return }
        request = nil
        current = nil
        action = nil
    }

    /// "Stop and Play": the open world is stopped through its ladder (it saves), and `action` runs when it has gone, and has saved.
    func confirm() {
        guard let request, phase == .asking else { return }
        phase = .saving
        elapsed = 0
        let open = request.open
        let endsBefore = play.lastRetiredEnd?.serial ?? 0
        play.stop()
        waiting = Task { @MainActor [weak self] in
            let clock = ContinuousClock()
            let began = clock.now
            while let self, self.play.retiredWorlds.contains(where: { NameFolding.same($0, open) }), clock.now - began < self.maxWait,
                  !Task.isCancelled {
                try? await Task.sleep(for: self.poll)
                let seconds = Int((clock.now - began).components.seconds)
                if seconds != self.elapsed { self.elapsed = seconds }
            }
            guard let self, !Task.isCancelled else { return }
            let go = self.action
            self.action = nil
            self.waiting = nil
            // Still being written after the longest wait: the other world does not start over it.
            if self.play.retiredWorlds.contains(where: { NameFolding.same($0, open) }) {
                self.fail(request, "\(request.open) is taking longer than expected to save")
                return
            }
            // It ended, but not well.
            if let end = self.play.lastRetiredEnd, end.serial > endsBefore, NameFolding.same(end.world, open), !end.saved {
                self.fail(request, end.reason)
                return
            }
            self.request = nil
            self.current = nil
            self.phase = .asking
            if self.isQuitting() { return }
            // Nobody is there to see it: it is not opened. The open world is saved and closed, and the Worlds page says so.
            guard self.windowShown() else {
                self.onDropped?(request.droppedNote)
                return
            }
            go?()
        }
    }

    private func fail(_ request: SwitchRequest, _ reason: String?) {
        self.request = request
        current = request
        phase = .failed(reason)
    }

    /// The sheet went away by another way than its buttons (Escape, or the quit ending sheets). Asking: it is "Keep Playing". Saving: the
    /// sheet is only closed, the world is already stopping and what was asked for follows (it comes back if the save fails). A failure is read.
    func sheetDismissed() {
        switch phase {
        case .asking: keepPlaying()
        case .saving: request = nil
        case .failed:
            request = nil
            current = nil
            phase = .asking
        }
    }
}

extension WorldSwitcher {
    /// The switcher of the alife folder a `RunController` is set to (its `runs/` directory is where the save point is measured).
    static func forApp(controller: RunController, play: PlayController) -> WorldSwitcher {
        WorldSwitcher(play: play, runsDirectory: { controller.repoReady ? controller.repoURL.appendingPathComponent("runs") : nil })
    }
}

extension SwitchRequest: Identifiable {
    public var id: String { title }
}

/// The sheet: what is asked and what happens, the safe button first and the default. While the open world saves it shows the wait.
struct SwitchSheet: View {
    @ObservedObject var switcher: WorldSwitcher
    let request: SwitchRequest

    private var symbol: String {
        switch switcher.phase {
        case .asking: return "arrow.left.arrow.right.circle"
        case .saving: return "externaldrive.badge.checkmark"
        case .failed: return "exclamationmark.triangle"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: symbol)
                    .font(.system(size: 34)).foregroundStyle(switcher.phase.isFailure ? Color.orange : Color.accentColor)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 8) {
                    switch switcher.phase {
                    case .asking:
                        Text(request.title).font(.headline)
                        Text(request.reason)
                        Text(request.detail).foregroundStyle(.secondary)
                    case .saving:
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            Text(request.savingTitle).font(.headline)
                        }
                        Text(request.savingDetail(elapsed: switcher.elapsed)).foregroundStyle(.secondary)
                    case .failed(let reason):
                        Text(request.failedTitle).font(.headline)
                        Text(request.failedDetail(reason)).foregroundStyle(.secondary)
                    }
                }
                .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                switch switcher.phase {
                case .asking:
                    Button(request.confirmLabel) { switcher.confirm() }
                    Button(request.keepLabel) { switcher.keepPlaying() }
                        .keyboardShortcut(.defaultAction)
                case .saving:
                    Button(request.closePanelLabel) { switcher.sheetDismissed() }
                        .help("Close this panel. The save goes on, and the other world opens when it is done.")
                case .failed:
                    Button("OK") { switcher.sheetDismissed() }
                        .keyboardShortcut(.defaultAction)
                }
            }
        }
        .padding(20)
        .frame(width: 470)
        .onExitCommand { switcher.sheetDismissed() }
    }
}

extension WorldSwitcher.Phase {
    var isFailure: Bool {
        if case .failed = self { return true }
        return false
    }
}

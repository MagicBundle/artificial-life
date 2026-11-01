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
    /// The longest the open world is waited for; its own ladder ends it sooner (30 s, then 10 s).
    var maxWait: Duration = .seconds(60)
    var poll: Duration = .milliseconds(100)

    private var action: (@MainActor () -> Void)?
    private var waiting: Task<Void, Never>?

    init(play: PlayController, runsDirectory: @escaping () -> URL? = { nil }) {
        self.play = play
        self.runsDirectory = runsDirectory
    }

    deinit { waiting?.cancel() }

    /// A switch is under way (the sheet asks or waits): a second request is not taken.
    var isBusy: Bool { request != nil || waiting != nil }

    /// `goal` needs the open world to stop. Returns true when a sheet takes the request (`then` runs once the world is saved, if the person
    /// confirms); false when nothing needs to stop (no world is live, or the one open is still starting: the caller goes on, or says why not).
    @discardableResult
    func ask(_ goal: SwitchRequest.Goal, then action: @escaping @MainActor () -> Void) -> Bool {
        guard play.phase == .live, let open = play.world, !open.isEmpty else { return false }
        guard !isBusy else { return true }
        let save = runsDirectory().flatMap { SwitchWording.newestSave(of: open, in: $0) }
        let seconds = SwitchWording.estimateSeconds(newestSaveBytes: save?.bytes, compressed: save?.compressed ?? false)
        request = SwitchRequest(open: open, goal: goal, estimateSeconds: seconds)
        phase = .asking
        elapsed = 0
        self.action = action
        return true
    }

    /// "Keep Playing": nothing changes.
    func keepPlaying() {
        guard phase == .asking else { return }
        request = nil
        action = nil
    }

    /// "Stop and Play": the open world is stopped through its ladder (it saves), and `action` runs when it has gone.
    func confirm() {
        guard let request, phase == .asking else { return }
        phase = .saving
        elapsed = 0
        let open = request.open
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
            self.request = nil
            self.phase = .asking
            if !self.isQuitting() { go?() }
        }
    }

    /// The sheet went away by another way than its buttons (Escape, or the quit ending sheets). Asking: it is "Keep Playing". Saving: the
    /// sheet is only hidden, the world is already stopping and what was asked for follows.
    func sheetDismissed() {
        switch phase {
        case .asking: keepPlaying()
        case .saving: request = nil
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

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: switcher.phase == .asking ? "arrow.left.arrow.right.circle" : "externaldrive.badge.checkmark")
                    .font(.system(size: 34)).foregroundStyle(.tint)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 8) {
                    if switcher.phase == .asking {
                        Text(request.title).font(.headline)
                        Text(request.reason)
                        Text(request.detail).foregroundStyle(.secondary)
                    } else {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            Text(request.savingTitle).font(.headline)
                        }
                        Text(request.savingDetail(elapsed: switcher.elapsed)).foregroundStyle(.secondary)
                    }
                }
                .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                if switcher.phase == .asking {
                    Button(request.confirmLabel) { switcher.confirm() }
                    Button(request.keepLabel) { switcher.keepPlaying() }
                        .keyboardShortcut(.defaultAction)
                } else {
                    Button("Hide") { switcher.sheetDismissed() }
                        .help("Close this window. The world still opens when the other one is saved.")
                }
            }
        }
        .padding(20)
        .frame(width: 470)
        .onExitCommand { switcher.sheetDismissed() }
    }
}

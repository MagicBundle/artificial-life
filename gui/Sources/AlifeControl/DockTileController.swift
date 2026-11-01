import AlifeCore
import AppKit
import Combine

/// What the Dock icon can be made to show. The app's is `SystemDockTile`; the tests record instead, so that a test run never touches the
/// Dock.
@MainActor
protocol DockTileSurface: AnyObject {
    /// The icon with a bar: `fraction` of the run, nil when it cannot be told; `text` is "62%" or what is going on.
    func showProgress(fraction: Double?, text: String)
    func clearProgress()
    func setBadge(_ text: String?)
    /// One bounce of the icon, only while the app is not in front (the system ignores it otherwise).
    func bounceIfInactive()
}

/// Shows a run's progress on the Dock icon while it goes, and tells the person when it ends (docs/APP.md D3 = C): the bar goes, the icon
/// bounces once if the app is not in front, and a badge ("Done", "Failed") stays until the window is shown again. A run that the person
/// stopped themselves is not news. It never opens a window and never asks anything.
@MainActor
final class DockTileController: ObservableObject {
    let controller: RunController
    let surface: any DockTileSurface
    /// The step the run in progress is to reach, nil when unknown. The app gives the pipeline's, else the run's own `config.json`.
    var target: () -> Int?
    /// The main window is on screen (a badge is for a window that is not).
    var windowShown: () -> Bool = { AppDelegate.current?.mainWindowIsShown ?? false }
    /// A run is going, and the step the newest numbers read (the tests give theirs; the numbers are read from files every two seconds).
    lazy var isRunning: () -> Bool = { [unowned self] in controller.hasRunJob }
    lazy var step: () -> Int? = { [unowned self] in controller.metrics?.step }

    private var cancellables: Set<AnyCancellable> = []
    private var badgeShown = false
    private var lastExitSerial = 0

    init(controller: RunController, surface: any DockTileSurface, target: @escaping () -> Int? = { nil }) {
        self.controller = controller
        self.surface = surface
        self.target = target
        lastExitSerial = controller.lastRunExit?.serial ?? 0
        // (a published value is announced before it is stored: the look at the run waits for the next turn of the main actor)
        controller.$job.removeDuplicates().sink { [weak self] _ in MainActor.assumeIsolated { self?.scheduleRefresh() } }.store(in: &cancellables)
        controller.$metrics.sink { [weak self] _ in MainActor.assumeIsolated { self?.scheduleRefresh() } }.store(in: &cancellables)
        controller.$lastRunExit.dropFirst().sink { [weak self] exit in
            MainActor.assumeIsolated { if let exit { self?.ended(exit) } }
        }.store(in: &cancellables)
        let center = NotificationCenter.default
        for name in [NSApplication.didBecomeActiveNotification, NSWindow.didBecomeKeyNotification] {
            center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.windowMayHaveComeBack() }
            }
        }
    }

    private var refreshQueued = false
    private var barShown = false

    private func scheduleRefresh() {
        guard !refreshQueued else { return }
        refreshQueued = true
        Task { @MainActor [weak self] in
            self?.refreshQueued = false
            self?.refresh()
        }
    }

    /// The bar follows the run; a run that is not going leaves the icon alone.
    func refresh() {
        guard isRunning() else {
            clearBar()
            return
        }
        // A new run takes the old run's badge away.
        clearBadge()
        let fraction = DockProgress.fraction(step: step(), target: target())
        barShown = true
        surface.showProgress(fraction: fraction, text: fraction.map(DockProgress.percentText) ?? "Growing")
    }

    private func clearBar() {
        guard barShown else { return }
        barShown = false
        surface.clearProgress()
    }

    /// A run ended (exposed to the tests, which feed it exits).
    func ended(_ exit: RunExit) {
        guard exit.serial > lastExitSerial else { return }
        lastExitSerial = exit.serial
        clearBar()
        guard DockProgress.endIsNews(stopRequested: exit.stopRequested && !exit.forced, byTimer: exit.byTimer) else { return }
        // The badge is for a person who is not looking at the window; one who is does not need it. The bounce is ignored by the system while
        // the app is in front.
        if !windowShown() {
            surface.setBadge(DockProgress.endBadge(status: exit.status, signalled: exit.signalled))
            badgeShown = true
        }
        surface.bounceIfInactive()
    }

    private func windowMayHaveComeBack() {
        if badgeShown, windowShown() { clearBadge() }
    }

    private func clearBadge() {
        guard badgeShown else { return }
        badgeShown = false
        surface.setBadge(nil)
    }
}

/// The real Dock tile.
@MainActor
final class SystemDockTile: DockTileSurface {
    private let view = DockProgressView(frame: NSRect(x: 0, y: 0, width: 128, height: 128))

    func showProgress(fraction: Double?, text: String) {
        view.fraction = fraction
        view.text = text
        let tile = NSApp.dockTile
        if tile.contentView !== view { tile.contentView = view }
        tile.display()
    }

    func clearProgress() {
        let tile = NSApp.dockTile
        guard tile.contentView != nil else { return }
        tile.contentView = nil
        tile.display()
    }

    func setBadge(_ text: String?) { NSApp.dockTile.badgeLabel = text }

    func bounceIfInactive() {
        if !NSApp.isActive { NSApp.requestUserAttention(.informationalRequest) }
    }
}

/// The icon with a bar along its foot and the percentage in it.
final class DockProgressView: NSView {
    var fraction: Double? { didSet { needsDisplay = true } }
    var text = "" { didSet { needsDisplay = true } }

    override func draw(_ dirtyRect: NSRect) {
        NSApplication.shared.applicationIconImage?.draw(in: bounds)
        let track = NSRect(x: 8, y: 6, width: bounds.width - 16, height: 22)
        let shape = NSBezierPath(roundedRect: track, xRadius: 11, yRadius: 11)
        NSColor.black.withAlphaComponent(0.55).setFill()
        shape.fill()
        let inner = track.insetBy(dx: 3, dy: 3)
        if let fraction, fraction > 0 {
            NSGraphicsContext.saveGraphicsState()
            NSBezierPath(roundedRect: inner, xRadius: 8, yRadius: 8).addClip()
            NSColor.controlAccentColor.setFill()
            NSRect(x: inner.minX, y: inner.minY, width: inner.width * fraction, height: inner.height).fill()
            NSGraphicsContext.restoreGraphicsState()
        }
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 12, weight: .bold), .foregroundColor: NSColor.white,
        ]
        let size = (text as NSString).size(withAttributes: attributes)
        (text as NSString).draw(at: NSPoint(x: track.midX - size.width / 2, y: track.midY - size.height / 2), withAttributes: attributes)
    }
}

extension DockTileController {
    /// The run's target: the pipeline's when it grows the run, else the `run.steps` its own `config.json` was written with.
    static func configTarget(run: String, repo: URL) -> Int? {
        let url = repo.appendingPathComponent("runs/\(run)/config.json")
        guard let data = try? Data(contentsOf: url),
              let config = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let steps = (config["run"] as? [String: Any])?["steps"] as? Int, steps >= 1 else { return nil }
        return steps
    }
}

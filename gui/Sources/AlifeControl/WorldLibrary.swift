import AlifeCore
import AppKit
import Foundation

/// The world library of the `alife` window (docs/APP.md §5.3, DESIGN §33.4.12): what is under `runs/`, as the sidebar and the Worlds page
/// read it. The scan runs on a serial utility queue and never on the main thread; every look at an entry runs on a lane with a deadline
/// (`LibraryScanner`), so a volume that hangs costs the deadline once and holds nothing else up. The scan is cached per directory and asked
/// for only when something may have changed: the app comes to the front, the folder changes (a watch on `runs/`, which is re-opened when
/// `runs/` is replaced, removed or made later), a world is opened, closed or finished saving, a run this app grows moves, the player asks.
/// What no file shows, a world held or written by another process, is probed on a timer that is cheap and runs only while the app is active.
@MainActor
final class WorldLibrary: ObservableObject {
    /// The library as the pages read it, merged with what this app is doing.
    @Published private(set) var state = LibraryState()
    /// False until the first scan has finished: the page then says it is looking instead of saying it is empty.
    @Published private(set) var hasScanned = false
    @Published private(set) var isScanning = false
    /// The Worlds page's filter and search; the sidebar sets them too.
    @Published var filter: LibraryFilter = .all
    @Published var query = ""
    /// The world the player chose (a card, a sidebar row); the page marks it. Meaningful on the Worlds page only (`revealFromMenu`).
    @Published var selected: String?
    /// The sidebar row the player chose last (so that the arrow keys go on from it, whatever the row opened): see `SidebarItem.current`.
    @Published var sidebarAnchor: SidebarItem?
    /// Asks the Worlds page to scroll to a world (a choice made in the sidebar; a click on a card that is already in view scrolls nothing).
    @Published private(set) var reveal: RevealRequest?
    struct RevealRequest: Equatable {
        var name: String
        var serial: Int
    }
    private var revealSerial = 0
    /// Set when the chosen world was gone at the next look.
    @Published var vanished: String?
    @Published private(set) var lastScan: LibraryScanStats?
    /// Counts the times the folder was scanned (the tests wait on it).
    @Published private(set) var scans = 0

    private let runsDirectory: () -> URL?
    private let scanner: LibraryScanner
    private let queue = DispatchQueue(label: "alife.library.scan", qos: .utility)
    private let lockHeld: @Sendable (URL) -> Bool
    private let watchesFolder: Bool
    private let probeInterval: Duration
    private let requiresActiveApp: Bool
    /// How a folder is shown in Finder (the tests give theirs).
    /// The tests turn the watch's events off to show that the probe notices a replaced `runs/` by itself (an event can be missed).
    var watchEventsEnabled = true
    var revealURL: (URL) -> Void = { NSWorkspace.shared.activateFileViewerSelecting([$0]) }
    private var snapshot = LibrarySnapshot()
    private var live = LiveFacts()
    private var notes: [String: String] = [:]
    private var noteTimes: [String: Date] = [:]
    private var scanning = false
    private var rescanWanted = false
    private var arming = false
    private var watchedPath: String?
    private var watchedInode: UInt64?
    private var watcher: DispatchSourceFileSystemObject?
    private var debounce: Task<Void, Never>?
    private var timer: Task<Void, Never>?
    private nonisolated(unsafe) var activation: NSObjectProtocol?
    private nonisolated(unsafe) var deactivation: NSObjectProtocol?
    private var growingRescans = 0

    /// Seconds between two probes of the locks while the app is active, and the number of probes between two scans while this app grows
    /// a run (its save points then appear without any other sign).
    static let defaultProbeInterval: Duration = .seconds(4)
    static let probesPerGrowingScan = 5

    /// `runsDirectory` is `<alife folder>/runs`, nil without a usable folder. `lockHeld` is how a world's `play.lock` is probed, and
    /// `watchesFolder`, `probeInterval` and `requiresActiveApp` can be changed by the tests.
    init(runsDirectory: @escaping () -> URL?, scanner: LibraryScanner = LibraryScanner(), observesApp: Bool = true,
         watchesFolder: Bool = true, probeInterval: Duration = WorldLibrary.defaultProbeInterval, requiresActiveApp: Bool = true,
         lockHeld: @escaping @Sendable (URL) -> Bool = { PlayLockProbe.isHeld(directory: $0) }) {
        self.runsDirectory = runsDirectory
        self.scanner = scanner
        self.lockHeld = lockHeld
        self.watchesFolder = watchesFolder
        self.probeInterval = probeInterval
        self.requiresActiveApp = requiresActiveApp
        if observesApp {
            activation = NotificationCenter.default.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.refresh()
                    self?.startTimer()
                }
            }
            deactivation = NotificationCenter.default.addObserver(forName: NSApplication.didResignActiveNotification, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.stopTimer() }
            }
        }
    }

    deinit {
        if let activation { NotificationCenter.default.removeObserver(activation) }
        if let deactivation { NotificationCenter.default.removeObserver(deactivation) }
        watcher?.cancel()
        timer?.cancel()
    }

    /// The library of the alife folder a `RunController` is set to.
    static func forApp(run: RunController) -> WorldLibrary {
        WorldLibrary(runsDirectory: { run.repoReady ? run.repoURL.appendingPathComponent("runs") : nil })
    }

    // MARK: Asking

    /// The worlds this app is starting, playing or saving: their lock is not probed (`LibraryScanner.probe`).
    private var skipped: Set<String> { live.saving.union(live.openWorld.map { [$0] } ?? []) }

    /// Scans `runs/` again, off the main thread. A request while a scan runs makes one more scan after it.
    func refresh() {
        guard let runs = runsDirectory() else {
            stopWatching()
            snapshot = LibrarySnapshot()
            hasScanned = true
            recompute()
            return
        }
        if scanning {
            rescanWanted = true
            return
        }
        scanning = true
        isScanning = true
        armWatch(runs)
        let scanner = scanner
        let lockHeld = lockHeld
        let queue = queue
        let skip = skipped
        Task {
            let result = await withCheckedContinuation { (continuation: CheckedContinuation<(LibrarySnapshot, LibraryScanStats), Never>) in
                queue.async {
                    let scanned = scanner.scan(runsDirectory: runs)
                    let probed = scanner.probe(scanned, runsDirectory: runs, skipping: skip, lockHeld: lockHeld)
                    continuation.resume(returning: (probed, scanner.lastStats))
                }
            }
            finishScan(runs: runs, snapshot: result.0, stats: result.1)
        }
    }

    private func finishScan(runs: URL, snapshot scanned: LibrarySnapshot, stats: LibraryScanStats) {
        scanning = false
        isScanning = false
        scans += 1
        lastScan = stats
        if runsDirectory() != runs {
            // The folder was changed while the scan ran: what it found is another folder's.
            refresh()
            return
        }
        hasScanned = true
        if scanned != snapshot { snapshot = scanned }
        recompute()
        armWatch(runs)
        if rescanWanted {
            rescanWanted = false
            refresh()
        }
    }

    /// Looks at the locks and the running worlds again, without reading any file again.
    private func probe() {
        guard let runs = runsDirectory(), !scanning, hasScanned else { return }
        let scanner = scanner
        let lockHeld = lockHeld
        let queue = queue
        let current = snapshot
        let skip = skipped
        Task {
            let probed = await withCheckedContinuation { (continuation: CheckedContinuation<LibrarySnapshot, Never>) in
                queue.async { continuation.resume(returning: scanner.probe(current, runsDirectory: runs, skipping: skip, lockHeld: lockHeld)) }
            }
            // A scan that finished meanwhile is newer than this probe.
            guard !scanning, snapshot.records == current.records else { return }
            if watcher != nil, let watched = watchedInode, probed.runsIdentity != watched {
                // `runs/` was replaced or removed: the watch is on a folder that is no longer there.
                stopWatching()
                refresh()
                return
            }
            guard probed != snapshot else { return }
            snapshot = probed
            recompute()
        }
    }

    // MARK: Timer and folder watch

    /// Starts the probe timer when the app is active; the notification of the app coming to the front does it too.
    func startTimer() {
        guard timer == nil, !requiresActiveApp || (NSApp?.isActive ?? true) else { return }
        let interval = probeInterval
        timer = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: interval)
                guard !Task.isCancelled, let self else { return }
                self.tick()
            }
        }
    }

    func stopTimer() {
        timer?.cancel()
        timer = nil
    }

    private func tick() {
        // No watch: `runs/` was not there when it was last looked for, or was removed. Looking again is a scan of nothing until it is back.
        if watchesFolder, watcher == nil, !arming {
            refresh()
            return
        }
        if live.growing != nil {
            growingRescans += 1
            if growingRescans >= Self.probesPerGrowingScan {
                growingRescans = 0
                refresh()
                return
            }
        }
        probe()
    }

    /// Watches `runs/` itself: a world made, removed or renamed there is seen at once (a save point written inside a world is not: the
    /// timer, the activation and the Worlds page appearing cover that). The `open` runs on the scan queue, never on the main thread; the
    /// watch is re-armed when `runs/` is replaced, removed or made later.
    private func armWatch(_ runs: URL) {
        guard watchesFolder, watcher == nil, !arming else { return }
        arming = true
        // Queued at once, before the scan that `refresh` queues next: the folder is opened before it is listed, so a world made after the
        // listing is seen by the watch. (A change between the `open` and the moment the source is resumed is covered by the scan that follows
        // the arming.)
        queue.async { [weak self] in
            let fd = open(runs.path, O_EVTONLY)
            var info = stat()
            if fd >= 0 { fstat(fd, &info) }
            let inode = UInt64(info.st_ino)
            DispatchQueue.main.async {
                MainActor.assumeIsolated { self?.finishArming(runs, fd: fd, inode: inode) }
            }
        }
    }

    private func finishArming(_ runs: URL, fd: Int32, inode: UInt64) {
        arming = false
        guard fd >= 0 else { return }
        guard watcher == nil, runsDirectory() == runs else {
            close(fd)
            return
        }
        let source = DispatchSource.makeFileSystemObjectSource(fileDescriptor: fd, eventMask: [.write, .rename, .delete, .link], queue: .main)
        source.setEventHandler { [weak self] in
            MainActor.assumeIsolated {
                guard let self, self.watchEventsEnabled else { return }
                self.watchFired()
            }
        }
        source.setCancelHandler { close(fd) }
        source.resume()
        watcher = source
        watchedPath = runs.path
        watchedInode = inode
        scheduleScan()
    }

    private func stopWatching() {
        watcher?.cancel()
        watcher = nil
        watchedPath = nil
        watchedInode = nil
    }

    private func watchFired() {
        if let data = watcher?.data, !data.isDisjoint(with: [.delete, .rename]) {
            // `runs/` itself was moved or removed: this watch is on the old one. `refresh` opens a new one on whatever is at the path.
            stopWatching()
        }
        scheduleScan()
    }

    /// A burst of changes (a world being built writes many files) is one scan, a moment after the last.
    private func scheduleScan() {
        debounce?.cancel()
        debounce = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(400))
            guard !Task.isCancelled else { return }
            self?.refresh()
        }
    }

    // MARK: Telling it what this app is doing

    func update(live facts: LiveFacts) {
        guard facts != live else { return }
        let before = live
        live = facts
        recompute()
        // A world opened, closed or finished saving, and a run that started or ended, have new save points, a new date and no lock to show.
        if before.growing != facts.growing || before.openWorld != facts.openWorld || before.saving != facts.saving { refresh() }
    }

    /// What this session knows about a build that stopped (shown with its reason on the Unfinished card).
    func note(_ text: String, forBuild name: String) {
        guard notes[name] != text else { return }
        notes[name] = text
        noteTimes[name] = Date()
        recompute()
    }

    private func recompute() {
        let resolved = LibraryState.resolve(snapshot, live: live, notes: notes, noteTimes: noteTimes)
        if resolved != state { state = resolved }
        if let chosen = selected, state.row(named: chosen) == nil, !state.unfinished.contains(where: { $0.name == chosen }) {
            // The chosen world is no longer listed. It is *gone* only when the folder is gone; a folder that is still there but is no longer a
            // world (its save points were pruned, a build was restarted) is just not chosen any more.
            if hasScanned, !snapshot.names.contains(chosen) { vanished = chosen }
            selected = nil
        }
    }

    // MARK: Choosing

    /// Chooses a world: the filter and the search are cleared if they would hide it. `scroll` asks the page to bring its card into view.
    func select(_ name: String, scroll: Bool = true) {
        if let row = state.row(named: name) {
            if !state.visible(filter: filter, query: query).contains(where: { $0.name == row.name }) {
                filter = .all
                query = ""
            }
        } else if state.unfinished.contains(where: { $0.name == name }) {
            filter = .unfinished
            query = ""
        }
        vanished = nil
        selected = name
        if scroll {
            revealSerial += 1
            reveal = RevealRequest(name: name, serial: revealSerial)
        }
    }

    /// Whether the world's folder is still there, looked at off the main thread with a deadline (a click must not wait for a volume that
    /// hangs: no answer is `.unknown`, and the caller goes on).
    func checkThere(_ name: String) async -> Presence {
        guard let runs = runsDirectory() else { return .gone }
        let scanner = scanner
        let url = runs.appendingPathComponent(name)
        return await withCheckedContinuation { (continuation: CheckedContinuation<Presence, Never>) in
            DispatchQueue.global(qos: .userInitiated).async { continuation.resume(returning: scanner.presence(of: url)) }
        }
    }

    /// The player pressed a button of a world that is gone: say so, and look again.
    func noteGone(_ name: String) {
        vanished = name
        if selected == name { selected = nil }
        refresh()
    }

    /// Shows the world's folder in Finder; false when there is none to show (it is told, and no other folder is shown in its place).
    @discardableResult
    func reveal(_ name: String) async -> Bool {
        guard let runs = runsDirectory() else { return false }
        if await checkThere(name) == .gone {
            noteGone(name)
            return false
        }
        revealURL(runs.appendingPathComponent(name, isDirectory: true))
        return true
    }

    /// File > Reveal in Finder: the chosen world while the Worlds page is the one in view (a world that is gone is told, and nothing else is
    /// shown instead), else `fallback` (the Lab's run, or the alife folder).
    func revealFromMenu(page: ShellPage, fallback: () -> Void) async {
        if page == .worlds, let name = selected {
            await reveal(name)
            return
        }
        fallback()
    }

    /// Why Keep growing must not continue this directory; nil when it may. A played world is continued by playing it; a directory that this
    /// window has open, saves, or that another process holds or writes has a writer already (`alife resume` takes no `play.lock`).
    func reasonNotToResume(_ name: String) -> String? {
        if let row = state.row(named: name) {
            if row.kind == .played {
                return "'runs/\(name)' is a played world. Play it from Worlds, or make a planet from it; Keep growing continues research worlds."
            }
            switch row.state {
            case .live, .paused, .starting: return "'runs/\(name)' is open in this window."
            case .saving: return "'runs/\(name)' is still being saved."
            case .inUseElsewhere: return "'runs/\(name)' is in use by another app or process. Wait until it stops."
            case .growing, .saved, .unfinished: return nil
            }
        }
        if state.unfinished.contains(where: { $0.name == name }) { return "'runs/\(name)' is an unfinished build, not a run." }
        return nil
    }

    // MARK: Words

    /// "3 worlds are on a drive that is not connected." / "...on the drive “Archive”, which is not connected."
    var offlineSentence: String? {
        let n = state.offline.count
        guard n > 0 else { return nil }
        let count = n == 1 ? "1 world is" : "\(n) worlds are"
        if state.offlineVolumes.count == 1, let drive = state.offlineVolumes.first {
            return "\(count) on the drive \u{201C}\(drive)\u{201D}, which is not connected. Connect it to see \(n == 1 ? "it" : "them") here."
        }
        return "\(count) on a drive that is not connected. Connect it to see \(n == 1 ? "it" : "them") here."
    }

    /// The other things the list leaves out, each in its own words (a broken link is not a drive that is away).
    var problemSentences: [String] {
        var out: [String] = []
        func count(_ p: EntryProblem) -> Int { state.problems.values.filter { $0 == p }.count }
        let broken = count(.brokenLink), loops = count(.loop), denied = count(.noPermission), other = count(.unreadable)
        if broken > 0 { out.append("\(broken) \(broken == 1 ? "link in runs/ leads" : "links in runs/ lead") to a place that is not there.") }
        if loops > 0 { out.append("\(loops) \(loops == 1 ? "link in runs/ leads" : "links in runs/ lead") back to itself, so it cannot be read.") }
        if denied > 0 {
            out.append("alife is not allowed to read \(denied) \(denied == 1 ? "entry" : "entries") in runs/. Allow it in System Settings, Privacy & Security.")
        }
        if other > 0 { out.append("\(other) \(other == 1 ? "entry" : "entries") in runs/ could not be read (the disk reported an error).") }
        let slow = state.slow.count
        if slow > 0 {
            out.append("\(slow) \(slow == 1 ? "entry" : "entries") in runs/ did not answer in time (a drive that is slow or not responding), so \(slow == 1 ? "it" : "they") may be missing or out of date.")
        }
        if state.listingSlow { out.append("runs/ did not answer in time. This is the last list.") }
        return out
    }

    /// "4 folders in runs/ have no save point, so they are not worlds."
    var othersSentence: String? {
        let n = state.others.count
        guard n > 0 else { return nil }
        return n == 1 ? "1 folder in runs/ has no save point, so it is not a world and is not listed."
            : "\(n) folders in runs/ have no save point, so they are not worlds and are not listed."
    }
}

extension SidebarItem {
    /// The row the sidebar shows as chosen. The row the player chose last stays chosen while the page still shows what it opened (so that the
    /// arrow keys go on from it: "All N worlds..." and "Unfinished" open a list and used to hand the selection back to the Worlds row), else
    /// the world chosen on the Worlds page, else the page.
    @MainActor
    static func current(page: ShellPage, library: WorldLibrary) -> SidebarItem {
        if page == .worlds {
            if let anchor = library.sidebarAnchor {
                switch anchor {
                case .allWorlds where library.filter == .all && library.selected == nil && library.query.isEmpty: return anchor
                case .unfinished where library.filter == .unfinished && library.selected == nil: return anchor
                case .world(let name) where library.selected == name: return anchor
                default: break
                }
            }
            if let name = library.selected { return .world(name) }
        }
        return .page(page)
    }
}

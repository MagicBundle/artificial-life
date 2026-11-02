import Darwin
import Foundation

// The scan behind the world library (docs/APP.md §5.3, DESIGN §33.4.12). It is the launcher's `PlayCatalogScanner` made fit for a long
// list: pure and synchronous (the app runs it off the main thread), cached per directory so that a second scan costs a few `stat` calls
// instead of reading every JSON file again, aware of entries that are links to a drive that is not connected, and with the one fact
// that changes without a file changing (a held `play.lock`) kept out of the cache and probed on its own.

/// `runs/NAME:STEP` (or `NAME:STEP`, or a path ending so), the way `play.json` and `meta.json` name the run a world was made from.
package struct SourceRef: Equatable, Sendable {
    package var name: String
    package var step: Int?

    package init(name: String, step: Int?) {
        self.name = name
        self.step = step
    }

    package static func parse(_ text: String?) -> SourceRef? {
        guard var rest = text?.trimmingCharacters(in: .whitespaces), !rest.isEmpty else { return nil }
        var step: Int?
        if let colon = rest.lastIndex(of: ":"), let n = Int(rest[rest.index(after: colon)...]) {
            step = n
            rest = String(rest[..<colon])
        }
        while rest.hasSuffix("/") { rest.removeLast() }
        let name = rest.split(separator: "/").last.map(String.init) ?? ""
        return name.isEmpty || rest == "runs" ? nil : SourceRef(name: name, step: step)
    }
}

/// What the library knows about one directory of `runs/`, read once and kept until one of the stamps of `LibraryScanner.Stamps` moves.
/// Nothing in it depends on a running process.
package struct LibraryRecord: Equatable, Sendable, Identifiable {
    package var name: String
    /// A valid `play.json`: the directory is a world that `alife play` can continue (a *Played* world).
    package var play: PlayWorldFile?
    /// The steps of its save points, ascending.
    package var steps: [Int]
    /// `meta.json`'s `status` (`running`, `done`, `extinct`, `violation`, `interrupted`).
    package var status: String?
    /// `meta.json`'s `parent` (`runs/NAME:STEP`): the run it was copied from, when it was made from one.
    package var parent: SourceRef?
    /// `world.dims`, `world.geometry` (`"box"` when the config has a `world` section without one), `world.size`, `world.radius` of a sphere,
    /// the shape of a sphere and `weather.enabled` of `config.json`; nil when the file is unreadable.
    package var dims: Int?
    package var geometry: String?
    package var worldSize: Double?
    package var planetRadius: Double?
    package var planetShape: PlanetShape?
    package var weatherOn: Bool
    /// Whether `events/` has a chunk (the exact houses of a planet are rebuilt from them).
    package var hasEvents: Bool
    /// The directory's own modification date, which `PlayCatalog` has always used.
    package var directoryModified: Date
    /// When the world was last saved: the newest of the directory's date and its `checkpoints/`'s (a new save point does not touch the
    /// directory itself).
    package var lastSaved: Date
    /// The directory holds something besides `play.lock` and `.DS_Store`.
    package var hasContent: Bool
    /// The directory is what `alife play` leaves when a build stops (`PlayCatalogScanner.isUnfinishedBuild`), *if* no process holds
    /// its lock: with the lock held it is a build in progress.
    package var looksUnfinished: Bool
    /// `meta.json`'s `play_build` mark (`planet` or `preset`), nil for anything `alife play` did not begin.
    package var buildMark: String?
    /// `run.metrics_every` of `config.json` (100 when unreadable): how many steps lie between two rows of `metrics.jsonl`, which sets how long
    /// a world may be quiet and still be running (`LibraryScanner.signWindow`).
    package var metricsEvery: Int = 100

    package var id: String { name }
    package var isPlayed: Bool { play != nil }
}

/// Why an entry of `runs/` is not listed, in the engine's and the system's terms; the page says each in its own words.
package enum EntryProblem: Equatable, Sendable {
    /// A link whose target is under `/Volumes/NAME` and is not there (ENOENT): a drive that is not connected.
    case driveNotConnected(volume: String)
    /// A link whose target is not there and is not on a volume: a broken link.
    case brokenLink
    /// A link that leads back to itself (ELOOP).
    case loop
    /// The system does not let this app look (EACCES, EPERM).
    case noPermission
    /// Any other error (a volume that answers with an input/output error, a stale network share).
    case unreadable
}

/// What a scan found, plus what the last probe found. Equal snapshots draw equal pages.
package struct LibrarySnapshot: Equatable, Sendable {
    package var records: [LibraryRecord] = []
    /// Entries that are not listed and why, by name.
    package var problems: [String: EntryProblem] = [:]
    /// Entries whose check did not finish in time (a volume that is slow or hung): their last known record is kept when there is one.
    package var slow: Set<String> = []
    /// The listing of `runs/` itself did not finish in time: this is the last known library.
    package var listingSlow = false
    /// Directories whose `play.lock` was held at the last probe: a world being played or still saving, by any process.
    package var held: Set<String> = []
    /// Worlds that another process seems to be writing: `status` `running` and a sign of life (`metrics.jsonl`, `checkpoints/`, `meta.json`,
    /// the directory) within the world's window (`LibraryScanner.signWindow`). Played worlds too: `alife resume` takes no `play.lock`.
    package var writtenElsewhere: Set<String> = []
    /// Worlds whose status says `running` and that have shown no sign of life for a long while: they were cut off.
    package var cutOff: Set<String> = []
    /// The inode of `runs/` when it was last looked at (the watch on it is re-opened when it changes), nil when `runs/` is not there.
    package var runsIdentity: UInt64?

    package init() {}

    /// Links to a drive that is not connected, by name, and the drives (`/Volumes/NAME`), distinct, sorted.
    package var offline: [String] { problems.filter { if case .driveNotConnected = $0.value { true } else { false } }.keys.sorted() }
    package var offlineVolumes: [String] {
        Array(Set(problems.values.compactMap { if case .driveNotConnected(let v) = $0 { v } else { nil } })).sorted()
    }

    package var names: Set<String> { Set(records.map(\.name)).union(problems.keys).union(slow) }
}

/// Counts of one scan, for the tests and the measurements.
package struct LibraryScanStats: Equatable, Sendable {
    package var listed = 0
    /// Directories whose files were read; the others came from the cache.
    package var read = 0
    package var reused = 0
    package var offline = 0
    package var slow = 0
    package var seconds: Double = 0
}

/// What a click-time look at one folder found.
package enum Presence: Equatable, Sendable { case there, gone, unknown }

/// Reads `runs/` and remembers what it read. One scan at a time (the app calls it from a serial queue). **Every look at the file system of an
/// entry, a `stat`, an `open`, a read of a small file, runs on a concurrent lane with a deadline**: a link into a volume that hangs costs the
/// deadline once, is reported as slow, keeps its last known record and is not asked again while the first look is still stuck; the rest of
/// the library is not held up.
package final class LibraryScanner: @unchecked Sendable {
    /// The modification stamps that decide whether a directory is read again: date to the nanosecond and size, of the directory, `checkpoints/`,
    /// `events/`, `meta.json`, `play.json` and `config.json`.
    struct Stamps: Equatable, Sendable {
        var directory: Stamp?
        var checkpoints: Stamp?
        var events: Stamp?
        var meta: Stamp?
        var play: Stamp?
        var config: Stamp?

        var newest: Date? { [directory, checkpoints, events, meta, play, config].compactMap { $0?.date }.max() }
    }

    struct Stamp: Equatable, Sendable {
        var seconds: Int
        var nanos: Int
        var size: Int
        var date: Date { Date(timeIntervalSince1970: Double(seconds) + Double(nanos) / 1e9) }
    }

    private struct Cached {
        var stamps: Stamps
        var record: LibraryRecord
        /// The newest stamp was older than `settle` when the directory was read: a file rewritten in the same timestamp granule as that read
        /// (a disk with one-second dates) would otherwise be missed for good.
        var settled: Bool
    }

    private enum Inspected: Sendable {
        case directory(Stamps, LibraryRecord, read: Bool, settled: Bool)
        case problem(EntryProblem)
        case skip
    }

    private let lock = NSLock()
    private var cache: [String: Cached] = [:]
    private let stuck = StuckSet()
    private var lastSnapshot = LibrarySnapshot()
    private let lane = DispatchQueue(label: "alife.library.lane", qos: .userInitiated, attributes: .concurrent)
    package private(set) var lastStats = LibraryScanStats()
    package let settle: TimeInterval
    package let timeout: TimeInterval
    /// Called on the lane at the start of every look at an entry, with its name (the tests make one entry hang with it).
    package nonisolated(unsafe) var inspectHook: (@Sendable (String) -> Void)?

    /// `settle`: seconds a stamp must be older than the read to be trusted (2: coarse disk dates). `timeout`: the deadline of each look.
    package init(settle: TimeInterval = 2, timeout: TimeInterval = 3) {
        self.settle = settle
        self.timeout = timeout
    }

    /// The shortest time a world may be quiet and still be running. The engine writes a row of `metrics.jsonl` every `metrics_every` steps
    /// (100 by default, any multiple allowed); a run is assumed to make at least 4 steps a second, and three rows' time is allowed, so the
    /// default is 75 s. `checkpoints/` (a long write), `meta.json` (a resume sets `running` before it has loaded) and the directory count
    /// as signs of life too.
    package static func signWindow(metricsEvery: Int) -> TimeInterval { max(20, 0.75 * Double(max(metricsEvery, 1))) }

    /// How long a `running` world must be silent before it is called cut off: ten times its window, at least ten minutes.
    package static func cutOffAfter(metricsEvery: Int) -> TimeInterval { max(600, 10 * signWindow(metricsEvery: metricsEvery)) }

    // MARK: Scanning

    /// Scans `runsDirectory`. A directory is read again only when a stamp moved (or was too fresh to trust). `held` and `writtenElsewhere`
    /// of the result are empty: `probe` fills them.
    package func scan(runsDirectory: URL, fileManager: FileManager = .default) -> LibrarySnapshot {
        let started = DispatchTime.now().uptimeNanoseconds
        var stats = LibraryScanStats()
        let listing = Lane.run(lane: lane, timeout: timeout, pending: nil, jobs: [("runs", { @Sendable () -> [String] in
            ((try? FileManager.default.contentsOfDirectory(atPath: runsDirectory.path)) ?? []).filter { !$0.hasPrefix(".") }
        })])
        guard let names = listing.done["runs"] else {
            lock.lock()
            defer { lock.unlock() }
            var stale = lastSnapshot
            stale.listingSlow = true
            return stale
        }
        let jobs: [(String, @Sendable () -> Inspected)] = names.map { name in
            (name, { @Sendable [self] in self.inspect(runsDirectory.appendingPathComponent(name, isDirectory: true), name: name) })
        }
        let result = Lane.run(lane: lane, timeout: timeout, pending: stuck, jobs: jobs)
        let identity = Self.identity(of: runsDirectory.path)
        lock.lock()
        defer { lock.unlock() }
        var snapshot = LibrarySnapshot()
        snapshot.runsIdentity = identity
        stats.listed = names.count
        var seen = Set<String>()
        for name in names {
            guard let inspected = result.done[name] else {
                // Slow: keep the last known record, and say so.
                snapshot.slow.insert(name)
                stats.slow += 1
                if let old = cache[name] {
                    snapshot.records.append(old.record)
                    seen.insert(name)
                }
                continue
            }
            switch inspected {
            case .skip: continue
            case .problem(let problem):
                snapshot.problems[name] = problem
                if case .driveNotConnected = problem { stats.offline += 1 }
            case .directory(let stamps, let record, let read, let settled):
                seen.insert(name)
                if read {
                    cache[name] = Cached(stamps: stamps, record: record, settled: settled)
                    stats.read += 1
                } else {
                    stats.reused += 1
                }
                snapshot.records.append(record)
            }
        }
        for gone in cache.keys where !seen.contains(gone) { cache[gone] = nil }
        snapshot.records.sort { $0.name < $1.name }
        stats.seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        lastStats = stats
        lastSnapshot = snapshot
        return snapshot
    }

    private func cached(_ name: String) -> Cached? {
        lock.lock()
        defer { lock.unlock() }
        return cache[name]
    }

    /// One entry, on the lane: what kind of thing it is, and when it is a directory its stamps and (cached or freshly read) record.
    private func inspect(_ url: URL, name: String) -> Inspected {
        inspectHook?(name)
        switch Self.kind(of: url.path) {
        case .other: return .skip
        case .dangling(let target, let code): return .problem(Self.problem(target: target, errno: code))
        case .unreadable(let code): return .problem(Self.problem(target: "", errno: code))
        case .directory:
            let stamps = Self.stamps(of: url)
            if let old = cached(name), old.stamps == stamps, old.settled {
                return .directory(stamps, old.record, read: false, settled: true)
            }
            let record = Self.read(url: url, name: name, stamps: stamps, fileManager: .default)
            let settled = stamps.newest.map { Date().timeIntervalSince($0) > settle } ?? true
            return .directory(stamps, record, read: true, settled: settled)
        }
    }

    static func problem(target: String, errno code: Int32) -> EntryProblem {
        switch code {
        case ENOENT, ENOTDIR:
            if let volume = volume(of: target) { return .driveNotConnected(volume: volume) }
            return .brokenLink
        case ELOOP: return .loop
        case EACCES, EPERM: return .noPermission
        default: return .unreadable
        }
    }

    // MARK: Probing

    /// Which worlds are in use by a process this scan cannot see: the held `play.lock`s of the Played worlds and of the directories that look
    /// like a build, and the worlds (research or played) that say `running` and show a sign of life within their window
    /// (`signWindow`: `metrics.jsonl`, `checkpoints/`, `meta.json`, the directory). Cheap (one `open` and `flock` per played world, four
    /// `stat`s per running world), each on the lane with a deadline: a world whose look did not finish keeps its last answer. `lockHeld` is
    /// injectable so that a measurement on someone else's folder can skip the locks (taking one, however briefly, is not read-only), and
    /// `skip` are the worlds this app is itself starting, playing or saving: their lock is not touched (a probe in the very moment `alife play`
    /// takes it would make the start fail) and they count as held.
    package func probe(
        _ snapshot: LibrarySnapshot, runsDirectory: URL, now: Date = Date(), skipping skip: Set<String> = [],
        lockHeld: @escaping @Sendable (URL) -> Bool = { PlayLockProbe.isHeld(directory: $0) }
    ) -> LibrarySnapshot {
        struct Answer: Sendable {
            var held: Bool?
            var written: Bool
            var cutOff: Bool
        }
        var jobs: [(String, @Sendable () -> Answer)] = []
        for record in snapshot.records {
            let dir = runsDirectory.appendingPathComponent(record.name, isDirectory: true)
            let name = record.name
            let wantsLock = (record.isPlayed || record.looksUnfinished) && !skip.contains(name)
            let wantsLife = record.status == "running" && (record.isPlayed || !record.steps.isEmpty)
            guard wantsLock || wantsLife else { continue }
            let window = Self.signWindow(metricsEvery: record.metricsEvery)
            let cutOffAfter = Self.cutOffAfter(metricsEvery: record.metricsEvery)
            jobs.append((name, {
                var answer = Answer(held: nil, written: false, cutOff: false)
                if wantsLock { answer.held = lockHeld(dir) }
                if wantsLife {
                    let signs = ["metrics.jsonl", "checkpoints", "meta.json", ""].compactMap {
                        Self.stamp(path: $0.isEmpty ? dir.path : dir.appendingPathComponent($0).path)?.date
                    }
                    if let newest = signs.max() {
                        let age = now.timeIntervalSince(newest)
                        answer.written = age < window
                        answer.cutOff = age > cutOffAfter
                    }
                }
                return answer
            }))
        }
        let result = Lane.run(lane: lane, timeout: timeout, pending: nil, jobs: jobs)
        var out = snapshot
        out.held = []
        out.writtenElsewhere = []
        out.cutOff = []
        out.runsIdentity = Self.identity(of: runsDirectory.path)
        for (name, _) in jobs {
            guard let answer = result.done[name] else {
                // The look did not finish: keep what was known.
                if snapshot.held.contains(name) { out.held.insert(name) }
                if snapshot.writtenElsewhere.contains(name) { out.writtenElsewhere.insert(name) }
                if snapshot.cutOff.contains(name) { out.cutOff.insert(name) }
                continue
            }
            if answer.held == true { out.held.insert(name) }
            if answer.written { out.writtenElsewhere.insert(name) }
            if answer.cutOff, answer.held != true { out.cutOff.insert(name) }
        }
        for name in skip where snapshot.records.contains(where: { $0.name == name && ($0.isPlayed || $0.looksUnfinished) }) { out.held.insert(name) }
        return out
    }

    /// Whether `directory` is still there, looked at on the lane with a deadline (a click must not wait for a volume that hangs: an
    /// answer that does not come is `.unknown`, and the caller goes on).
    package func presence(of directory: URL) -> Presence {
        let path = directory.path
        let result = Lane.run(lane: lane, timeout: min(timeout, 1.5), pending: nil, jobs: [("p", { @Sendable () -> Kind in Self.kind(of: path) })])
        guard let kind = result.done["p"] else { return .unknown }
        return kind == .directory ? .there : .gone
    }

    // MARK: Reading one directory

    private static func read(url: URL, name: String, stamps: Stamps, fileManager fm: FileManager) -> LibraryRecord {
        let play = PlayWorldFile.read(at: url.appendingPathComponent(PlayLockProbe.playFile))
        let meta = PlayCatalogScanner.readJSON(url.appendingPathComponent("meta.json"))
        let config = PlayCatalogScanner.readJSON(url.appendingPathComponent("config.json"))
        let world = config?["world"] as? [String: Any]
        let weather = config?["weather"] as? [String: Any]
        let run = config?["run"] as? [String: Any]
        let geometry = (world?["geometry"] as? String) ?? (world == nil ? nil : "box")
        let sphere = (world?["geometry"] as? String) == "sphere"
        let contents = PlayCatalogScanner.entries(in: url, fileManager: fm)
        let directoryModified = stamps.directory?.date ?? .distantPast
        return LibraryRecord(
            name: name, play: play, steps: PlayCatalogScanner.checkpointSteps(in: url, fileManager: fm),
            status: meta?["status"] as? String, parent: SourceRef.parse(meta?["parent"] as? String),
            dims: world?["dims"] as? Int, geometry: geometry, worldSize: (world?["size"] as? NSNumber)?.doubleValue,
            planetRadius: sphere ? (world?["radius"] as? NSNumber)?.doubleValue : nil,
            planetShape: sphere ? PlayCatalogScanner.planetShape(of: world) : nil,
            weatherOn: weather?["enabled"] as? Bool ?? false,
            hasEvents: hasChunk(in: url.appendingPathComponent("events")),
            directoryModified: directoryModified,
            lastSaved: max(directoryModified, stamps.checkpoints?.date ?? .distantPast),
            hasContent: !contents.subtracting([PlayLockProbe.lockFile]).isEmpty,
            looksUnfinished: play == nil && PlayCatalogScanner.isUnfinishedBuild(url, meta: meta, fileManager: fm),
            buildMark: (meta?[PlayBuildMarker.metaKey] as? String).flatMap { PlayBuildMarker.kinds.contains($0) ? $0 : nil },
            metricsEvery: (run?["metrics_every"] as? Int).flatMap { $0 >= 1 ? $0 : nil } ?? play?.cadence.metricsEvery ?? 100)
    }

    /// Whether `directory` has an entry that is not hidden, without listing it (an `events/` of a long run has thousands).
    static func hasChunk(in directory: URL) -> Bool {
        guard let dir = opendir(directory.path) else { return false }
        defer { closedir(dir) }
        while let entry = readdir(dir) {
            let name = withUnsafePointer(to: &entry.pointee.d_name) {
                $0.withMemoryRebound(to: CChar.self, capacity: Int(entry.pointee.d_namlen) + 1) { String(cString: $0) }
            }
            if !name.hasPrefix(".") { return true }
        }
        return false
    }

    // MARK: Stat

    package enum Kind: Equatable, Sendable {
        case directory
        /// A link whose target cannot be reached: the target's text and the error (`ENOENT`, `ELOOP`, `EACCES`...).
        case dangling(String, Int32)
        /// Something that is not a directory (a file, a log).
        case other
        /// `lstat` itself failed with this error (permission, input/output).
        case unreadable(Int32)
    }

    package static func kind(of path: String) -> Kind {
        var link = stat()
        guard lstat(path, &link) == 0 else { return errno == ENOENT ? .other : .unreadable(errno) }
        if (link.st_mode & S_IFMT) == S_IFLNK {
            var target = stat()
            if stat(path, &target) != 0 {
                let code = errno
                return .dangling((try? FileManager.default.destinationOfSymbolicLink(atPath: path)) ?? "", code)
            }
            return (target.st_mode & S_IFMT) == S_IFDIR ? .directory : .other
        }
        return (link.st_mode & S_IFMT) == S_IFDIR ? .directory : .other
    }

    /// The name of the drive in `/Volumes/NAME/...`, nil for any other path.
    static func volume(of target: String) -> String? {
        let parts = target.split(separator: "/", omittingEmptySubsequences: true)
        guard parts.count >= 2, parts[0] == "Volumes" else { return nil }
        return String(parts[1])
    }

    static func stamp(path: String) -> Stamp? {
        var info = stat()
        guard stat(path, &info) == 0 else { return nil }
        return Stamp(seconds: Int(info.st_mtimespec.tv_sec), nanos: Int(info.st_mtimespec.tv_nsec), size: Int(info.st_size))
    }

    static func identity(of path: String) -> UInt64? {
        var info = stat()
        return stat(path, &info) == 0 ? UInt64(info.st_ino) : nil
    }

    static func stamps(of url: URL) -> Stamps {
        Stamps(
            directory: stamp(path: url.path), checkpoints: stamp(path: url.appendingPathComponent("checkpoints").path),
            events: stamp(path: url.appendingPathComponent("events").path), meta: stamp(path: url.appendingPathComponent("meta.json").path),
            play: stamp(path: url.appendingPathComponent(PlayLockProbe.playFile).path),
            config: stamp(path: url.appendingPathComponent("config.json").path))
    }
}

/// The looks that were started and have not come back: a hung entry is not asked again while its first look is stuck.
private final class StuckSet: @unchecked Sendable {
    private let lock = NSLock()
    private var keys = Set<String>()
    /// False when `key` is already out.
    func begin(_ key: String) -> Bool { lock.lock(); defer { lock.unlock() }; return keys.insert(key).inserted }
    func end(_ key: String) { lock.lock(); keys.remove(key); lock.unlock() }
}

/// Runs jobs on a concurrent queue and waits for them until a deadline. A job that is late is left running and its result is dropped.
private enum Lane {
    final class Collector<T: Sendable>: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [String: T] = [:]
        func set(_ key: String, _ value: T) { lock.lock(); values[key] = value; lock.unlock() }
        var all: [String: T] { lock.lock(); defer { lock.unlock() }; return values }
    }

    static func run<T: Sendable>(
        lane: DispatchQueue, timeout: TimeInterval, pending: StuckSet?, jobs: [(String, @Sendable () -> T)]
    ) -> (done: [String: T], late: Set<String>) {
        guard !jobs.isEmpty else { return ([:], []) }
        let collector = Collector<T>()
        let group = DispatchGroup()
        for (key, work) in jobs {
            if let pending, !pending.begin(key) { continue }
            group.enter()
            lane.async {
                collector.set(key, work())
                pending?.end(key)
                group.leave()
            }
        }
        _ = group.wait(timeout: .now() + timeout)
        let done = collector.all
        return (done, Set(jobs.map(\.0)).subtracting(done.keys))
    }
}

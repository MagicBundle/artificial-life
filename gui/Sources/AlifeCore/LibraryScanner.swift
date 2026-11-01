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

    package var id: String { name }
    package var isPlayed: Bool { play != nil }
}

/// What a scan found, plus what the last probe found. Equal snapshots draw equal pages.
package struct LibrarySnapshot: Equatable, Sendable {
    package var records: [LibraryRecord] = []
    /// Entries of `runs/` that are links whose target is not there (an archive on a drive that is not connected), by name.
    package var offline: [String] = []
    /// The drive names of the offline links (`/Volumes/NAME/...`), distinct, sorted.
    package var offlineVolumes: [String] = []
    /// Directories whose `play.lock` was held at the last probe: a world being played or still saving, by any process.
    package var held: Set<String> = []
    /// Research runs that another process seems to be writing (`status` "running" and `metrics.jsonl` changed a moment ago).
    package var writtenElsewhere: Set<String> = []

    package init() {}

    package var names: Set<String> { Set(records.map(\.name)).union(offline) }
}

/// Counts of one scan, for the tests and the measurements.
package struct LibraryScanStats: Equatable, Sendable {
    package var listed = 0
    /// Directories whose files were read; the others came from the cache.
    package var read = 0
    package var reused = 0
    package var offline = 0
    package var seconds: Double = 0
}

/// Reads `runs/` and remembers what it read. One scan at a time (the app calls it from a serial queue); the lock makes a stray second
/// call safe, not fast.
package final class LibraryScanner: @unchecked Sendable {
    /// The modification stamps that decide whether a directory is read again.
    struct Stamps: Equatable {
        var directory: Stamp?
        var checkpoints: Stamp?
        var events: Stamp?
        var meta: Stamp?
        var play: Stamp?
        var config: Stamp?
    }

    struct Stamp: Equatable {
        var seconds: Int
        var nanos: Int
        var date: Date { Date(timeIntervalSince1970: Double(seconds) + Double(nanos) / 1e9) }
    }

    private struct Cached {
        var stamps: Stamps
        var record: LibraryRecord
    }

    private let lock = NSLock()
    private var cache: [String: Cached] = [:]
    package private(set) var lastStats = LibraryScanStats()

    package init() {}

    /// A window in which a research run's `metrics.jsonl` counts as being written now. The engine writes a row every `metrics_every`
    /// steps (100 by default), which a slow preset takes tens of seconds to reach.
    package static let writeWindow: TimeInterval = 20

    /// Scans `runsDirectory`. A directory is read again only when a stamp moved. `held` and `writtenElsewhere` of the result are empty:
    /// `probe` fills them.
    package func scan(runsDirectory: URL, fileManager fm: FileManager = .default) -> LibrarySnapshot {
        lock.lock()
        defer { lock.unlock() }
        let started = DispatchTime.now().uptimeNanoseconds
        var stats = LibraryScanStats()
        var snapshot = LibrarySnapshot()
        let names = ((try? fm.contentsOfDirectory(atPath: runsDirectory.path)) ?? []).filter { !$0.hasPrefix(".") }
        var seen = Set<String>()
        var volumes = Set<String>()
        for name in names {
            let url = runsDirectory.appendingPathComponent(name, isDirectory: true)
            stats.listed += 1
            switch Self.kind(of: url.path) {
            case .directory:
                break
            case .dangling(let target):
                snapshot.offline.append(name)
                if let volume = Self.volume(of: target) { volumes.insert(volume) }
                stats.offline += 1
                continue
            case .other:
                continue
            }
            seen.insert(name)
            let stamps = Self.stamps(of: url)
            if let cached = cache[name], cached.stamps == stamps {
                snapshot.records.append(cached.record)
                stats.reused += 1
                continue
            }
            let record = Self.read(url: url, name: name, stamps: stamps, fileManager: fm)
            cache[name] = Cached(stamps: stamps, record: record)
            snapshot.records.append(record)
            stats.read += 1
        }
        for gone in cache.keys where !seen.contains(gone) { cache[gone] = nil }
        snapshot.records.sort { $0.name < $1.name }
        snapshot.offline.sort()
        snapshot.offlineVolumes = volumes.sorted()
        stats.seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        lastStats = stats
        return snapshot
    }

    /// Which worlds are in use by a process this scan cannot see: the held `play.lock`s of the Played worlds and of the directories that
    /// look like a build, and the research runs that say `running` and whose `metrics.jsonl` changed within `writeWindow`. Cheap (one
    /// `open` and `flock` per world, one `stat` per running run); the app calls it on a timer. `lockHeld` is injectable so that a
    /// measurement on someone else's folder can skip the locks (taking one, however briefly, is not read-only).
    package func probe(
        _ snapshot: LibrarySnapshot, runsDirectory: URL, now: Date = Date(),
        lockHeld: (URL) -> Bool = { PlayLockProbe.isHeld(directory: $0) }
    ) -> LibrarySnapshot {
        var out = snapshot
        out.held = []
        out.writtenElsewhere = []
        for record in snapshot.records {
            let dir = runsDirectory.appendingPathComponent(record.name, isDirectory: true)
            if record.isPlayed || record.looksUnfinished {
                if lockHeld(dir) { out.held.insert(record.name) }
            } else if record.status == "running", !record.steps.isEmpty {
                let metrics = dir.appendingPathComponent("metrics.jsonl").path
                if let stamp = Self.stamp(path: metrics), now.timeIntervalSince(stamp.date) < Self.writeWindow {
                    out.writtenElsewhere.insert(record.name)
                }
            }
        }
        return out
    }

    // MARK: Reading one directory

    private static func read(url: URL, name: String, stamps: Stamps, fileManager fm: FileManager) -> LibraryRecord {
        let play = PlayWorldFile.read(at: url.appendingPathComponent(PlayLockProbe.playFile))
        let meta = PlayCatalogScanner.readJSON(url.appendingPathComponent("meta.json"))
        let config = PlayCatalogScanner.readJSON(url.appendingPathComponent("config.json"))
        let world = config?["world"] as? [String: Any]
        let weather = config?["weather"] as? [String: Any]
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
            buildMark: (meta?[PlayBuildMarker.metaKey] as? String).flatMap { PlayBuildMarker.kinds.contains($0) ? $0 : nil })
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

    package enum Kind: Equatable {
        case directory
        /// A link whose target does not exist, with the target's path.
        case dangling(String)
        case other
    }

    package static func kind(of path: String) -> Kind {
        var link = stat()
        guard lstat(path, &link) == 0 else { return .other }
        if (link.st_mode & S_IFMT) == S_IFLNK {
            var target = stat()
            if stat(path, &target) != 0 {
                return .dangling((try? FileManager.default.destinationOfSymbolicLink(atPath: path)) ?? "")
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
        return Stamp(seconds: Int(info.st_mtimespec.tv_sec), nanos: Int(info.st_mtimespec.tv_nsec))
    }

    static func stamps(of url: URL) -> Stamps {
        Stamps(
            directory: stamp(path: url.path), checkpoints: stamp(path: url.appendingPathComponent("checkpoints").path),
            events: stamp(path: url.appendingPathComponent("events").path), meta: stamp(path: url.appendingPathComponent("meta.json").path),
            play: stamp(path: url.appendingPathComponent(PlayLockProbe.playFile).path),
            config: stamp(path: url.appendingPathComponent("config.json").path))
    }
}

import Foundation

// What a played world takes on disk and which saves it keeps (DESIGN §34.10). The client measures the world's directory itself, with
// `stat` and nothing else (no file is opened, the engine does no work for it); the engine owns the save mode and echoes it in
// `play.saves` and `CLCK.flags`. Everything here is pure or reads only file attributes, so it is tested on a temporary directory.

/// One measurement of a world directory.
package struct StorageSample: Equatable, Sendable {
    /// Everything under `checkpoints/` (a save being written, `*.tmp`, included).
    package var savesBytes: Int64
    /// Everything else: the event chunks, the observers' files, the metrics, the director's and the player's logs, the reports.
    package var logsBytes: Int64
    /// The complete saves (`step_<t>.safetensors`).
    package var saveCount: Int
    /// The size of the newest complete save, 0 when there is none.
    package var newestSaveBytes: Int64
    /// The complete saves that are also linked from another file (a planet's origin, linked to its source's save on the same volume,
    /// DESIGN 34.11): they take the disk once, but both worlds count them. Counted here, so the details can say so.
    package var sharedSaveCount: Int
    /// What the volume of the world has free, if it said.
    package var freeBytes: Int64?

    package var totalBytes: Int64 { savesBytes + logsBytes }

    package init(savesBytes: Int64 = 0, logsBytes: Int64 = 0, saveCount: Int = 0, newestSaveBytes: Int64 = 0,
                 sharedSaveCount: Int = 0, freeBytes: Int64? = nil) {
        self.savesBytes = savesBytes
        self.logsBytes = logsBytes
        self.saveCount = saveCount
        self.newestSaveBytes = newestSaveBytes
        self.sharedSaveCount = sharedSaveCount
        self.freeBytes = freeBytes
    }
}

/// Measures a world directory from file attributes only. A file that appears or disappears during the walk (a save written under a
/// temporary name and renamed, an old save pruned) is counted or not, never an error.
package enum StorageScan {
    /// The checkpoint a world holds, `step_<ten digits>.safetensors`, or its compressed form `step_<ten digits>.safetensors.zst`
    /// (DESIGN 34.11). A temporary name (`*.tmp`) is not a save.
    package static func saveStep(of name: String) -> Int? {
        var stem = Substring(name)
        if stem.hasSuffix(".zst") { stem = stem.dropLast(4) }
        guard stem.hasPrefix("step_"), stem.hasSuffix(".safetensors") else { return nil }
        return Int(stem.dropFirst("step_".count).dropLast(".safetensors".count))
    }

    /// The size of a regular file on disk (allocated blocks, which is what the volume loses), nil for anything else or a file that
    /// is gone.
    private static func size(of url: URL) -> Int64? {
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .totalFileAllocatedSizeKey, .fileSizeKey]
        guard let values = try? url.resourceValues(forKeys: keys), values.isRegularFile == true else { return nil }
        return Int64(values.totalFileAllocatedSize ?? values.fileSize ?? 0)
    }

    /// What a file is on its volume: its device and inode, and how many names it has (its hard links). One stat.
    private struct Identity: Hashable {
        let device: UInt64
        let inode: UInt64
        let links: Int
    }

    private static func identity(of url: URL, fileManager fm: FileManager) -> Identity? {
        guard let attributes = try? fm.attributesOfItem(atPath: url.path),
              let device = (attributes[.systemNumber] as? NSNumber)?.uint64Value,
              let inode = (attributes[.systemFileNumber] as? NSNumber)?.uint64Value,
              let links = (attributes[.referenceCount] as? NSNumber)?.intValue else { return nil }
        return Identity(device: device, inode: inode, links: links)
    }

    /// The free space of the volume `url` is on: what the system would let an important write use, else the plain figure.
    package static func freeBytes(at url: URL) -> Int64? {
        let keys: Set<URLResourceKey> = [.volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey]
        guard let values = try? url.resourceValues(forKeys: keys) else { return nil }
        if let important = values.volumeAvailableCapacityForImportantUsage, important > 0 { return important }
        return values.volumeAvailableCapacity.map(Int64.init)
    }

    /// Measures `world`. The saves are the files of `checkpoints/`, followed when that directory is itself a link (a world's saves
    /// kept on another volume); the logs are every other file of the world. Sizes are what the files take on disk (allocated blocks).
    package static func measure(world: URL, fileManager fm: FileManager = .default) -> StorageSample {
        let root = world.resolvingSymlinksInPath()  // a world archived to another volume is a link to it
        var sample = StorageSample(freeBytes: freeBytes(at: root))
        if let walk = fm.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey],
                                    options: [], errorHandler: { _, _ in true }) {
            for case let url as URL in walk {
                if walk.level == 1, url.lastPathComponent == "checkpoints" {
                    walk.skipDescendants()  // counted below, through its link if it is one
                    continue
                }
                if let bytes = size(of: url) { sample.logsBytes += bytes }
            }
        }
        let saves = root.appendingPathComponent("checkpoints", isDirectory: true).resolvingSymlinksInPath()
        var newest: (step: Int, bytes: Int64)?
        var counted = Set<Identity>()  // a save linked twice inside this world is one file here: counted once
        if let walk = fm.enumerator(at: saves, includingPropertiesForKeys: [.isRegularFileKey],
                                    options: [], errorHandler: { _, _ in true }) {
            for case let url as URL in walk {
                guard let bytes = size(of: url) else { continue }
                guard let step = saveStep(of: url.lastPathComponent) else {
                    sample.savesBytes += bytes  // not a save (a save being written, LATEST)
                    continue
                }
                let identity = identity(of: url, fileManager: fm)
                if let identity, !counted.insert(identity).inserted { continue }
                sample.savesBytes += bytes
                sample.saveCount += 1
                if (identity?.links ?? 1) > 1 { sample.sharedSaveCount += 1 }
                if newest.map({ step > $0.step }) ?? true { newest = (step, bytes) }
            }
        }
        sample.newestSaveBytes = newest?.bytes ?? 0
        return sample
    }
}

/// Sizes as text: decimal units (what the Finder shows), one decimal under 10, none above.
package enum StorageFormat {
    private static let units: [(name: String, scale: Double)] = [
        ("B", 1), ("kB", 1e3), ("MB", 1e6), ("GB", 1e9), ("TB", 1e12),
    ]

    package static func bytes(_ count: Int64) -> String {
        let n = Double(max(count, 0))
        var index = 0
        // (a value that would print as 1000 of a unit is shown in the next one)
        while index + 1 < units.count, n >= units[index + 1].scale * 0.9995 { index += 1 }
        let value = n / units[index].scale
        if index == 0 { return String(format: "%.0f B", value) }
        return String(format: value < 9.95 ? "%.1f %@" : "%.0f %@", value, units[index].name)
    }

    /// A change in size: "+180 MB", "-1.4 GB", "+0 B".
    package static func signed(_ count: Int64) -> String {
        (count < 0 ? "-" : "+") + bytes(abs(count))
    }
}

/// How worried the player should be about the space left.
package enum StorageHealth: Int, Comparable, Sendable {
    case ok, low, critical

    package static func < (a: Self, b: Self) -> Bool { a.rawValue < b.rawValue }

    /// A save of this size is assumed when the world has none yet (a 3D planet's checkpoint is about 700 MB).
    package static let typicalSaveBytes: Int64 = 700_000_000
    /// "Low" below this many saves of room, "critical" below `criticalSaves`: a save is written whole before the old one goes, so
    /// the next one needs one save of room (and a little more).
    package static let lowSaves: Int64 = 3
    package static let criticalNumerator: Int64 = 5  // 5/4 of a save
    package static let criticalDenominator: Int64 = 4

    package static func level(free: Int64?, saveBytes: Int64) -> StorageHealth {
        guard let free else { return .ok }
        let save = saveBytes > 0 ? saveBytes : typicalSaveBytes
        if free * criticalDenominator < save * criticalNumerator { return .critical }
        if free < save * lowSaves { return .low }
        return .ok
    }
}

/// How the size grew since the world went live and since the chapter began. Fed with the totals of the samples, the chapter of the
/// latest clock and whether a save was being written: a chapter ends with a save, which belongs to the chapter it closes, so a
/// chapter's growth counts from the first sample taken once that save is done (a world attached mid-chapter counts from its first
/// sample). A prune makes a negative growth.
package struct StorageTrend: Equatable, Sendable {
    package private(set) var sessionStart: Int64?
    package private(set) var chapter: UInt32?
    package private(set) var chapterStart: Int64?

    package init() {}

    package mutating func note(total: Int64, chapter now: UInt32?, saving: Bool = false) {
        if sessionStart == nil { sessionStart = total }
        guard let now else { return }
        if chapter != now {
            chapter = now
            chapterStart = nil
        }
        if chapterStart == nil, !saving { chapterStart = total }
    }

    package func sinceStart(_ total: Int64) -> Int64? { sessionStart.map { total - $0 } }
    package func sinceChapter(_ total: Int64) -> Int64? { chapterStart.map { total - $0 } }
}

/// The save mode the window shows: what it asked and the world has not yet confirmed, else the latest `CLCK`, else the header, else
/// `all` (a server that does not know the line keeps its last saves).
package enum SaveModeView {
    package static func current(clock: PlayClock?, header: PlayHeader.Saves?, requested: SaveMode?) -> SaveMode {
        if let requested { return requested }
        if let clock { return clock.savesSingle ? .single : (clock.savesNewestOnly ? .newest : .all) }
        return header.flatMap { SaveMode(rawValue: $0.mode) } ?? .all
    }

    /// Whether the window offers switching to `target`: the world's server knows the line (it sent `play.saves`), and `target` is
    /// `all` in a research run played in place (`adopted`), whose older checkpoints are the research's: the world refuses the other
    /// two lines there, so the choice does not pretend to take them.
    package static func offers(_ target: SaveMode, header: PlayHeader.Saves?) -> Bool {
        guard let header else { return false }
        return target == .all || header.adopted != true
    }
}

/// What the Storage strip and its details say, from a sample. Pure, so the text is tested without a window.
package struct StorageReadout: Equatable, Sendable {
    package var mode: SaveMode
    /// "2.6 GB", or "measuring…" before the first sample.
    package var total: String
    /// What the strip says to a screen reader: the size, and the room left when the disk is getting full ("2.1 GB, 1.2 GB free").
    package var stripText: String
    /// The room left, printed small beside the size while the disk is getting full (nil while it is not).
    package var room: String?
    package var saves: String
    package var saveCount: Int
    package var logs: String
    package var free: String?
    package var sinceStart: String?
    package var sinceChapter: String?
    package var health: StorageHealth
    /// Why the player should look at the disk, if so.
    package var warning: String?
    /// More saves are on disk than the newest save and the origin (`newest`, `all`) or than the newest save alone (`single`) keep:
    /// the next save deletes the others. (The details say it only where the mode deletes; `all` keeps its last few.)
    package var pruningPending: Bool { mode == .single ? saveCount >= 2 : saveCount >= 3 }
    /// The saves of this world that are linked from another (a planet's origin, DESIGN 34.11), said once in the details: they are on
    /// disk once and counted in both worlds. Nil when there are none.
    package var shared: String?

    package init(sample: StorageSample?, trend: StorageTrend = StorageTrend(), mode: SaveMode) {
        self.mode = mode
        guard let sample else {
            total = "measuring…"
            stripText = total
            saves = "-"
            saveCount = 0
            logs = "-"
            health = .ok
            return
        }
        let size = StorageFormat.bytes(sample.totalBytes)
        total = size
        saves = StorageFormat.bytes(sample.savesBytes)
        saveCount = sample.saveCount
        if sample.sharedSaveCount > 0 {
            let n = sample.sharedSaveCount
            shared = "\(n) save\(n == 1 ? " is" : "s are") linked from another world: on disk once, counted in both."
        }
        logs = StorageFormat.bytes(sample.logsBytes)
        free = sample.freeBytes.map { StorageFormat.bytes($0) + " free" }
        sinceStart = trend.sinceStart(sample.totalBytes).map(StorageFormat.signed)
        sinceChapter = trend.sinceChapter(sample.totalBytes).map(StorageFormat.signed)
        health = StorageHealth.level(free: sample.freeBytes, saveBytes: sample.newestSaveBytes)
        let save = StorageFormat.bytes(sample.newestSaveBytes > 0 ? sample.newestSaveBytes : StorageHealth.typicalSaveBytes)
        let room = health == .ok ? nil : free
        self.room = room
        stripText = room.map { size + ", " + $0 } ?? size
        switch health {
        case .ok: break
        case .low: warning = "Low disk space: \(free ?? "") (a save is about \(save))."
        case .critical: warning = "Almost no disk space: the next save (about \(save)) may not fit."
        }
    }
}

import AlifeCore
import Foundation

/// What the files of one world say right now: its last row of numbers, the autopilot's interventions and journal, how the engine says it ended
/// and the settings it was made with (docs/APP.md §3.7, §7.5). It is read by *name*, from the world's own directory: it does not depend on
/// the form of the Set up page (`RunSettings.targetRun`) or on the run this app grows (`RunController.trackedRun`), so a world that is not
/// growing has its numbers and its journal on Details, and a world that grows has them in the Run panel.
struct WorldReading: Equatable {
    var world: String
    /// The newest complete row of `metrics.jsonl`, with the feature keys that only some rows carry kept from the rows before; nil without one.
    var metrics: MetricsSnapshot?
    /// `director.jsonl` and `journal.jsonl`; `hasLog` false when the world ran without an autopilot.
    var director: DirectorSnapshot?
    /// `meta.json`'s `status` (`running`, `done`, `extinct`, `violation`, `interrupted`).
    var status: String?
    /// `config.json`, in the page's words; nil when it cannot be read.
    var madeWith: MadeWith?
    /// The folder was there at the read.
    var exists: Bool
}

/// The reading itself: pure and synchronous, run off the main thread.
enum WorldReadingIO {
    /// `memory` is the feature keys seen for this world so far (a row without a heavy key keeps the last value of it).
    static func read(runDirectory: URL, world: String, memory: [String: Double] = [:], includeMadeWith: Bool) -> WorldReading {
        var exists: ObjCBool = false
        let there = FileManager.default.fileExists(atPath: runDirectory.path, isDirectory: &exists) && exists.boolValue
        guard there else { return WorldReading(world: world, metrics: nil, director: nil, status: nil, madeWith: nil, exists: false) }
        var metrics = MetricsTailer.readLast(at: runDirectory.appendingPathComponent("metrics.jsonl"))
        if var merged = metrics {
            merged.features = memory.merging(merged.features) { _, new in new }
            metrics = merged
        }
        return WorldReading(
            world: world, metrics: metrics, director: DirectorTailer.snapshot(runDirectory: runDirectory),
            status: status(in: runDirectory),
            madeWith: includeMadeWith ? MadeWith.read(runDirectory: runDirectory) : nil, exists: true)
    }
}

extension WorldReadingIO {
    /// The `status` the engine wrote in `meta.json`; nil when it cannot be read.
    static func status(in directory: URL) -> String? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("meta.json")),
              let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return meta["status"] as? String
    }
}

/// Follows one world's files for the Run panel and for Details. It polls only while someone shows it (`retain`/`release`), every two seconds,
/// off the main thread, and never lets an older read overwrite a newer one or a read of another world overwrite this one.
@MainActor
final class WorldReader: ObservableObject {
    @Published private(set) var reading: WorldReading?
    /// The world followed (its name in `runs/`).
    private(set) var world: String?

    private let runsDirectory: () -> URL?
    private let interval: Duration
    private var shown = 0
    private var task: Task<Void, Never>?
    private var serial = 0
    private var applied = 0
    private var memory: [String: Double] = [:]
    private var madeWithOf: String?
    private var madeWith: MadeWith?

    /// `runsDirectory` is `<alife folder>/runs`, nil without a usable folder.
    init(runsDirectory: @escaping () -> URL?, interval: Duration = .seconds(2)) {
        self.runsDirectory = runsDirectory
        self.interval = interval
    }

    /// Follows `world`. A different world clears what was read at once: numbers of another world are never shown under this one's name.
    func follow(_ name: String?) {
        guard name != world else { return }
        world = name
        reading = nil
        memory = [:]
        madeWithOf = nil
        madeWith = nil
        serial += 1  // (a read of the world before is on its way: it is not applied)
        applied = serial
        if shown > 0, name != nil { readSoon() }
    }

    /// Someone shows the readings: they are kept up to date until the matching `release`.
    func retain() {
        shown += 1
        guard shown == 1 else { return }
        readSoon()
        task = Task { [weak self, interval] in
            while !Task.isCancelled {
                try? await Task.sleep(for: interval)
                guard let self, !Task.isCancelled else { return }
                self.readSoon()
            }
        }
    }

    func release() {
        shown = max(shown - 1, 0)
        guard shown == 0 else { return }
        task?.cancel()
        task = nil
    }

    var isPolling: Bool { task != nil }

    /// Reads the files again, whenever someone shows them.
    func readSoon() { Task { await readNow() } }

    /// One read, applied if it is still the newest and the world is still this one. The tests await it.
    func readNow() async {
        guard let name = world, let runs = runsDirectory() else {
            if reading != nil { reading = nil }
            return
        }
        serial += 1
        let mine = serial
        let directory = runs.appendingPathComponent(name, isDirectory: true)
        let memory = memory
        let includeMadeWith = madeWithOf != name
        let result = await Task.detached(priority: .utility) {
            WorldReadingIO.read(runDirectory: directory, world: name, memory: memory, includeMadeWith: includeMadeWith)
        }.value
        guard mine > applied, world == name else { return }
        applied = mine
        var shownReading = result
        if includeMadeWith {
            madeWith = result.madeWith
            if result.madeWith != nil { madeWithOf = name }  // (a config that is not written yet is read again)
        } else {
            shownReading.madeWith = madeWith
        }
        self.memory = shownReading.metrics?.features ?? [:]
        if reading != shownReading { reading = shownReading }
    }
}

import Foundation

// What the launcher lists under `runs/`: the worlds that `alife play` can continue (a directory with `play.json`), the
// directories of a planet whose build did not finish, and the runs a planet can be made from (DESIGN §33.3.3).

/// `play.json` (version 1), as `RunDir.write_play_json` writes it.
package struct PlayWorldFile: Equatable, Sendable, Codable {
    package struct Cadence: Equatable, Sendable, Codable {
        package var metricsEvery: Int
        package var heavyMetricsEvery: Int
        package var checkpointEvery: Int
        package var keepCheckpoints: Int

        private enum CodingKeys: String, CodingKey {
            case metricsEvery = "metrics_every"
            case heavyMetricsEvery = "heavy_metrics_every"
            case checkpointEvery = "checkpoint_every"
            case keepCheckpoints = "keep_checkpoints"
        }
    }

    package struct Weather: Equatable, Sendable, Codable {
        package var dcRef: Double

        private enum CodingKeys: String, CodingKey {
            case dcRef = "dc_ref"
        }
    }

    package struct Planet: Equatable, Sendable, Codable {
        package var source: String?
        package var step: Int
        package var weatherSpinup: Int?
        /// `"exact"`, `"clade"` or `"none"`.
        package var houses: String
        package var madeAtStep: Int

        private enum CodingKeys: String, CodingKey {
            case source, step, houses
            case weatherSpinup = "weather_spinup"
            case madeAtStep = "made_at_step"
        }
    }

    package var playVersion: Int
    package var baseRate: Double
    package var cadence: Cadence
    package var tick: Int
    package var weather: Weather?
    package var planet: Planet?

    private enum CodingKeys: String, CodingKey {
        case playVersion = "play_version"
        case baseRate = "base_rate"
        case cadence, tick, weather, planet
    }

    /// Reads `play.json`; nil when it is not a version-1 object.
    package static func read(at url: URL) -> PlayWorldFile? {
        guard let data = try? Data(contentsOf: url),
              let file = try? JSONDecoder().decode(PlayWorldFile.self, from: data), file.playVersion == 1 else {
            return nil
        }
        return file
    }
}

/// A world the launcher can resume.
package struct PlayWorld: Equatable, Sendable, Identifiable {
    package var name: String
    package var file: PlayWorldFile
    /// The step of the newest checkpoint, nil if there is none (the directory was cut).
    package var newestStep: Int?
    /// `meta.json`'s status (`done`, `interrupted`, `running`, ...).
    package var status: String?
    package var modified: Date
    /// Another process holds `play.lock`: the world is being played, or is still saving its quit checkpoint.
    package var locked: Bool

    package var id: String { name }
}

/// A run a planet can be made from.
package struct PlaySource: Equatable, Sendable, Identifiable {
    package var name: String
    /// The steps of the checkpoints, ascending.
    package var checkpoints: [Int]
    /// The `world.dims` of its config, nil if unreadable.
    package var dims: Int?
    /// `weather.enabled` of its config: a source whose weather is on is forked without the overlay and the spin-up.
    package var weatherOn: Bool
    /// Whether it has event chunks, which the exact houses of a planet are rebuilt from (`--houses auto` falls back to
    /// the clades without them).
    package var hasEvents: Bool
    package var modified: Date

    package var id: String { name }
}

/// What a scan of `runs/` found.
package struct PlayCatalog: Equatable, Sendable {
    /// Worlds with a `play.json`, newest first.
    package var worlds: [PlayWorld] = []
    /// Directories that look like a planet whose build was cancelled or failed: a run with no `play.json` and no
    /// checkpoint. Not resumable; the app deletes nothing.
    package var unfinished: [String] = []
    /// Runs with a checkpoint, newest first (including worlds: a planet can be forked again).
    package var sources: [PlaySource] = []

    package init() {}

    package var existingNames: Set<String> {
        Set(worlds.map(\.name)).union(unfinished).union(sources.map(\.name))
    }
}

package enum PlayCatalogScanner {
    /// Scans the directories under `runsDirectory`.
    package static func scan(runsDirectory: URL, fileManager fm: FileManager = .default) -> PlayCatalog {
        var catalog = PlayCatalog()
        let keys: [URLResourceKey] = [.isDirectoryKey, .contentModificationDateKey]
        let urls = (try? fm.contentsOfDirectory(
            at: runsDirectory, includingPropertiesForKeys: keys, options: [.skipsHiddenFiles])) ?? []
        for url in urls {
            guard let values = try? url.resourceValues(forKeys: Set(keys)), values.isDirectory == true else { continue }
            let name = url.lastPathComponent
            let modified = values.contentModificationDate ?? .distantPast
            let steps = checkpointSteps(in: url, fileManager: fm)
            let meta = readJSON(url.appendingPathComponent("meta.json"))
            if let file = PlayWorldFile.read(at: url.appendingPathComponent(PlayLockProbe.playFile)) {
                catalog.worlds.append(PlayWorld(
                    name: name, file: file, newestStep: steps.last, status: meta?["status"] as? String,
                    modified: modified, locked: PlayLockProbe.isHeld(directory: url)))
            } else if steps.isEmpty, meta != nil || fm.fileExists(atPath: url.appendingPathComponent("config.json").path) {
                catalog.unfinished.append(name)
            }
            if !steps.isEmpty {
                let config = readJSON(url.appendingPathComponent("config.json"))
                let world = config?["world"] as? [String: Any]
                let weather = config?["weather"] as? [String: Any]
                catalog.sources.append(PlaySource(
                    name: name, checkpoints: steps, dims: world?["dims"] as? Int,
                    weatherOn: weather?["enabled"] as? Bool ?? false,
                    hasEvents: hasEntries(url.appendingPathComponent("events"), fileManager: fm),
                    modified: modified))
            }
        }
        catalog.worlds.sort { $0.modified > $1.modified }
        catalog.unfinished.sort()
        catalog.sources.sort { $0.modified > $1.modified }
        return catalog
    }

    /// The steps of `checkpoints/step_<t>.safetensors` under `directory`, ascending.
    package static func checkpointSteps(in directory: URL, fileManager fm: FileManager = .default) -> [Int] {
        let names = (try? fm.contentsOfDirectory(atPath: directory.appendingPathComponent("checkpoints").path)) ?? []
        return names.compactMap { name -> Int? in
            guard name.hasPrefix("step_"), name.hasSuffix(".safetensors") else { return nil }
            return Int(name.dropFirst(5).dropLast(12))
        }.sorted()
    }

    private static func hasEntries(_ url: URL, fileManager fm: FileManager) -> Bool {
        ((try? fm.contentsOfDirectory(atPath: url.path)) ?? []).contains { !$0.hasPrefix(".") }
    }

    private static func readJSON(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
}

/// `play.lock`: the advisory `flock` of the process that plays a directory (`runio.PlayLock`).
package enum PlayLockProbe {
    package static let playFile = "play.json"
    package static let lockFile = "play.lock"

    /// Whether another process holds the lock of `directory`. A missing file means nobody does; a lock taken to find out
    /// is dropped with the descriptor. (A stale file left by a crash locks nothing.)
    package static func isHeld(directory: URL) -> Bool {
        let fd = open(directory.appendingPathComponent(lockFile).path, O_RDWR)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        if flock(fd, LOCK_EX | LOCK_NB) == 0 { return false }
        return errno == EWOULDBLOCK
    }

    /// The names among `names` (directories of `runsDirectory`) whose lock is held.
    package static func held(among names: [String], in runsDirectory: URL) -> Set<String> {
        Set(names.filter { isHeld(directory: runsDirectory.appendingPathComponent($0)) })
    }
}

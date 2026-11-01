import Foundation

@testable import AlifeCore

/// Scratch `runs/` folders for the library's tests and measurements: research runs, played worlds, unfinished builds, dangling links.
enum LibraryFixture {
    static let playJSON = """
        {"play_version": 1, "base_rate": 14.4, "cadence": {"metrics_every": 250, "heavy_metrics_every": 250,
         "checkpoint_every": 6000, "keep_checkpoints": 3}, "tick": 25, "weather": {"dc_ref": 0.5},
         "planet": %PLANET%}
        """

    static func write(_ text: String, _ name: String, in dir: URL) throws {
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent(name).deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8)
    }

    static func config(sphere: Bool, dims: Int = 3, weather: Bool = false) -> String {
        let world = sphere
            ? #"{"dims": 3, "geometry": "sphere", "radius": 36, "height": 16, "nside": 8, "layers": 8, "size": 64}"#
            : #"{"dims": \#(dims), "geometry": "box", "size": 64}"#
        return #"{"world": \#(world), "weather": {"enabled": \#(weather)}}"#
    }

    static func setModified(_ url: URL, _ date: Date) throws {
        try FileManager.default.setAttributes([.modificationDate: date], ofItemAtPath: url.path)
    }

    /// A research run with save points.
    @discardableResult
    static func research(_ runs: URL, _ name: String, steps: [Int] = [30_000], status: String = "done", sphere: Bool = true,
                         dims: Int = 3, parent: String? = nil, events: Bool = true) throws -> URL {
        let dir = runs.appendingPathComponent(name)
        var meta = #"{"status": "\#(status)", "start_step": 0"#
        if let parent { meta += #", "parent": "\#(parent)""# }
        try write(meta + "}", "meta.json", in: dir)
        try write(config(sphere: sphere, dims: dims), "config.json", in: dir)
        for step in steps { try write("x", "checkpoints/step_\(String(format: "%010d", step)).safetensors", in: dir) }
        if events { try write("{}", "events/chunk_0.json", in: dir) }
        return dir
    }

    /// A played world, made from `source` (`runs/NAME:STEP`) when given.
    @discardableResult
    static func played(_ runs: URL, _ name: String, source: String? = nil, steps: [Int] = [450_000]) throws -> URL {
        let dir = runs.appendingPathComponent(name)
        let planet = source.map { #"{"source": "\#($0)", "step": 0, "weather_spinup": 1000, "houses": "clade", "made_at_step": 0}"# } ?? "null"
        try write(playJSON.replacingOccurrences(of: "%PLANET%", with: planet), "play.json", in: dir)
        try write(#"{"status": "interrupted", "start_step": 0, "play_build": "planet"}"#, "meta.json", in: dir)
        try write(config(sphere: true, weather: true), "config.json", in: dir)
        for step in steps { try write("x", "checkpoints/step_\(String(format: "%010d", step)).safetensors", in: dir) }
        return dir
    }

    /// What `alife play` leaves when a build stops: the mark, no `play.json`, with or without a first save point.
    @discardableResult
    static func unfinished(_ runs: URL, _ name: String, steps: [Int] = [], status: String = "interrupted", parent: String? = "runs/seed:30000") throws -> URL {
        let dir = runs.appendingPathComponent(name)
        var meta = #"{"status": "\#(status)", "play_build": "planet""#
        if let parent { meta += #", "parent": "\#(parent)""# }
        try write(meta + "}", "meta.json", in: dir)
        try write(config(sphere: true), "config.json", in: dir)
        for step in steps { try write("x", "checkpoints/step_\(String(format: "%010d", step)).safetensors", in: dir) }
        return dir
    }

    /// `count` directories of a plausible library: about a third played worlds (each made from the research run before it), the rest research
    /// runs, an unfinished build for every twentieth, and a folder without a save point for every thirtieth.
    static func many(_ runs: URL, count: Int, now: Date = Date()) throws {
        for i in 0..<count {
            let name = String(format: "world_%03d", i)
            let dir: URL
            if i % 20 == 19 {
                dir = try unfinished(runs, name, steps: i % 40 == 19 ? [] : [100], parent: "runs/world_000:30000")
            } else if i % 30 == 29 {
                dir = runs.appendingPathComponent(name)
                try write("{}", "metrics.jsonl", in: dir)
            } else if i % 3 == 2 {
                dir = try played(runs, name, source: "runs/\(String(format: "world_%03d", i - 1)):30000", steps: [450_000, 456_000])
            } else {
                dir = try research(runs, name, steps: [10_000, 20_000, 30_000], sphere: i % 2 == 0)
            }
            // (whole seconds: Foundation's resource values and `stat` round a fraction differently)
            let when = Date(timeIntervalSince1970: now.timeIntervalSince1970.rounded(.down) - Double(i) * 3600)
            if FileManager.default.fileExists(atPath: dir.appendingPathComponent("checkpoints").path) {
                try setModified(dir.appendingPathComponent("checkpoints"), when)
            }
            try setModified(dir, when)
        }
    }
}

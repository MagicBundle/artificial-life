import Foundation

// The Details page of a world (docs/APP.md §2.4, §3.7): what the world is, how it is doing and what can be done with it, in plain words, with
// the engine's names as secondary text. Pure; the page draws it. Nothing here depends on a run that this app is growing: a world that is not
// growing has its numbers, its settings and its journal here too.

package enum DetailsGuide {
    // MARK: Names

    /// A world's name as a title: the world a planet was grown for is "Grown world of planet-1" (its directory, `planet-1-grow`, is for the
    /// line of small type under the title).
    package static func title(_ name: String) -> String {
        guard name.hasSuffix(SetupPlan.growSuffix), name.count > SetupPlan.growSuffix.count else { return name }
        return "Grown world of " + String(name.dropLast(SetupPlan.growSuffix.count))
    }

    /// The engine's name for the world: the folder under `runs/`, in small type.
    package static func folderLine(_ name: String) -> String { "runs/\(name)" }

    // MARK: What it is

    /// The sentences that say what kind of world this is and what it can do. `grownFor` is the played world a research world was grown for.
    package static func whatItIs(_ row: LibraryRow, grownFor: String?) -> [String] {
        var out: [String] = []
        switch row.kind {
        case .played:
            out.append("A world you play. It runs live in Watch, and you can use powers, read its chronicle and its reports.")
            if let from = row.madeFrom {
                // (a planet made from the world that was grown for it says "its grown world", not "the grown world of" itself)
                let source = NameFolding.same(from.name, row.name + SetupPlan.growSuffix) ? "its grown world" : SetupIntent.shown(from.name)
                out.append("It was made from \(source)" + (from.step.map { " at step \(WorldsGuide.grouped($0))." } ?? "."))
            }
        case .research:
            out.append("A research world. It grows by itself, and you watch it from outside; it cannot be played as it is.")
            if row.shape == .flat {
                out.append(WorldsGuide.flatHasNoPlanet)
            } else if let grownFor {
                out.append("It is the grown world of \(grownFor). To play it, make a planet from it.")
            } else if let unmade = row.unmadePlanet {
                out.append("The planet \(unmade) was not made. Make planet… finishes it.")
            } else {
                out.append("To play it, make a planet from it.")
            }
        }
        if let parent = row.parent {
            out.append("Branched from \(SetupIntent.shown(parent.name))" + (parent.step.map { " at step \(WorldsGuide.grouped($0))." } ?? "."))
        }
        return out
    }

    /// The worlds branched from `name` (their `meta.json` names it as `parent`), in the library's order.
    package static func branches(of name: String, in rows: [LibraryRow]) -> [String] {
        rows.filter { $0.kind == .research && $0.parent.map { NameFolding.same($0.name, name) } == true && $0.name != name }.map(\.name)
    }

    // MARK: How it is doing

    /// The state in a sentence with the step and the time: "Growing now: step 12,300 of 30,000." / "Saved at step 30,000, 2 hours ago."
    package static func progress(_ row: LibraryRow, now: Date) -> String {
        switch row.state {
        case .growing:
            guard let step = row.step else { return "Growing now." }
            if let target = row.targetSteps, target > step { return "Growing now: step \(WorldsGuide.grouped(step)) of \(WorldsGuide.grouped(target))." }
            return "Growing now: step \(WorldsGuide.grouped(step))."
        case .live, .paused:
            return (row.state == .live ? "Open in Watch and running" : "Open in Watch, paused") + (row.step.map { " at step \(WorldsGuide.grouped($0))." } ?? ".")
        case .starting:
            return "Being opened. A planet's weather warm-up takes a while."
        case .saving:
            return "Closed, and writing its last save point."
        case .inUseElsewhere:
            return "Another app or process is using it" + (row.step.map { " (step \(WorldsGuide.grouped($0)))." } ?? ".")
        case .saved, .unfinished:
            guard let step = row.step else { return "It has no save point yet." }
            return "Saved at step \(WorldsGuide.grouped(step)), \(WorldsGuide.relative(row.lastSaved, now: now))."
        }
    }

    // MARK: Words for the numbers

    /// The status the engine wrote in `meta.json`, as a sentence; nil for `running` (the state says it) and for what is not known.
    package static func ended(status: String?, played: Bool = false) -> String? {
        switch status {
        // (a played world is closed at a save point as a matter of course: that is not news)
        case "done" where !played: return "It reached the step it was asked to grow to."
        case "interrupted" where !played: return "It was stopped before its last step, at a save point."
        case "done", "interrupted": return nil
        case "extinct": return "Everything in it died out."
        case "violation": return "The engine's own checks found a problem in it. Check world says more."
        default: return nil
        }
    }

    // MARK: Where the numbers come from

    /// A line under the numbers: when they are from and why they may be missing.
    package static func numbersCaption(hasMetrics: Bool, growing: Bool) -> String {
        if hasMetrics { return growing ? "Read from the world's metrics file every two seconds while it grows." : "The world's last row of numbers." }
        return growing ? "No numbers yet. The first appear a few seconds after the engine starts." : "This world has no numbers file."
    }
}

// MARK: - The settings a world was made with

/// What a world's `config.json` and `meta.json` say it was made with, in the page's words.
package struct MadeWith: Equatable, Sendable {
    package struct Row: Equatable, Sendable, Identifiable {
        package var title: String
        package var value: String
        /// The engine's key, for a tooltip.
        package var key: String
        package var id: String { key }
    }

    package var seed: Int?
    /// The step it was asked to grow to (`run.steps`).
    package var steps: Int?
    package var created: Date?
    /// "Off" or "Gardener, every 1,000 steps, with a language model every 25,000".
    package var autopilot: String
    /// The shape and size, in words.
    package var shape: String
    package var settings: [Row]

    package static func read(runDirectory: URL) -> MadeWith? {
        guard let data = try? Data(contentsOf: runDirectory.appendingPathComponent("config.json")),
              let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        let run = root["run"] as? [String: Any] ?? [:]
        let world = root["world"] as? [String: Any] ?? [:]
        func int(_ v: Any?) -> Int? {
            guard let n = v as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
            return Int(exactly: n.doubleValue)
        }
        var made = MadeWith(seed: int(run["seed"]), steps: int(run["steps"]), created: nil, autopilot: "Off", shape: "", settings: [])
        let dims = int(world["dims"]) ?? 2
        let geometry = world["geometry"] as? String
        let size = (world["size"] as? NSNumber)?.doubleValue
        switch WorldShape.of(dims: dims, geometry: geometry) {
        case .planet?:
            made.shape = "Planet" + ((world["radius"] as? NSNumber).map { ", radius \(Self.short($0.doubleValue))" } ?? "")
        case .box?:
            made.shape = "3D box" + (size.map { ", \(Self.short($0)) wide" } ?? "") + ((world["height"] as? NSNumber).map { ", \(Self.short($0.doubleValue)) high" } ?? "")
        case .flat?, nil:
            made.shape = "Flat" + (size.map { ", \(Self.short($0)) wide" } ?? "")
        }
        if let meta = (try? JSONSerialization.jsonObject(with: Data(contentsOf: runDirectory.appendingPathComponent("meta.json")))) as? [String: Any],
           let stamp = meta["created_utc"] as? String {
            made.created = ISO8601DateFormatter().date(from: stamp)
        }
        if let source = try? DirectorSource(configJSON: data), source.enabled || source.settings.mode != .off {
            made.autopilot = Self.autopilot(source.settings)
        }
        let levers = Levers.forkBaseline(runDirectory: runDirectory).levers
        var keys = LeverKeys.dynamics.map(\.key)
        if dims == 3 { keys += LeverKeys.world3D.map(\.key) }
        if geometry == "sphere" { keys += LeverKeys.sphere.map(\.key) }
        for key in keys {
            guard let title = LeverKeys.title(key), let text = LeverKeys.text(key, in: levers), !text.isEmpty else { continue }
            made.settings.append(Row(title: title, value: text, key: key))
        }
        return made
    }

    private static func short(_ value: Double) -> String { value == value.rounded() ? String(Int(value)) : String(format: "%g", value) }

    package static func autopilot(_ s: DirectorSettings) -> String {
        var line = s.persona.label
        if let rule = Parse.integer(s.ruleEvery) { line += ", deciding every \(WorldsGuide.grouped(rule)) steps" }
        if s.mode == .replay { line += ", replaying its log" }
        if s.llmOn, let every = Parse.integer(s.llmEvery) { line += ", with a local model every \(WorldsGuide.grouped(every)) steps" }
        return line
    }
}

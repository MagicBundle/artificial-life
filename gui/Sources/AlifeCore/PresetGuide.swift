import Foundation

// MARK: - The preset guide

/// What the Lab's preset guide says about a preset (DESIGN §27.14): a short description for someone who does not know the
/// engine, the numbers of the world read from the preset's own file, and the snapshot of a real run with what it cost.
///
/// Nothing numeric is typed here. The shape and size of the world and the population limits come from the preset file
/// (`PresetFacts`, through `PresetFile`, the engine default where the file sets nothing: the `64k` file is empty), and the speed, the step and
/// the population of the snapshot from `gui/PresetSnapshots/manifest.json`, which `gui/scripts/make-preset-snapshots.sh` writes from real runs.
/// The words below say only what the preset files and DESIGN state; `PresetGuideTests` holds them to the files.

/// The three kinds of world, as the compare sheet groups them.
package enum PresetGroup: CaseIterable, Identifiable, Sendable {
    case flat, box3D, planet

    package var id: Self { self }

    package var title: String {
        switch self {
        case .flat: return "Flat worlds"
        case .box3D: return "3D box worlds"
        case .planet: return "Planets"
        }
    }

    /// One line under the group's title.
    package var blurb: String {
        switch self {
        case .flat: return "A square sheet whose edges wrap round, seen from above."
        case .box3D: return "A water column with a floor and a surface, with gravity."
        case .planet: return "A shell round a small planet, with gravity towards its centre."
        }
    }

    package var presets: [Preset] { Preset.allCases.filter { $0.group == self } }
}

/// The words about one preset.
package struct PresetGuide: Equatable, Sendable {
    /// What the world is.
    package var what: String
    /// What it is good for.
    package var goodFor: String
    /// What it costs, in words; the measured speed is the manifest's.
    package var cost: String
}

extension Preset {
    package var group: PresetGroup {
        switch self {
        case .tiny, .dev, .k64, .k256: return .flat
        case .tiny3d, .k64_3d, .mini3d: return .box3D
        case .tiny3s, .mini3s, .k64_3s: return .planet
        }
    }

    /// The description. Every statement is in the preset's TOML file, in DESIGN, README or the docs of the results (the source is named in
    /// the comment), or is a comparison of the speeds and memory measured for `gui/PresetSnapshots/manifest.json` (the comparisons are
    /// marked "measured": they were true of the M5 Pro run that made the snapshots).
    package var guide: PresetGuide {
        switch self {
        case .tiny:
            // tiny.toml: "tests and CI", steps = 500. Measured: the fastest and the smallest of the ten.
            return PresetGuide(
                what: "The smallest flat world: a small square whose edges wrap round, with a few hundred organisms to start.",
                goodFor: "Tests, checking that something works, trying a lever. Its own default run is only 500 steps.",
                cost: "The cheapest of the ten: the fastest per step and the least memory.")
        case .dev:
            // dev.toml: "interactive development".
            return PresetGuide(
                what: "A mid-size flat world, the one used for interactive development, with room for thousands of organisms.",
                goodFor: "Watching groups and bodies form, and short experiments, without waiting.",
                cost: "Light: several times faster per step than 64k.")
        case .k64:
            // 64k.toml: "the dataclass defaults"; README: "the target configuration"; RunConfig.steps = 100 000; docs/RESULTS_3D.md: runs/64k_s0.
            return PresetGuide(
                what: "The engine's default flat world and its target size: tens of thousands of organisms on a wide wrapped square.",
                goodFor: "Real flat-world experiments. A run is 100 000 steps by default, and the long runs in the docs were made at this size.",
                cost: "Heavy: roughly a quarter of dev's speed per step.")
        case .k256:
            // 256k.toml: "large runs"; 4 x the area and 4 x n_max of 64k.
            return PresetGuide(
                what: "The largest world: four times the area and four times the organisms of the 64k preset.",
                goodFor: "Large runs, when 64k is too small.",
                cost: "By far the heaviest: several times slower per step than 64k, and nearly four times its memory.")
        case .tiny3d:
            // tiny3d.toml: "the 3D tiny world for tests, CI and first looks"; "plus a floor, a surface at height 24, gravity, a buoyancy actuator and a vertical food profile".
            return PresetGuide(
                what: "The smallest 3D world: a block of water with a floor and a surface. Gravity pulls organisms down; they swim and can float. Food is richest on the floor, with a thinner band under the surface.",
                goodFor: "Tests and a first look at life in 3D. Its file marks its values as provisional.",
                cost: "Very cheap, nearly as fast as tiny.")
        case .mini3d:
            // mini3d.toml: "a quarter of 64k3d's area at the same density"; DESIGN 37.1: the storm160 overlay is composed on mini3d.
            return PresetGuide(
                what: "A quarter of the 64k3d world's area, with the same physics, the same food and the same density of organisms.",
                goodFor: "3D experiments that need the full 3D ecology without the cost of the big world. The storm160 weather overlay is made for it.",
                cost: "Moderate: close to four times faster per step than 64k3d.")
        case .k64_3d:
            // 64k3d.toml: "the 3D world at 64k scale", a 320 x 320 x 80 water column; docs/RESULTS_3D.md: the first long 3D run.
            return PresetGuide(
                what: "The full-size 3D world: the tiny3d rules (gravity, buoyancy, food on the floor and under the surface) in a large water column.",
                goodFor: "Long 3D runs. The first long 3D run in the docs was made in it.",
                cost: "Heavy: a little slower per step than 64k, which has the same organism limit.")
        case .tiny3s:
            // tiny3s.toml: "the tiny sphere world for tests, CI and first looks"; "Everything else is tiny3d's"; the same volume as tiny3d.
            return PresetGuide(
                what: "The smallest planet: a thin shell round a small ball, with gravity towards its centre. The planet counterpart of tiny3d, with the same volume and the same rules for organisms.",
                goodFor: "Tests and a first look at a round world, which has no edges to wrap.",
                cost: "Very cheap, a little slower per step than tiny3d.")
        case .mini3s:
            // mini3s.toml: "the sphere analogue of mini3d ... The per-agent physics and economy are mini3d's".
            return PresetGuide(
                what: "A mid-size planet: the round counterpart of mini3d, with its per-organism rules.",
                goodFor: "Planet experiments without the cost of the big planet, and comparing a round world with a box one.",
                cost: "Moderate: slower per step than mini3d, at about three quarters of its speed.")
        case .k64_3s:
            // 64k3s.toml: "the 64k sphere world, the sphere analogue of 64k3d"; "The weather is the `stormsph` overlay".
            return PresetGuide(
                what: "The full-size planet: the round counterpart of 64k3d, with the same organism limit.",
                goodFor: "Long runs on a planet. The stormsph weather overlay is made for it.",
                cost: "Heavy: a little slower per step than 64k3d.")
        }
    }
}

// MARK: - The numbers, from the preset file

/// The shape and the size of a preset's world and its population limits, read from `src/alife/presets/NAME.toml` (`PresetFile`).
package struct PresetFacts: Equatable, Sendable {
    package enum Shape: Equatable, Sendable {
        /// A square sheet (`world.dims = 2`).
        case flat
        /// A water column (`world.dims = 3`, `world.geometry = "box"`).
        case box3D
        /// A shell round a planet (`world.geometry = "sphere"`).
        case planet

        package var group: PresetGroup {
            switch self {
            case .flat: return .flat
            case .box3D: return .box3D
            case .planet: return .planet
            }
        }
    }

    package var shape: Shape
    /// `world.size`: the side of the square (a planet's image length, which is no size of the planet: not shown for one).
    package var size: Double
    /// `world.height`: the water column's height, or the planet's shell thickness; 0 in a flat world.
    package var height: Double
    /// `world.radius`: the planet's radius; 0 elsewhere.
    package var radius: Double
    /// `population.n_max`: the most organisms alive at once.
    package var nMax: Int
    /// `population.n_init`: the organisms a run starts with.
    package var nInit: Int

    /// The engine's defaults (`WorldConfig` and `PopulationConfig` in src/alife/config.py; the `64k` preset file sets none of them).
    /// `PresetGuideTests` compares them with that file.
    package static let engineDefaults = PresetFacts(shape: .flat, size: 1024, height: 0, radius: 0, nMax: 65_536, nInit: 16_384)

    package init(shape: Shape, size: Double, height: Double, radius: Double, nMax: Int, nInit: Int) {
        self.shape = shape
        self.size = size
        self.height = height
        self.radius = radius
        self.nMax = nMax
        self.nInit = nInit
    }

    /// The facts of a preset file; a key it does not set is the engine default.
    package init(file: PresetFile) {
        let d = Self.engineDefaults
        if file.string("world", "geometry") == "sphere" {
            shape = .planet
        } else if (file.int("world", "dims") ?? 2) == 3 {
            shape = .box3D
        } else {
            shape = .flat
        }
        size = file.double("world", "size") ?? d.size
        height = file.double("world", "height") ?? d.height
        radius = file.double("world", "radius") ?? d.radius
        nMax = file.int("population", "n_max") ?? d.nMax
        nInit = file.int("population", "n_init") ?? d.nInit
    }

    /// The facts of the named preset in the repository, or nil if its file cannot be read.
    package static func preset(_ preset: Preset, repo: URL) -> PresetFacts? {
        PresetFile.preset(preset.rawValue, repo: repo).map(PresetFacts.init(file:))
    }

    /// An integer with thousands set apart by a thin space, whatever the locale ("65 536").
    package static func count(_ n: Int) -> String {
        let digits = String(abs(n))
        var out = ""
        for (i, c) in digits.reversed().enumerated() {
            if i > 0, i % 3 == 0 { out.append("\u{2009}") }
            out.append(c)
        }
        return (n < 0 ? "-" : "") + String(out.reversed())
    }

    /// "128 x 128", "320 x 320 x 80 high", "radius 90, shell 40 thick".
    package var worldText: String {
        let x = "\u{2009}\u{00D7}\u{2009}"
        let side = World3DLevers.text(size), up = World3DLevers.text(height)
        switch shape {
        case .flat: return "\(side)\(x)\(side)"
        case .box3D: return "\(side)\(x)\(side)\(x)\(up) high"
        case .planet: return "radius \(World3DLevers.text(radius)), shell \(up) thick"
        }
    }

    /// "up to 1 024 organisms, 256 at the start".
    package var populationText: String {
        "up to \(Self.count(nMax)) organisms, \(Self.count(nInit)) at the start"
    }

    package var shapeName: String {
        switch shape {
        case .flat: return "Flat world"
        case .box3D: return "3D box world"
        case .planet: return "Planet"
        }
    }
}

// MARK: - The snapshots and their manifest

/// `gui/PresetSnapshots/manifest.json`: what each snapshot is, written by `gui/scripts/make-preset-snapshots.sh`.
package struct PresetSnapshotManifest: Equatable, Sendable, Decodable {
    package struct Entry: Equatable, Sendable, Decodable {
        package var preset: String
        package var seed: Int
        /// The step of the frame shown.
        package var step: Int
        /// The organisms alive in that frame.
        package var population: Int
        /// Steps per second of the engine at that population, measured alone on `machine`.
        package var stepsPerSecond: Double
        /// `git rev-parse HEAD` of the engine that made the run.
        package var gitRev: String
        /// The day the run was made (UTC, `YYYY-MM-DD`).
        package var date: String
        /// What the picture shows of the world: "the whole world", or "a 512 x 320 window of typical density" for a large flat one.
        package var view: String?

        private enum CodingKeys: String, CodingKey {
            case preset, seed, step, population, stepsPerSecond = "steps_per_s", gitRev = "git_rev", date, view
        }
    }

    /// The Mac the speeds were measured on ("Apple M5 Pro").
    package var machine: String
    package var snapshots: [Entry]

    package func entry(for preset: Preset) -> Entry? { snapshots.first { $0.preset == preset.rawValue } }

    package static func url(repo: URL) -> URL { repo.appendingPathComponent("gui/PresetSnapshots/manifest.json") }

    package static func imageURL(_ preset: Preset, repo: URL) -> URL {
        repo.appendingPathComponent("gui/PresetSnapshots/\(preset.rawValue).jpg")
    }

    /// The manifest of the repository; nil when it is missing or unreadable (the guide then shows no snapshot).
    package static func load(repo: URL) -> PresetSnapshotManifest? {
        guard let data = try? Data(contentsOf: url(repo: repo)) else { return nil }
        return try? JSONDecoder().decode(PresetSnapshotManifest.self, from: data)
    }
}

/// Everything the preset guide shows for one preset.
package struct PresetGuideEntry: Equatable, Sendable {
    package var preset: Preset
    package var facts: PresetFacts?
    package var snapshot: PresetSnapshotManifest.Entry?
    package var imageURL: URL

    package init(preset: Preset, facts: PresetFacts?, snapshot: PresetSnapshotManifest.Entry?, imageURL: URL) {
        self.preset = preset
        self.facts = facts
        self.snapshot = snapshot
        self.imageURL = imageURL
    }

    package var guide: PresetGuide { preset.guide }

    /// "A real run, seed 1, step 30 000, 21 400 alive in the whole world, shown whole." (nil without a snapshot)
    package func snapshotCaption() -> String? {
        guard let s = snapshot else { return nil }
        let shown = s.view.map { " Shows \($0)." } ?? ""
        return "A real run, seed \(s.seed), step \(PresetFacts.count(s.step)), \(PresetFacts.count(s.population)) alive.\(shown)"
    }

    /// "41 steps/s, measured on an Apple M5 Pro." (nil without a snapshot)
    package func speedCaption(machine: String) -> String? {
        guard let s = snapshot else { return nil }
        let rate = s.stepsPerSecond
        let text = String(format: rate >= 10 ? "%.0f" : "%.1f", rate)
        return "\(text) steps/s, measured on \(machine.isEmpty ? "the Mac that made the snapshots" : machine)."
    }
}

/// The guide's content for a repository: the entries in the compare sheet's order, loaded once.
package struct PresetGuideContent: Equatable, Sendable {
    package var machine: String
    package var entries: [PresetGuideEntry]

    package static let empty = PresetGuideContent(machine: "", entries: [])

    package init(machine: String, entries: [PresetGuideEntry]) {
        self.machine = machine
        self.entries = entries
    }

    package static func load(repo: URL) -> PresetGuideContent {
        let manifest = PresetSnapshotManifest.load(repo: repo)
        let entries = Preset.allCases.map {
            PresetGuideEntry(preset: $0, facts: PresetFacts.preset($0, repo: repo), snapshot: manifest?.entry(for: $0),
                             imageURL: PresetSnapshotManifest.imageURL($0, repo: repo))
        }
        return PresetGuideContent(machine: manifest?.machine ?? "", entries: entries)
    }

    package func entry(for preset: Preset) -> PresetGuideEntry? { entries.first { $0.preset == preset } }
    package func entries(in group: PresetGroup) -> [PresetGuideEntry] { entries.filter { $0.preset.group == group } }
}

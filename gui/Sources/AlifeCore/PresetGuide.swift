import Foundation

// MARK: - The preset guide

/// What the Lab's preset guide says about a preset (DESIGN §27.14): a short description for someone who does not know the
/// engine, the numbers of the world read from the preset's own file, and the snapshot of a real run with what it cost.
///
/// What is held to the data, and by what:
/// - The numbers of the world (shape, size, height, radius, `n_max`, `n_init`) are not typed here: they come from the preset file
///   (`PresetFacts`, through `PresetFile`; the engine default where the file sets nothing, the `64k` file being empty).
///   `PresetGuideTests` compares them with the files.
/// - Every comparison of speed or memory ("several times faster than 64k") is data (`PresetComparison`: the two presets, the metric and
///   the band its ratio must lie in), its sentence is built from that data, and `PresetGuideTests` fails when the committed
///   `gui/PresetSnapshots/manifest.json` no longer supports one (a remade snapshot, another machine).
/// - The rest of the words state what the preset files, DESIGN, README and the results docs state (the source is named beside each
///   text). No test holds them: they are checked by reading, and `PresetGuideTests` only checks that each is there.

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
        case .flat: return "A square sheet seen from above. Its edges wrap round: an organism that leaves one edge comes back at the opposite one."
        case .box3D: return "A block of water with a floor, a surface and gravity, seen from outside."
        case .planet: return "A thin layer round a small planet, with gravity towards its centre. A picture shows one side of the planet."
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
    /// What it costs: a lead in words, then the sentences of its comparisons (`PresetComparison`, `PresetExtreme`).
    package var cost: String
}

// MARK: - Comparisons held to the manifest

/// A statement that one preset is faster than, slower than or heavier than another, as data: `preset`'s value over `reference`'s, taken from
/// the manifest, must lie in `band`. The sentence is the words for that band.
package struct PresetComparison: Equatable, Sendable {
    package enum Metric: Equatable, Sendable { case speed, memory }
    package var preset: Preset
    package var reference: Preset
    package var metric: Metric
    package var band: ClosedRange<Double>
    package var sentence: String

    /// `preset` over `reference` in the manifest; nil when either has no value.
    package func ratio(in manifest: PresetSnapshotManifest) -> Double? {
        guard let a = manifest.entry(for: preset), let b = manifest.entry(for: reference) else { return nil }
        switch metric {
        case .speed: return b.stepsPerSecond > 0 ? a.stepsPerSecond / b.stepsPerSecond : nil
        case .memory:
            guard let m = a.peakMemoryBytes, let n = b.peakMemoryBytes, n > 0 else { return nil }
            return Double(m) / Double(n)
        }
    }

    package func holds(in manifest: PresetSnapshotManifest) -> Bool { ratio(in: manifest).map(band.contains) ?? false }
}

/// A statement that a preset has the highest speed or the lowest memory of all the presets that have a snapshot (`Preset.unmeasured` have none).
package struct PresetExtreme: Equatable, Sendable {
    package enum Kind: Equatable, Sendable { case fastest, lightest }
    package var preset: Preset
    package var kind: Kind
    package var sentence: String

    package func holds(in manifest: PresetSnapshotManifest) -> Bool {
        guard let mine = manifest.entry(for: preset) else { return false }
        let others = manifest.snapshots.filter { $0.preset != preset.rawValue }
        guard others.count == Preset.measured.count - 1 else { return false }
        switch kind {
        case .fastest: return others.allSatisfy { mine.stepsPerSecond > $0.stepsPerSecond }
        case .lightest:
            guard let m = mine.peakMemoryBytes else { return false }
            return others.allSatisfy { $0.peakMemoryBytes.map { m < $0 } ?? false }
        }
    }
}

extension Preset {
    /// Every comparison the words make, with the band the manifest must keep it in. The bands are the width of the wording ("several
    /// times", "a little"), and are wide enough for a rerun of the same runs and narrow enough to fail on a different picture of the world.
    package static let comparisons: [PresetComparison] = [
        PresetComparison(preset: .dev, reference: .k64, metric: .speed, band: 3.0...4.5, sentence: "Several times faster than 64k."),
        PresetComparison(preset: .k64, reference: .dev, metric: .speed, band: 0.20...0.35, sentence: "About a quarter of dev's speed."),
        PresetComparison(preset: .k256, reference: .k64, metric: .speed, band: 0.10...0.34, sentence: "Several times slower than 64k."),
        PresetComparison(preset: .k256, reference: .k64, metric: .memory, band: 3.4...4.0, sentence: "Nearly four times the memory of 64k."),
        PresetComparison(preset: .tiny3d, reference: .tiny, metric: .speed, band: 0.80...0.95, sentence: "Nearly as fast as tiny."),
        PresetComparison(preset: .mini3d, reference: .k64_3d, metric: .speed, band: 3.5...4.2, sentence: "Close to four times faster than 64k3d."),
        PresetComparison(preset: .k64_3d, reference: .k64, metric: .speed, band: 0.70...0.90, sentence: "Somewhat slower than 64k, which has room for the same number of organisms."),
        PresetComparison(preset: .tiny3s, reference: .tiny3d, metric: .speed, band: 0.70...0.90, sentence: "A little slower than tiny3d."),
        PresetComparison(preset: .mini3s, reference: .mini3d, metric: .speed, band: 0.70...0.80, sentence: "About three quarters of mini3d's speed."),
        PresetComparison(preset: .k64_3s, reference: .k64_3d, metric: .speed, band: 0.80...0.95, sentence: "A little slower than 64k3d."),
    ]

    package static let extremes: [PresetExtreme] = [
        PresetExtreme(preset: .tiny, kind: .fastest, sentence: "The fastest of the ten that have been measured."),
        PresetExtreme(preset: .tiny, kind: .lightest, sentence: "It needs the least memory of the ten that have been measured."),
    ]
}

extension Preset {
    /// The presets with no snapshot, no speed and no memory: the extra-large planets, which are not calibrated and have not been run for the
    /// guide. They have no `PresetComparison` (a band would be invented); the guide says in words that they are heavier and shows no figure.
    package static let unmeasured: Set<Preset> = [.k128_3s, .k256_3s, .k576_3s]

    /// Every other preset: the ones the manifest must have an entry and a picture for.
    package static var measured: [Preset] { allCases.filter { !unmeasured.contains($0) } }

    package var group: PresetGroup {
        switch self {
        case .tiny, .dev, .k64, .k256: return .flat
        case .tiny3d, .k64_3d, .mini3d: return .box3D
        case .tiny3s, .mini3s, .k64_3s, .k128_3s, .k256_3s, .k576_3s: return .planet
        }
    }

    /// The description, in words for someone who does not know the engine. The source of each is named in the comment. The sentences about
    /// speed and memory are the comparisons above, appended to the lead of the cost.
    package var guide: PresetGuide {
        let lead: String, what: String, goodFor: String
        switch self {
        case .tiny:
            // tiny.toml: "tests and CI"; run.steps = 500.
            what = "The smallest flat world: a small square sheet whose edges wrap round, with a few hundred organisms to start."
            goodFor = "Tests, checking that something works, trying out a setting. Its own default run is only 500 steps."
            lead = "Very cheap to run."
        case .dev:
            // dev.toml: "interactive development"; README: organisms bond into bodies.
            what = "A mid-size flat world, the one used for interactive development, with room for thousands of organisms."
            goodFor = "Watching organisms cluster and bond together, and short experiments without a long wait."
            lead = "Light."
        case .k64:
            // 64k.toml: "the dataclass defaults"; README: "the target configuration"; RunConfig.steps = 100 000; docs/RESULTS_3D.md: runs/64k_s0.
            what = "The engine's default flat world, and the size it is designed to run well at: tens of thousands of organisms on a wide square."
            goodFor = "Real flat-world experiments. A run lasts 100 000 steps by default, and the long runs in the docs were made at this size."
            lead = "Heavy."
        case .k256:
            // 256k.toml: "large runs"; 4 x the area and 4 x n_max of 64k.
            what = "The largest flat world: four times the area and four times the organisms of the 64k world."
            goodFor = "Large runs, when 64k is too small."
            lead = "By far the heaviest."
        case .tiny3d:
            // tiny3d.toml: "the 3D tiny world for tests, CI and first looks"; floor, surface, gravity, buoyancy actuator, benthic + photic (amp 0.5) food.
            what = "The smallest 3D world: a small block of water with a floor below and a surface above. Gravity pulls organisms down; they swim, and they can float up against gravity. Food is richest on the floor, with a weaker band just under the surface."
            goodFor = "Tests and a first look at life in 3D. Its file marks its values as provisional (not yet tuned)."
            lead = "Very cheap."
        case .mini3d:
            // mini3d.toml: "a quarter of 64k3d's area at the same density"; DESIGN 37.1: the storm160 overlay is composed on mini3d.
            what = "A quarter of the area of 64k3d, the full-size 3D world, with the same physics, the same food and the same density of organisms."
            goodFor = "3D experiments that need the full 3D ecology without the cost of the big world. An optional storm-weather extra, storm160, is made for it (it is not part of this preset)."
            lead = "Moderate."
        case .k64_3d:
            // 64k3d.toml: "the 3D world at 64k scale", a water column; docs/RESULTS_3D.md: the first long 3D run.
            what = "The full-size 3D world: a large block of water with a floor and a surface, with the same kinds of rules as tiny3d (gravity, floating, food richest on the floor)."
            goodFor = "Long 3D runs. The first long 3D run in the docs was made in it."
            lead = "Heavy."
        case .tiny3s:
            // tiny3s.toml: "the tiny sphere world for tests, CI and first looks"; "Everything else is tiny3d's"; the same volume as tiny3d.
            what = "The smallest planet: organisms live in a thin layer between a small planet's surface and a ceiling above it, and gravity pulls them towards its centre. It is the planet version of tiny3d, with the same volume and the same rules for organisms."
            goodFor = "Tests and a first look at a round world, which has no edges to wrap."
            lead = "Very cheap."
        case .mini3s:
            // mini3s.toml: "the sphere analogue of mini3d ... The per-agent physics and economy are mini3d's".
            what = "A mid-size planet: the planet version of mini3d, with the same physics and energy costs for each organism."
            goodFor = "Planet experiments without the cost of the big planet, and comparing a round world with a box one."
            lead = "Moderate."
        case .k64_3s:
            // 64k3s.toml: "the 64k sphere world, the sphere analogue of 64k3d"; n_max equal to 64k3d's; "The weather is the `stormsph` overlay".
            what = "The full-size planet: the planet version of 64k3d, with room for the same number of organisms."
            goodFor = "Long runs on a planet. An optional storm-weather extra, stormsph, is made for it (it is not part of this preset)."
            lead = "Heavy."
        case .k128_3s:
            // 128k3s.toml: "64k3s scaled by floor area at equal density"; radius 128.6, n_max 131 072; "NOT CALIBRATED"; no stormsph overlay is made for it.
            what = "An extra-large planet: about twice the area of 64k3s, with room for twice as many organisms."
            goodFor = "Not yet calibrated: a test of how large a planet your Mac can run. No storm weather is made for it, so it is for research."
            lead = "Heavier than 64k3s. It has not been measured: a larger planet runs slower in proportion to the number of organisms it holds."
        case .k256_3s:
            // 256k3s.toml: "64k3s scaled by floor area at equal density"; radius 180, n_max 262 144; "NOT CALIBRATED".
            what = "A very large planet: four times the area of 64k3s, with room for four times as many organisms."
            goodFor = "Not yet calibrated: a test of how large a planet your Mac can run. No storm weather is made for it, so it is for research."
            lead = "Much heavier than 64k3s. It has not been measured: a larger planet runs slower in proportion to the number of organisms it holds."
        case .k576_3s:
            // 576k3s.toml: "64k3s scaled by floor area at equal density"; radius 270, n_max 589 824; "NOT CALIBRATED".
            what = "The largest planet: nine times the area of 64k3s, with room for nine times as many organisms."
            goodFor = "Not yet calibrated: a test of how large a planet your Mac can run. No storm weather is made for it, so it is for research."
            lead = "By far the heaviest planet. It has not been measured: a larger planet runs slower in proportion to the number of organisms it holds, and this one may be too large to play."
        }
        let sentences = Self.extremes.filter { $0.preset == self }.map(\.sentence) + Self.comparisons.filter { $0.preset == self }.map(\.sentence)
        return PresetGuide(what: what, goodFor: goodFor, cost: ([lead] + sentences).joined(separator: " "))
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
            if i > 0, i % 3 == 0 { out.append("\u{202F}") }  // a narrow no-break space: a number is never split over two lines
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

    /// What the number in a name like "64k" means, when it is what this preset's file says: "64k" means room for 65 536 organisms (64 × 1 024).
    /// nil for a name without one, and when the file's `n_max` is not that many (then the note would be false).
    package static func nameNote(_ preset: Preset, nMax: Int) -> String? {
        let name = preset.rawValue
        guard let k = Int(name.prefix(while: \.isNumber)), name.dropFirst(String(k).count).first == "k", nMax == k * 1024 else { return nil }
        return "\u{201C}\(String(k))k\u{201D} means room for \(count(nMax)) organisms (\(k) \u{00D7} 1\u{202F}024)."
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
        /// The organisms alive in that frame, in the whole world.
        package var population: Int
        /// Of those, the ones inside the window of a large flat world's picture (nil when the picture is not a window).
        package var windowPopulation: Int?
        /// What the picture shows: "the whole world", "the whole box", "one side of the planet", or "a 512 × 320 window of the world, of typical density".
        package var view: String?
        /// Steps per second of the engine at that population, measured alone on `machine`.
        package var stepsPerSecond: Double
        /// The peak memory of the process that measured the speed (GPU buffers included), approximate.
        package var peakMemoryBytes: Int?
        /// `git rev-parse HEAD` of the engine that made the run.
        package var gitRev: String
        /// The day the run was made (UTC, `YYYY-MM-DD`).
        package var date: String

        private enum CodingKeys: String, CodingKey {
            case preset, seed, step, population, windowPopulation = "window_population", view, stepsPerSecond = "steps_per_s",
                 peakMemoryBytes = "peak_memory_bytes", gitRev = "git_rev", date
        }
    }

    /// The Mac the speeds were measured on ("Apple M5 Pro").
    package var machine: String
    /// Whether the working tree had uncommitted changes (outside the snapshots) when the manifest was made.
    package var gitDirty: Bool?
    package var snapshots: [Entry]

    private enum CodingKeys: String, CodingKey { case machine, gitDirty = "git_dirty", snapshots }

    package init(machine: String, gitDirty: Bool?, snapshots: [Entry]) {
        self.machine = machine
        self.gitDirty = gitDirty
        self.snapshots = snapshots
    }

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

    /// What "64k" and the like mean in this preset's name (nil for the other names).
    package var nameNote: String? { facts.flatMap { PresetFacts.nameNote(preset, nMax: $0.nMax) } }

    /// "A real run (seed 1) at step 30 000: 116 382 organisms alive in the whole world. The picture shows a 512 × 320 window of the world, of
    /// typical density, with 3 912 of them in it." (nil without a snapshot)
    package func snapshotCaption() -> String? {
        guard let s = snapshot else { return nil }
        var text = "A real run (seed \(s.seed)) at step \(PresetFacts.count(s.step)): \(PresetFacts.count(s.population)) organisms alive in the whole world."
        if let view = s.view {
            text += " The picture shows \(view)"
            if let n = s.windowPopulation { text += ", with \(PresetFacts.count(n)) of them in it" }
            text += "."
        }
        return text
    }

    /// "445 steps/s with this many organisms alive, measured on Apple M5 Pro." (nil without a snapshot)
    package func speedCaption(machine: String) -> String? {
        guard let s = snapshot else { return nil }
        let rate = s.stepsPerSecond
        let text = String(format: rate >= 10 ? "%.0f" : "%.1f", rate)
        return "\(text) steps/s with this many organisms alive, measured on \(machine.isEmpty ? "the Mac that made the snapshots" : machine)."
    }

    /// "Peak memory about 8.7 GB (approximate)." (nil when the manifest has none)
    package func memoryCaption() -> String? {
        guard let bytes = snapshot?.peakMemoryBytes else { return nil }
        return "Peak memory about \(Self.memoryText(bytes)) (approximate)."
    }

    /// 8 970 000 000 -> "9.0 GB"; 250 000 000 -> "250 MB".
    package static func memoryText(_ bytes: Int) -> String {
        if bytes >= 1_000_000_000 { return String(format: "%.1f GB", Double(bytes) / 1e9) }
        return "\(Int((Double(bytes) / 1e7).rounded()) * 10) MB"
    }

    /// What the picture is, for a screen reader.
    package var imageLabel: String {
        guard let s = snapshot else { return "No picture of \(preset.rawValue) is available" }
        return "A frame of a real \(preset.rawValue) run at step \(PresetFacts.count(s.step)). It shows \(s.view ?? "the world")."
    }

    /// The whole row for a screen reader: everything the row shows, in reading order.
    package func spokenText(machine: String) -> String {
        var parts = [preset.rawValue]
        if let facts { parts += ["\(facts.shapeName), \(facts.worldText).", facts.populationText + "."] }
        if let nameNote { parts.append(nameNote) }
        parts += [guide.what, "Good for: \(guide.goodFor)", "Cost: \(guide.cost)"]
        parts += [snapshotCaption(), speedCaption(machine: machine), memoryCaption()].compactMap { $0 }
        if snapshot == nil { parts.append("No snapshot has been made for this preset.") }
        return parts.joined(separator: " ")
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

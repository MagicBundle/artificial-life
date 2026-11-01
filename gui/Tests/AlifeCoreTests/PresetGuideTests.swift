import Foundation
import ImageIO
import Testing

@testable import AlifeCore

/// The preset guide (`PresetGuide.swift`): that every engine preset has a guide entry, that the numbers it shows are what the preset files say
/// (read here by a second, independent reader of the raw text, with the defaults taken from `config.py`), and that the committed snapshots
/// and their manifest are complete.
@Suite struct PresetGuideTests {
    private static let config = try! String(contentsOf: Repo.root.appendingPathComponent("src/alife/config.py"), encoding: .utf8)
    private static let sphereNames: Set<String> = ["tiny3s", "mini3s", "64k3s"]

    /// The names of `PRESETS: tuple[str, ...] = (...)` in config.py.
    private static func enginePresets() -> [String] {
        guard let start = config.range(of: "PRESETS: tuple[str, ...] = (") else { return [] }
        let rest = config[start.upperBound...]
        guard let end = rest.range(of: ")") else { return [] }
        return rest[..<end.lowerBound].split(whereSeparator: { $0 == "," || $0.isNewline })
            .map { $0.trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "\"")) }
            .filter { !$0.isEmpty }
    }

    /// `name: type = value` of a dataclass line of config.py ("n_max: int = 65536  # N rows"), the first one in the class `cls`.
    private static func engineDefault(_ cls: String, _ name: String) -> Double? {
        guard let at = config.range(of: "class \(cls):") else { return nil }
        for line in config[at.upperBound...].split(separator: "\n").prefix(40) {
            let t = line.trimmingCharacters(in: .whitespaces)
            guard t.hasPrefix("\(name):"), let eq = t.firstIndex(of: "=") else { continue }
            let value = t[t.index(after: eq)...].split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false)[0]
                .trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "_", with: "")
            return Double(value)
        }
        return nil
    }

    /// The number set by `key = value` in the table `table` of the raw preset text, by a plain line scan: nil when the file does not set it.
    private static func rawValue(_ text: String, table: String, key: String) -> Double? {
        var current = ""
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("[") { current = String(line.dropFirst().prefix(while: { $0 != "]" })); continue }
            guard current == table, line.hasPrefix(key), let eq = line.firstIndex(of: "=") else { continue }
            guard line[line.startIndex..<eq].trimmingCharacters(in: .whitespaces) == key else { continue }
            let value = line[line.index(after: eq)...].split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false)[0]
                .trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "_", with: "")
            return Double(value)
        }
        return nil
    }

    private static func rawString(_ text: String, table: String, key: String) -> String? {
        var current = ""
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("[") { current = String(line.dropFirst().prefix(while: { $0 != "]" })); continue }
            guard current == table, line.hasPrefix(key), let eq = line.firstIndex(of: "=") else { continue }
            let value = line[line.index(after: eq)...].trimmingCharacters(in: .whitespaces)
            if value.hasPrefix("\"") { return value.dropFirst().prefix(while: { $0 != "\"" }).description }
        }
        return nil
    }

    // MARK: Coverage

    @Test func everyEnginePresetHasAGuideAndEveryGuideAnEnginePreset() {
        let engine = Self.enginePresets()
        #expect(engine.count >= 10, "config.PRESETS was not found in config.py: \(engine)")
        #expect(Set(engine) == Set(Preset.allCases.map(\.rawValue)),
                "config.PRESETS and the Swift Preset enum differ: a new preset needs a case in Preset and its entry in Preset.guide")
        for preset in Preset.allCases {
            let guide = preset.guide
            #expect(guide.what.count > 30 && guide.goodFor.count > 20 && guide.cost.count > 20, "\(preset.rawValue): a guide text is empty")
        }
    }

    @Test func theSpherePresetsOfTheEngineAreThePlanets() {
        let planets = Set(PresetGroup.planet.presets.map(\.rawValue))
        #expect(planets == Self.sphereNames)
        let engine = Self.config
        #expect(engine.contains("SPHERE_PRESETS: tuple[str, ...] = (\"tiny3s\", \"mini3s\", \"64k3s\")"))
    }

    @Test func everyPresetIsInExactlyOneGroup() {
        var seen: [Preset] = []
        for group in PresetGroup.allCases { seen += group.presets }
        #expect(seen.count == Preset.allCases.count)
        #expect(Set(seen) == Set(Preset.allCases))
    }

    // MARK: The numbers

    @Test func theEngineDefaultsAreConfigPy() {
        let d = PresetFacts.engineDefaults
        #expect(Self.engineDefault("WorldConfig", "size") == d.size)
        #expect(Self.engineDefault("WorldConfig", "height") == d.height)
        #expect(Self.engineDefault("WorldConfig", "radius") == d.radius)
        #expect(Self.engineDefault("PopulationConfig", "n_max") == Double(d.nMax))
        #expect(Self.engineDefault("PopulationConfig", "n_init") == Double(d.nInit))
        #expect(Self.engineDefault("WorldConfig", "dims") == 2)
        #expect(Self.config.contains("geometry: str = \"box\""))
        #expect(d.shape == .flat)
    }

    @Test(arguments: Preset.allCases)
    func theFactsOfEveryPresetAreWhatItsFileSays(preset: Preset) throws {
        let text = try Repo.preset(preset.rawValue)
        let facts = try #require(PresetFacts.preset(preset, repo: Repo.root))
        func value(_ table: String, _ key: String, default cls: String) -> Double {
            Self.rawValue(text, table: table, key: key) ?? Self.engineDefault(cls, key) ?? .nan
        }
        #expect(facts.size == value("world", "size", default: "WorldConfig"))
        #expect(facts.height == value("world", "height", default: "WorldConfig"))
        #expect(facts.radius == value("world", "radius", default: "WorldConfig"))
        #expect(Double(facts.nMax) == value("population", "n_max", default: "PopulationConfig"))
        #expect(Double(facts.nInit) == value("population", "n_init", default: "PopulationConfig"))
        let geometry = Self.rawString(text, table: "world", key: "geometry") ?? "box"
        let dims = Int(Self.rawValue(text, table: "world", key: "dims") ?? 2)
        let expected: PresetFacts.Shape = geometry == "sphere" ? .planet : dims == 3 ? .box3D : .flat
        #expect(facts.shape == expected)
        #expect(facts.shape.group == preset.group, "\(preset.rawValue): the guide groups it as \(preset.group), its file says \(facts.shape)")
        #expect(facts.nInit <= facts.nMax)
    }

    @Test func the64kFileSetsNothingSoItsFactsAreTheDefaults() throws {
        #expect(PresetFacts.preset(.k64, repo: Repo.root) == PresetFacts.engineDefaults)
    }

    @Test func aPresetFileIsReadFromTheRepositoryItIsGiven() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let dir = scratch.url.appendingPathComponent("src/alife/presets", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try "[population]\nn_max = 2_048\nn_init = 100\n[world]\nsize = 96.5\ndims = 3\nheight = 20\n"
            .write(to: dir.appendingPathComponent("tiny3d.toml"), atomically: true, encoding: .utf8)
        let facts = try #require(PresetFacts.preset(.tiny3d, repo: scratch.url))
        #expect(facts == PresetFacts(shape: .box3D, size: 96.5, height: 20, radius: 0, nMax: 2048, nInit: 100))
        #expect(PresetFacts.preset(.dev, repo: scratch.url) == nil, "no file, no facts")
    }

    @Test func theTextsOfTheNumbers() {
        #expect(PresetFacts.count(65536) == "65\u{202F}536")
        #expect(PresetFacts.count(256) == "256")
        #expect(PresetFacts.count(1_048_576) == "1\u{202F}048\u{202F}576")
        let flat = PresetFacts(shape: .flat, size: 128, height: 0, radius: 0, nMax: 1024, nInit: 256)
        #expect(flat.worldText == "128\u{2009}\u{00D7}\u{2009}128")
        #expect(flat.populationText == "up to 1\u{202F}024 organisms, 256 at the start")
        let box = PresetFacts(shape: .box3D, size: 320, height: 80, radius: 0, nMax: 65536, nInit: 16384)
        #expect(box.worldText == "320\u{2009}\u{00D7}\u{2009}320\u{2009}\u{00D7}\u{2009}80 high")
        let planet = PresetFacts(shape: .planet, size: 1024, height: 40, radius: 90, nMax: 65536, nInit: 16384)
        #expect(planet.worldText == "radius 90, shell 40 thick", "a planet's world.size is no size of the planet and is not shown")
    }

    // MARK: Snapshots

    private static let manifest = PresetSnapshotManifest.load(repo: Repo.root)

    @Test func theCommittedManifestHasOneCompleteEntryPerPreset() throws {
        let manifest = try #require(Self.manifest, "gui/PresetSnapshots/manifest.json is missing or unreadable")
        #expect(!manifest.machine.isEmpty)
        #expect(manifest.gitDirty == false, "the final snapshots are made from committed code")
        #expect(manifest.snapshots.count == Preset.allCases.count)
        for preset in Preset.allCases {
            let entry = try #require(manifest.entry(for: preset), "\(preset.rawValue) has no manifest entry")
            let facts = try #require(PresetFacts.preset(preset, repo: Repo.root))
            #expect(entry.seed == 1)
            #expect(entry.step >= 5_000, "\(preset.rawValue): step \(entry.step)")
            #expect(entry.population > 0 && entry.population <= facts.nMax, "\(preset.rawValue): population \(entry.population)")
            #expect(entry.view?.isEmpty == false)
            if let inView = entry.windowPopulation {
                #expect(inView > 0 && inView <= entry.population && entry.view?.contains("window") == true, "\(preset.rawValue): \(inView) in view")
            } else {
                #expect(entry.view?.contains("window") != true, "\(preset.rawValue): a window without its count")
            }
            #expect((entry.peakMemoryBytes ?? 0) > 50_000_000, "\(preset.rawValue): memory \(String(describing: entry.peakMemoryBytes))")
            #expect(entry.stepsPerSecond > 0)
            #expect(entry.gitRev.count == 40 && entry.gitRev.allSatisfy(\.isHexDigit), "\(preset.rawValue): git rev \(entry.gitRev)")
            #expect(entry.date.range(of: #"^\d{4}-\d{2}-\d{2}$"#, options: .regularExpression) != nil, "\(preset.rawValue): date \(entry.date)")
        }
    }

    @Test(arguments: Preset.allCases)
    func theCommittedSnapshotIsAJPEGOfTheRightShapeAndSize(preset: Preset) throws {
        let url = PresetSnapshotManifest.imageURL(preset, repo: Repo.root)
        let data = try Data(contentsOf: url)
        #expect(data.count < 125_000, "\(preset.rawValue).jpg is \(data.count) bytes")
        #expect(data.count > 5_000, "\(preset.rawValue).jpg is only \(data.count) bytes")
        #expect(data.starts(with: [0xFF, 0xD8, 0xFF]), "not a JPEG")
        let source = try #require(CGImageSourceCreateWithData(data as CFData, nil))
        let props = try #require(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        #expect(props[kCGImagePropertyPixelWidth] as? Int == 640)
        #expect(props[kCGImagePropertyPixelHeight] as? Int == 400)
    }

    @Test func aRepositoryWithoutSnapshotsGivesEntriesWithoutThemAndNoCrash() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        #expect(PresetSnapshotManifest.load(repo: scratch.url) == nil)
        let content = PresetGuideContent.load(repo: scratch.url)
        #expect(content.entries.count == Preset.allCases.count)
        for entry in content.entries {
            #expect(entry.snapshot == nil && entry.facts == nil)
            #expect(entry.snapshotCaption() == nil && entry.speedCaption(machine: content.machine) == nil)
            #expect(!FileManager.default.fileExists(atPath: entry.imageURL.path))
            #expect(!entry.guide.what.isEmpty, "the words do not depend on the repository")
        }
        let dir = scratch.url.appendingPathComponent("gui/PresetSnapshots", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try "{ not json".write(to: dir.appendingPathComponent("manifest.json"), atomically: true, encoding: .utf8)
        #expect(PresetSnapshotManifest.load(repo: scratch.url) == nil, "a broken manifest is no manifest")
    }

    @Test func theCaptionsSayWhatTheSnapshotIsAndWhereTheSpeedWasMeasured() throws {
        let json = """
        {"machine": "Apple M5 Pro", "git_dirty": false, "snapshots": [{"preset": "64k", "seed": 1, "step": 30000, "population": 33400,
         "window_population": 3900, "steps_per_s": 41.3, "peak_memory_bytes": 2410000000, "git_rev": "397b0f588205173abf42df7911ff92e3a599c735",
         "date": "2026-11-01", "view": "a 512 \u{00D7} 320 window of the world, of typical density"}]}
        """
        let manifest = try JSONDecoder().decode(PresetSnapshotManifest.self, from: Data(json.utf8))
        let entry = PresetGuideEntry(preset: .k64, facts: PresetFacts.preset(.k64, repo: Repo.root), snapshot: manifest.entry(for: .k64),
                                     imageURL: URL(fileURLWithPath: "/nonexistent.jpg"))
        let caption = try #require(entry.snapshotCaption())
        #expect(caption == "A real run (seed 1) at step 30\u{202F}000: 33\u{202F}400 organisms alive in the whole world. The picture shows a 512 \u{00D7} 320 window of the world, of typical density, with 3\u{202F}900 of them in it.")
        #expect(entry.speedCaption(machine: manifest.machine) == "41 steps/s with this many organisms alive, measured on Apple M5 Pro.")
        #expect(entry.memoryCaption() == "Peak memory about 2.4 GB (approximate).")
        #expect(PresetGuideEntry.memoryText(250_000_000) == "250 MB" && PresetGuideEntry.memoryText(8_970_000_000) == "9.0 GB")
        var slow = entry
        slow.snapshot?.stepsPerSecond = 3.26
        #expect(slow.speedCaption(machine: "Apple M5 Pro")?.hasPrefix("3.3 steps/s") == true)
        #expect(manifest.entry(for: .dev) == nil)
        // a screen reader gets the whole row, not its name and first sentence
        let spoken = entry.spokenText(machine: manifest.machine)
        for part in [entry.guide.what, entry.guide.goodFor, entry.guide.cost, caption, entry.speedCaption(machine: manifest.machine)!, entry.memoryCaption()!,
                     entry.facts!.populationText, entry.nameNote!] {
            #expect(spoken.contains(part), "the spoken row lacks: \(part)")
        }
        #expect(entry.imageLabel.contains("512") && entry.imageLabel.contains("step 30"))
        // a manifest written before the window count and the memory existed still reads
        let old = #"{"machine": "m", "snapshots": [{"preset": "dev", "seed": 1, "step": 5000, "population": 10, "steps_per_s": 2, "git_rev": "x", "date": "d"}]}"#
        let decoded = try JSONDecoder().decode(PresetSnapshotManifest.self, from: Data(old.utf8))
        #expect(decoded.snapshots[0].windowPopulation == nil && decoded.snapshots[0].peakMemoryBytes == nil && decoded.gitDirty == nil)
    }

    // MARK: Words held to the data

    @Test func everyComparisonTheWordsMakeHoldsInTheCommittedManifest() throws {
        let manifest = try #require(Self.manifest)
        #expect(Preset.comparisons.count >= 10)
        for c in Preset.comparisons {
            let ratio = try #require(c.ratio(in: manifest), "\(c.preset.rawValue) against \(c.reference.rawValue): no value in the manifest")
            #expect(c.band.contains(ratio), "\"\(c.sentence)\" says \(c.preset.rawValue)/\(c.reference.rawValue) \(c.metric) is in \(c.band), the manifest has \(ratio)")
        }
        for e in Preset.extremes { #expect(e.holds(in: manifest), "\"\(e.sentence)\" does not hold in the manifest") }
    }

    @Test func aComparisonFailsWhenTheManifestStopsSupportingIt() throws {
        var manifest = try #require(Self.manifest)
        let dev = try #require(manifest.snapshots.firstIndex { $0.preset == "dev" })
        let claim = try #require(Preset.comparisons.first { $0.preset == .dev && $0.metric == .speed })
        #expect(claim.holds(in: manifest))
        manifest.snapshots[dev].stepsPerSecond = manifest.entry(for: .k64)!.stepsPerSecond * 1.2  // "several times faster" is no longer so
        #expect(!claim.holds(in: manifest))
        let big = try #require(manifest.snapshots.firstIndex { $0.preset == "256k" })
        let memory = try #require(Preset.comparisons.first { $0.metric == .memory })
        manifest.snapshots[big].peakMemoryBytes = manifest.entry(for: .k64)!.peakMemoryBytes! * 2
        #expect(!memory.holds(in: manifest))
        let tiny = try #require(manifest.snapshots.firstIndex { $0.preset == "tiny" })
        let fastest = try #require(Preset.extremes.first { $0.kind == .fastest })
        #expect(fastest.holds(in: try #require(Self.manifest)))
        manifest.snapshots[tiny].stepsPerSecond = 1
        #expect(!fastest.holds(in: manifest))
        #expect(!claim.holds(in: PresetSnapshotManifest(machine: "", gitDirty: nil, snapshots: [])), "no value, no support")
    }

    @Test func theCostTextIsItsLeadAndTheSentencesOfItsComparisons() {
        for preset in Preset.allCases {
            let own = Preset.extremes.filter { $0.preset == preset }.map(\.sentence) + Preset.comparisons.filter { $0.preset == preset }.map(\.sentence)
            for sentence in own { #expect(preset.guide.cost.contains(sentence)) }
        }
        #expect(Preset.k256.guide.cost.contains("Nearly four times the memory of 64k."))
    }

    @Test func theWordsAreInPlainLanguage() {
        let jargon = ["overlay", "buoyancy", "per-organism", "organism limit", "thinner", "per step", "groups and bodies", "n_max", "tiny3d rules"]
        for preset in Preset.allCases {
            let g = preset.guide
            for text in [g.what, g.goodFor, g.cost] {
                for word in jargon { #expect(!text.localizedCaseInsensitiveContains(word), "\(preset.rawValue): \"\(word)\" in: \(text)") }
            }
        }
        for group in PresetGroup.allCases { #expect(!group.blurb.localizedCaseInsensitiveContains("overlay")) }
    }

    @Test func theMeaningOfKInANameIsSaidOnlyWhereTheFileAgrees() throws {
        for preset in Preset.allCases {
            let facts = try #require(PresetFacts.preset(preset, repo: Repo.root))
            let note = PresetFacts.nameNote(preset, nMax: facts.nMax)
            if ["64k", "256k", "64k3d", "64k3s"].contains(preset.rawValue) {
                #expect(note?.contains(PresetFacts.count(facts.nMax)) == true, "\(preset.rawValue): \(String(describing: note))")
            } else {
                #expect(note == nil, "\(preset.rawValue)")
            }
        }
        #expect(PresetFacts.nameNote(.k64, nMax: 1000) == nil, "a note that the file does not bear out is not shown")
    }
}

import Foundation
import Testing

@testable import AlifeCore

/// `PresetFile`, the scalar reader that `Cadence` now stands on, against the cadence reader it replaced.
@Suite struct PresetFileTests {
    /// The `Cadence(presetTOML:)` of origin/main e1f718e, verbatim in behaviour: the reference for "the `[run]`
    /// reading is unchanged". It is unchanged on every file `tomllib` accepts, and on the odd files below that it
    /// does not (a key set twice is the exception, `aKeySetTwiceIsTheLastValue`).
    private func legacyCadence(_ text: String) -> Cadence {
        var cadence = Cadence.engineDefaults
        var inRun = false
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("[") {
                inRun = line.hasPrefix("[run]")
                continue
            }
            guard inRun, let eq = line.firstIndex(of: "=") else { continue }
            let key = line[..<eq].trimmingCharacters(in: .whitespaces)
            let value = line[line.index(after: eq)...].split(separator: "#", maxSplits: 1,
                                                              omittingEmptySubsequences: false)[0]
                .trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "_", with: "")
            guard let n = Int(value), n >= 1 else { continue }
            if key == "metrics_every" { cadence.metricsEvery = n }
            if key == "heavy_metrics_every" { cadence.heavyMetricsEvery = n }
        }
        return cadence
    }

    static let presetNames = ["tiny", "dev", "64k", "256k", "tiny3d", "fast"]

    @Test(arguments: presetNames)
    func cadenceOfEveryPresetFileIsUnchanged(name: String) throws {
        let text = try Repo.preset(name)
        #expect(Cadence(presetTOML: text) == legacyCadence(text))
    }

    @Test func everyPresetFileOfTheRepositoryIsCovered() throws {
        let files = try FileManager.default.contentsOfDirectory(atPath: Repo.presets.path)
            .filter { $0.hasSuffix(".toml") }.map { String($0.dropLast(5)) }
        #expect(Set(files) == Set(Self.presetNames), "a new preset file needs a line in presetNames")
    }

    @Test func theCadenceOfTheRealPresetsIsWhatTheirFilesSay() {
        #expect(Cadence.preset("tiny", repo: Repo.root) == Cadence(metricsEvery: 10, heavyMetricsEvery: 10))
        #expect(Cadence.preset("tiny3d", repo: Repo.root) == Cadence(metricsEvery: 10, heavyMetricsEvery: 10))
        #expect(Cadence.preset("dev", repo: Repo.root) == Cadence.engineDefaults)
        #expect(Cadence.preset("no-such-preset", repo: Repo.root) == nil)
    }

    @Test(arguments: [
        "[run]\nmetrics_every = 25\nheavy_metrics_every = 2_500\n",
        "[other]\nmetrics_every = 7\n[run]\nsteps = 5\n",
        "[run] # the run table\nmetrics_every = 30 # every thirty\nheavy_metrics_every=300#x\n",
        "[run]\nmetrics_every = 10.0\nheavy_metrics_every = 1e3\n",
        "[run]\nmetrics_every = 0\nheavy_metrics_every = -5\n",
        "[run]\nmetrics_every = \"10\"\nheavy_metrics_every = true\n",
        "[runner]\nmetrics_every = 3\n[run.sub]\nmetrics_every = 4\n[[run]]\nmetrics_every = 5\n",
        "metrics_every = 9\n[world]\nsize = 128.0\n",
        "[run]\r\nmetrics_every = 12\r\nheavy_metrics_every = 120\r\n",
        "   [run]   \n   metrics_every   =   15   \n\theavy_metrics_every\t=\t150\n",
        "[run]\nmetrics_every = 5\nmetrics_every = 6\n[world]\nmetrics_every = 99\n",
        "[ run ]\nmetrics_every = 8\n",
        "[run]\nmetrics_every = +20\nheavy_metrics_every = 0x10\n",
        "[run]\nmetrics_every\n= 4\nmetrics_every = \n",
        "[run\nmetrics_every = 7\n",
        "[run]\nmetrics_every = 7\n[world\nmetrics_every = 9\n",
        "[\nmetrics_every = 7\n",
        "",
    ])
    func cadenceOfOddFilesIsUnchanged(text: String) {
        #expect(Cadence(presetTOML: text) == legacyCadence(text))
    }

    /// The one difference from the reader it replaced: that one kept the last value that was an integer of at least 1,
    /// this one keeps the last scalar and `Cadence` then ignores a value below 1. TOML refuses a key set twice, so the
    /// engine would refuse such a preset file; this pins the behaviour rather than endorsing it.
    @Test func aKeySetTwiceIsTheLastValue() {
        let text = "[run]\nmetrics_every = 10\nmetrics_every = 0\n"
        #expect(legacyCadence(text).metricsEvery == 10)
        #expect(Cadence(presetTOML: text).metricsEvery == Cadence.engineDefaults.metricsEvery)
        #expect(PresetFile(text: text).int("run", "metrics_every") == 0)
        let later = "[run]\nmetrics_every = 10\nmetrics_every = 20\n"
        #expect(Cadence(presetTOML: later) == legacyCadence(later))
    }

    // MARK: Scalars

    @Test func scalarsAreIntegersAndNumbers() {
        let file = PresetFile(text: """
        top = 3
        [world]
        size = 128.0
        height = 24      # an integer, read as a number too
        dims = 3
        tiny = 5e-4
        negative = -0.25
        plus = +2
        grouped = 1_000_000
        name = "tiny3d"
        flag = true
        list = [0.16, 0.10, 0.16]
        inline = { a = 1 }
        inf_value = inf
        not_a_number = nan
        hex = 0x10
        missing_value =
        [physics]
        size = 7
        """)
        #expect(file.int("", "top") == 3)
        #expect(file.double("world", "size") == 128 && file.int("world", "size") == nil, "128.0 is a float")
        #expect(file.double("world", "height") == 24 && file.int("world", "height") == 24)
        #expect(file.int("world", "dims") == 3 && file.double("world", "dims") == 3)
        #expect(file.double("world", "tiny") == 5e-4)
        #expect(file.double("world", "negative") == -0.25)
        #expect(file.int("world", "plus") == 2)
        #expect(file.int("world", "grouped") == 1_000_000)
        for key in ["name", "flag", "list", "inline", "inf_value", "not_a_number", "hex", "missing_value", "absent"] {
            #expect(file.value("world", key) == nil, "\(key) is not a scalar")
        }
        #expect(file.double("physics", "size") == 7, "a key is scoped to its table")
        #expect(file.double("world", "size") == 128)
        #expect(file.double("nowhere", "size") == nil)
    }

    @Test func theRealTiny3DPresetIsRead() throws {
        let file = PresetFile(text: try Repo.preset("tiny3d"))
        #expect(file.int("world", "dims") == 3)
        #expect(file.double("world", "height") == 24)
        #expect(file.double("physics", "gravity") == 0.03)
        #expect(file.double("physics", "buoyancy_max") == 0.04)
        #expect(file.int("energy", "buoyancy_cost") == 16 && file.int("energy", "buoyancy_cap_cost") == 8)
        #expect(file.double("mutation", "buoy_init") == 0.5 && file.double("mutation", "buoy_sigma") == 0.02)
        #expect(file.double("fields", "benthic_lambda") == 8 && file.double("fields", "photic_amp") == 0.5)
        #expect(file.value("fields", "chem_diffusion") == nil, "an array is not a scalar")
        #expect(file.int("run", "metrics_every") == 10)
    }

    @Test func aMissingPresetFileIsNil() {
        #expect(PresetFile.preset("no-such-preset", repo: Repo.root) == nil)
        #expect(PresetFile.preset("tiny", repo: Repo.root) != nil)
    }
}

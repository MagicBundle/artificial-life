import Foundation
import Testing

@testable import AlifeCore

@Suite struct MetricsTailerTests {
    private func parse(_ line: String) -> MetricsSnapshot? { MetricsTailer.parse(Data(line.utf8)) }

    @Test func aTwoDRowHasNoHeightKeys() throws {
        let snapshot = try #require(parse("""
        {"step": 100, "n_alive": 812, "steps_per_s": 61.5, "clades_alive": 14, "generation_max": 9, "ledger_error": 0}
        """))
        #expect(snapshot.step == 100 && snapshot.nAlive == 812 && snapshot.stepsPerSecond == 61.5)
        #expect(snapshot.clades == 14 && snapshot.generationMax == 9 && snapshot.ledgerError == 0)
        #expect(snapshot.heightMean == nil && snapshot.groundedFrac == nil && snapshot.surfaceFrac == nil)
    }

    @Test func aThreeDRowCarriesTheHeightKeys() throws {
        let snapshot = try #require(parse("""
        {"step": 40, "n_alive": 200, "steps_per_s": 30.0, "clades_alive": 5, "generation_max": 2, "ledger_error": 0,
         "grounded_frac": 0.375, "height_mean": 6.25, "surface_frac": 0.0625, "buoy_heat": 12}
        """))
        #expect(snapshot.heightMean == 6.25 && snapshot.groundedFrac == 0.375 && snapshot.surfaceFrac == 0.0625)
    }

    @Test func nullHeightKeysAndBareNaNAreNil() throws {
        // Nobody alive: the height keys are null, and Python writes NaN for a ratio it could not form.
        let snapshot = try #require(parse("""
        {"step": 7, "n_alive": 0, "steps_per_s": NaN, "ledger_error": 0, "grounded_frac": NaN, "height_mean": null, "surface_frac": null}
        """))
        #expect(snapshot.nAlive == 0 && snapshot.stepsPerSecond == nil)
        #expect(snapshot.groundedFrac == nil && snapshot.heightMean == nil && snapshot.surfaceFrac == nil)
    }

    @Test func aRowWithoutAStepIsNotARow() {
        #expect(parse("{\"n_alive\": 3}") == nil)
        #expect(parse("not json") == nil)
        #expect(parse("[1, 2]") == nil)
    }

    @Test func theNewestCompleteRowWinsOverATornOne() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let file = scratch.url.appendingPathComponent("metrics.jsonl")
        let text = "{\"step\": 10, \"n_alive\": 5}\n{\"step\": 20, \"n_alive\": 6, \"grounded_frac\": 0.5}\n{\"step\": 30, \"n_al"
        try text.write(to: file, atomically: true, encoding: .utf8)
        let snapshot = try #require(MetricsTailer.readLast(at: file))
        #expect(snapshot.step == 20 && snapshot.groundedFrac == 0.5)
        #expect(MetricsTailer.readLast(at: scratch.url.appendingPathComponent("absent.jsonl")) == nil)
        try "".write(to: file, atomically: true, encoding: .utf8)
        #expect(MetricsTailer.readLast(at: file) == nil)
    }
}

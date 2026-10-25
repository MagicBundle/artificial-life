import AlifeCore
import Foundation
import Metal
import Testing
import os
import simd

@testable import AlifeControl

// The client's share of the play-loop gate (DESIGN §33.5.3): how long the World's draw takes on the GPU and the CPU while the
// engine is stepping beside it. `ALIFE_GPU_TESTS=1 swift test --filter WorldRenderBench` draws synthetic frames of the size of
// the committed fixtures and of a 64k3d world at 60 Hz, a new frame at the base rate (14.4 a second) and a new map every 25
// steps, and prints the draw time's p50, p95 and maximum and the draws that were dropped. Run `alife bench --play` at `--rate
// 1` beside it to read the two together.

/// Percentiles of a list of times, by linear interpolation between ranks.
enum BenchStats {
    struct Summary: Equatable, Codable {
        var count: Int
        var p50: Double
        var p95: Double
        var max: Double
    }

    /// The `q`-quantile (0...1) of `sorted`, which is ascending and not empty.
    static func percentile(_ sorted: [Double], _ q: Double) -> Double {
        let position = Swift.min(Swift.max(q, 0), 1) * Double(sorted.count - 1)
        let low = Int(position.rounded(.down)), high = Int(position.rounded(.up))
        return sorted[low] + (sorted[high] - sorted[low]) * (position - Double(low))
    }

    static func summary(_ values: [Double]) -> Summary? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        return Summary(count: values.count, p50: percentile(sorted, 0.5), p95: percentile(sorted, 0.95), max: sorted[sorted.count - 1])
    }
}

/// The arithmetic of the bench, without a GPU.
@Suite struct BenchStatsTests {
    @Test func percentilesInterpolateBetweenRanks() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0]
        #expect(BenchStats.percentile(sorted, 0) == 1 && BenchStats.percentile(sorted, 1) == 5)
        #expect(BenchStats.percentile(sorted, 0.5) == 3 && BenchStats.percentile(sorted, 0.25) == 2)
        #expect(abs(BenchStats.percentile(sorted, 0.95) - 4.8) < 1e-12)
        #expect(BenchStats.percentile([7], 0.95) == 7)
        #expect(BenchStats.percentile(sorted, 2) == 5 && BenchStats.percentile(sorted, -1) == 1, "clamped")
    }

    @Test func aSummaryOfUnsortedTimes() {
        let s = BenchStats.summary([9, 1, 5, 3, 7])
        #expect(s == BenchStats.Summary(count: 5, p50: 5, p95: 8.6, max: 9))
        #expect(BenchStats.summary([]) == nil)
    }

    @Test func theSyntheticFrameIsAWellFormedFrame() {
        let frame = BenchScene.frame(rows: 100, size: 320, height: 80, seed: 5, step: 7)
        #expect(frame.n == 100 && frame.pos.count == 300 && frame.uid.count == 100 && frame.rgb.count == 100)
        #expect(frame.step == 7 && frame.nBonds > 0)
        for k in 0..<frame.nBonds { #expect(frame.bond(k).i < frame.bond(k).j && frame.bond(k).j < frame.n) }
        for i in 0..<frame.n {
            let p = frame.position(i)
            #expect(p.x >= 0 && p.x < 320 && p.y >= 0 && p.y < 320 && p.z >= 0 && p.z <= 80)
        }
        #expect(frame == BenchScene.frame(rows: 100, size: 320, height: 80, seed: 5, step: 7), "deterministic")
        #expect(frame != BenchScene.frame(rows: 100, size: 320, height: 80, seed: 6, step: 7))
    }

    @Test func theScalesAndTheSettingsComeFromTheEnvironment() {
        #expect(BenchScene.rows(from: nil) == [64, 42_000])
        #expect(BenchScene.rows(from: "7,4096") == [7, 4096])
        #expect(BenchScene.rows(from: "nonsense") == [64, 42_000], "a bad list is the default")
        #expect(BenchScene.rows(from: "0,100") == [64, 42_000], "a scale of nothing is a bad list")
        #expect(BenchScene.integer(from: "120", default: 300) == 120 && BenchScene.integer(from: nil, default: 300) == 300)
        #expect(BenchScene.integer(from: "-3", default: 300) == 300 && BenchScene.integer(from: "x", default: 300) == 300)
    }
}

/// Synthetic worlds for the bench: the geometry of a 64k3d world (L = 320, height 80) with the rows scattered over it.
enum BenchScene {
    static let defaultRows = [64, 42_000]

    static func rows(from text: String?) -> [Int] {
        guard let text else { return defaultRows }
        let parsed = text.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard !parsed.isEmpty, parsed.allSatisfy({ $0 > 0 }) else { return defaultRows }
        return parsed
    }

    static func integer(from text: String?, default fallback: Int) -> Int {
        guard let text, let value = Int(text), value > 0 else { return fallback }
        return value
    }

    /// `rows` organisms scattered over a world of `size` and `height` by a seeded generator, a bond between two of every
    /// five neighbours in the listing, colours from the slot palette.
    static func frame(rows: Int, size: Float, height: Float, seed: UInt64, step: UInt64) -> PlayFrame {
        var state = seed &* 0x9E37_79B9_7F4A_7C15 &+ 1
        func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 40) / Float(1 << 24)
        }
        let positions = (0..<rows).map { _ in SIMD3<Float>(next() * size, next() * size, next() * height) }
        var frame = TestFrames.frame(positions, step: step)
        frame.rgb = (0..<rows).map { WorldPalette.slots[$0 % WorldPalette.slots.count] }
        frame.bonds = (0..<max(rows - 1, 0)).filter { $0 % 5 < 2 }.flatMap { [UInt32($0), UInt32($0 + 1)] }
        return frame
    }
}

@MainActor
@Suite(
    "World render bench",
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1",
        "set ALIFE_GPU_TESTS=1 to draw on the GPU")
)
struct WorldRenderBench {
    /// One measured scenario.
    struct Result: Codable {
        var rows: Int
        var scenario: String
        var draws: Int
        var dropped: Int
        var gpuMs: BenchStats.Summary?
        var cpuMs: BenchStats.Summary?
    }

    private func environment(_ key: String) -> String? { ProcessInfo.processInfo.environment[key] }

    private func outputDirectory() throws -> URL {
        let base = environment("ALIFE_GPU_TESTS_OUT").map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.temporaryDirectory.appendingPathComponent("alife-gpu-tests", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    /// Draws `scenario` for `draws` ticks of 60 Hz and returns what it cost.
    private func measure(rows: Int, scenario: String, level: WorldLevel, points: Bool, draws: Int, size: (Int, Int), hz: Int,
                         globeS: Float? = nil, minimap: Bool = false) async throws -> Result {
        let device = try #require(MTLCreateSystemDefaultDevice(), "no Metal device")
        let renderer = try Renderer(device: device)
        let fixture = try PlayFixtureStore.store("3d")
        var header = try #require(fixture.header)
        (header.size, header.height, header.capacity) = (320, 80, max(65_536, rows))
        renderer.configureWorld(header: header)
        let aspect = Float(size.0) / Float(size.1)
        var camera = Camera.world(for: header, aspect: aspect)
        switch level {
        case .planet: break
        case .region: camera.distance = WorldLevels.flyDistance(.region, size: 320, fovY: camera.fovY, aspect: aspect)
        case .organism: camera.distance = WorldLevels.flyDistance(.organism, size: 320, fovY: camera.fovY, aspect: aspect)
        }
        camera.target = SIMD3(160, 160, 0)
        renderer.camera = camera
        renderer.worldDraw.level = level
        var display = WorldDisplay()
        display.agentPoints = points
        if environment("ALIFE_BENCH_PHYSICAL") == "1" { display.sizeMode = .physical }  // v1's carpet: the dearest the viewer can draw
        renderer.worldDraw.display = display
        // The globe path (DESIGN §36.10): the view at the morph's scalar `globeS` (1: the globe, 0.5: mid-morph, 0: the flat close-up drawn
        // by the same mesh), centred on the equator at the populated side; the minimap in the corner when asked.
        var vantage: WorldVantage?
        if let globeS {
            let fov = camera.fovY
            let d = globeS >= 1 ? Morph.globeDistance(L: 320, fovY: fov, aspect: aspect)
                : (globeS <= 0 ? WorldLevels.flyDistance(.region, size: 320, fovY: fov, aspect: aspect, globe: true)
                   : Morph.distance(forS: globeS, L: 320, fovY: fov, aspect: aspect))
            let pose = WorldPose(x: 160, y: 40, z: 0, heading: 0, tilt: 0, distance: d, fovY: fov)
            vantage = WorldVantage(pose: pose, size: 320, height: 80, aspect: aspect, globe: true)
            renderer.worldDraw.vantage = vantage
            renderer.worldDraw.wideTiles = globeS < 0.3
        }
        if environment("ALIFE_BENCH_NO_GLOW") == "1" { renderer.worldDraw.glow = false }
        if minimap { renderer.worldDraw.inset = CGRect(x: 0.74, y: 0.62, width: 0.24, height: 0.34) }
        let frames = [0, 1].map { BenchScene.frame(rows: rows, size: 320, height: 80, seed: UInt64($0 + 1), step: UInt64($0)) }
        renderer.showWorld(frame: frames[0])
        // The state the shipped World draws in (DESIGN §36.7): the coverage radius of the rows in view, as `WorldModel` counts it.
        // Without it the organisms would be drawn at the 9 px absolute cap, which is not what anyone sees.
        if display.sizeMode == .capped, environment("ALIFE_BENCH_NO_COVER") != "1" {  // (that variable draws as the A0 bench of the review did)
            let inView = vantage.map { OrganismSizePolicy.rowsInView(frame: frames[0], vantage: $0) }
                ?? OrganismSizePolicy.rowsInView(frame: frames[0], camera: camera, aspect: aspect, size: 320)
            renderer.worldDraw.coverRadius = OrganismSizePolicy.coverRadius(area: Float(size.0 * size.1), rows: inView)
            renderer.worldDraw.pixelsPerPoint = 2
        }
        let map = try #require(fixture.map)
        renderer.deliverWorld(map: map, serial: 0, start: .current)
        renderer.showWorld(territory: fixture.territory, serial: 0)
        let target = try renderer.makeOffscreenTarget(width: size.0, height: size.1)

        let collected = OSAllocatedUnfairLock<[DrawTiming]>(initialState: [])
        renderer.timingHandler = { timing in collected.withLock { $0.append(timing) } }
        var warm = 0
        for _ in 0..<5 {  // pipelines, buffer sets, textures
            while try !renderer.drawNow(into: target) { try await Task.sleep(for: .milliseconds(2)) }
            warm += 1
        }
        try await settled(collected, count: warm)
        collected.withLock { $0.removeAll() }

        let clock = ContinuousClock()
        let begin = clock.now
        let period = Duration.seconds(1.0 / Double(hz))
        var submitted = 0, dropped = 0, mapSerial = 1
        for k in 0..<draws {
            try await clock.sleep(until: begin + period * k)
            // A frame at the base rate (14.4 a second) and a map every 25 steps (every 1.7 s at 1x).
            if k % max(hz * 10 / 144, 1) == 0 { renderer.showWorld(frame: frames[(k / 4) % 2]) }
            if k % max(hz * 17 / 10, 1) == 0 {
                renderer.deliverWorld(map: map, serial: mapSerial, start: .current)
                mapSerial += 1
            }
            if try renderer.drawNow(into: target) { submitted += 1 } else { dropped += 1 }
        }
        try await settled(collected, count: submitted)
        let timings = collected.withLock { $0 }
        return Result(rows: rows, scenario: scenario, draws: submitted, dropped: dropped,
                      gpuMs: BenchStats.summary(timings.map(\.gpuMs)), cpuMs: BenchStats.summary(timings.map(\.cpuMs)))
    }

    /// Waits for `count` completion handlers (they run on the GPU's thread).
    private func settled(_ collected: OSAllocatedUnfairLock<[DrawTiming]>, count: Int) async throws {
        let deadline = ContinuousClock.now + .seconds(30)
        while collected.withLock({ $0.count }) < count {
            #expect(ContinuousClock.now < deadline, "the GPU finished only \(collected.withLock { $0.count }) of \(count) draws")
            if ContinuousClock.now >= deadline { return }
            try await Task.sleep(for: .milliseconds(5))
        }
    }

    @Test func theDrawTimeOfTheWorldAtEachLevelAndScale() async throws {
        let rows = BenchScene.rows(from: environment("ALIFE_BENCH_ROWS"))
        let draws = BenchScene.integer(from: environment("ALIFE_BENCH_DRAWS"), default: 300)
        let hz = BenchScene.integer(from: environment("ALIFE_BENCH_HZ"), default: 60)
        let width = BenchScene.integer(from: environment("ALIFE_BENCH_WIDTH"), default: 2200)
        let height = BenchScene.integer(from: environment("ALIFE_BENCH_HEIGHT"), default: 1400)
        let scenarios: [(String, WorldLevel, Bool)] = [
            ("planet", .planet, false), ("planet with points", .planet, true), ("region", .region, false), ("organism", .organism, false),
        ]
        // The scenarios of the globe (the spec's 7.5): the globe with the minimap, mid-morph, the flat region and the organism drawn by the
        // globe path (the outlines), and the minimap alone over the planet level of the flat quad.
        typealias Scenario = (String, WorldLevel, Bool, Float?, Bool)
        let all: [Scenario] = scenarios.map { ($0.0, $0.1, $0.2, nil, false) } + [
            ("globe (s 1) with minimap", .planet, false, 1, true), ("morph (s 0.5)", .planet, false, 0.5, false),
            ("flat region with outlines (globe path)", .region, false, 0, false), ("minimap over the planet", .planet, false, nil, true),
        ]
        var results: [Result] = []
        for n in rows {
            for (name, level, points, globeS, minimap) in all {
                let r = try await measure(rows: n, scenario: name, level: level, points: points, draws: draws, size: (width, height), hz: hz,
                                          globeS: globeS, minimap: minimap)
                results.append(r)
                let gpu = r.gpuMs.map { String(format: "p50 %.2f p95 %.2f max %.2f", $0.p50, $0.p95, $0.max) } ?? "-"
                let cpu = r.cpuMs.map { String(format: "p50 %.2f p95 %.2f max %.2f", $0.p50, $0.p95, $0.max) } ?? "-"
                print("WorldRenderBench rows \(n) \(name): \(r.draws) draws, \(r.dropped) dropped; GPU ms \(gpu); CPU ms \(cpu)")
                #expect(r.draws > 0 && r.draws + r.dropped == draws, "every tick drew or was dropped")
                #expect((r.gpuMs?.max ?? 0) < 1000, "a draw that takes a second is a hang, not a measurement")
            }
        }
        let url = try outputDirectory().appendingPathComponent("world-render-bench.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(results).write(to: url)
        print("WorldRenderBench: wrote \(url.path)  (size \(width) x \(height) px at \(hz) Hz, \(draws) draws a scenario)")
    }
}

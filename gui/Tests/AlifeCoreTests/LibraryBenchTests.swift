import Foundation
import Testing

@testable import AlifeCore

/// The scan on a long library (docs/APP.md §5.3). The assertions are about work done (what is read again), not about seconds; the seconds
/// are printed. `ALIFE_LIBRARY_BENCH_SYNTH=DIR` builds 200 worlds in DIR (which must not exist) and measures them; `ALIFE_LIBRARY_BENCH_REAL=DIR`
/// measures an existing `runs/` folder read-only (directory listings and small JSON files; no lock is taken).
@Suite struct LibraryBenchTests {
    private static func ms(_ s: Double) -> String { String(format: "%.2f ms", s * 1000) }

    private static func measure(_ label: String, runs: URL, probeLocks: Bool) {
        let scanner = LibraryScanner(settle: 0)
        var out = "[library bench] \(label): "
        let cold = scanner.scan(runsDirectory: runs)
        let c = scanner.lastStats
        out += "\(c.listed) entries, \(c.read) read, \(c.offline) offline; cold scan \(ms(c.seconds))"
        var warm: [Double] = []
        for _ in 0..<20 {
            _ = scanner.scan(runsDirectory: runs)
            warm.append(scanner.lastStats.seconds)
        }
        warm.sort()
        out += "; warm scan median \(ms(warm[10])) max \(ms(warm[19])) (reread \(scanner.lastStats.read))"
        let t0 = DispatchTime.now().uptimeNanoseconds
        let lock: @Sendable (URL) -> Bool
        if probeLocks { lock = { PlayLockProbe.isHeld(directory: $0) } } else { lock = { _ in false } }
        let probed = scanner.probe(cold, runsDirectory: runs, lockHeld: lock)
        let t1 = DispatchTime.now().uptimeNanoseconds
        let state = LibraryState.resolve(probed)
        let t2 = DispatchTime.now().uptimeNanoseconds
        _ = LibraryState.tree(state.rows)
        _ = state.visible(filter: .all, query: "research planet")
        let t3 = DispatchTime.now().uptimeNanoseconds
        out += "; probe\(probeLocks ? "" : " (no locks)") \(ms(Double(t1 - t0) / 1e9)); resolve \(ms(Double(t2 - t1) / 1e9)) (\(state.rows.count) worlds, \(state.unfinished.count) unfinished, \(state.others.count) other); tree+search \(ms(Double(t3 - t2) / 1e9))"
        print(out)
    }

    @Test func aLongLibraryIsReadOnceAndThenOnlyStatted() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = scratch.url.appendingPathComponent("runs")
        try FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        try LibraryFixture.many(runs, count: 240)
        let scanner = LibraryScanner(settle: 0)
        _ = scanner.scan(runsDirectory: runs)
        #expect(scanner.lastStats.read == 240)
        _ = scanner.scan(runsDirectory: runs)
        #expect(scanner.lastStats.read == 0 && scanner.lastStats.reused == 240)
        // a minute of play on one world rereads one directory
        try LibraryFixture.write("x", "checkpoints/step_0000999999.safetensors", in: runs.appendingPathComponent("world_001"))
        _ = scanner.scan(runsDirectory: runs)
        #expect(scanner.lastStats.read == 1)
        Self.measure("synthetic, 240 directories", runs: runs, probeLocks: true)
    }

    @Test func measureAFolderOutsideTheTests() throws {
        let env = ProcessInfo.processInfo.environment
        if let synth = env["ALIFE_LIBRARY_BENCH_SYNTH"] {
            let runs = URL(fileURLWithPath: synth).appendingPathComponent("runs")
            try FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
            try LibraryFixture.many(runs, count: 200)
            Self.measure("synthetic, 200 directories, at \(synth)", runs: runs, probeLocks: true)
        }
        if let real = env["ALIFE_LIBRARY_BENCH_REAL"] {
            Self.measure("real folder \(real)", runs: URL(fileURLWithPath: real), probeLocks: false)
        }
    }
}

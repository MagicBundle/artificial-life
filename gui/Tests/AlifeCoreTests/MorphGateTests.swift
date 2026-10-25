import Foundation
import Testing
import simd

@testable import AlifeCore

/// The gate of build step 5 (spec 7.4): the strip on the real schedule and the three criteria of 7.1, written to a directory. Opt-in:
/// `ALIFE_MORPH_GATE_OUT=<dir> swift test --filter MorphGateTests`. A pure rasterisation of the mesh through `Morph.position`.
@Suite struct MorphGateTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ALIFE_MORPH_GATE_OUT"] != nil, "set ALIFE_MORPH_GATE_OUT=<directory>"))
    func theStripAndTheCriteria() throws {
        let out = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ALIFE_MORPH_GATE_OUT"]), isDirectory: true)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let raster = MorphRaster()
        var report = "morph gate: roll mode, schedule d_f=0.12L, a=0.15, w2 from 0.4, fill 0.80; 1100x800, mesh \(raster.count)x\(raster.count)\n"
        let targets: [Float] = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        for ybar in [0, 38, 80] as [Float] {
            var frames: [MorphRaster.Frame] = []
            for s in targets {
                let d = Morph.distance(forS: s, L: raster.L, fovY: raster.fovY, aspect: raster.aspect)
                let f = raster.render(distance: d, ybar: ybar, scale: 0.4, colour: true)
                frames.append(f)
                report += String(format: "ybar %3.0f  s %.2f  d %.4f L  coverage %5.1f%%\n", ybar, s, d / raster.L, f.coverage * 100)
            }
            try MorphRaster.writeSheet(frames, columns: 4, to: out.appendingPathComponent("strip_ybar\(Int(ybar)).png"))
            let c = raster.criteria(ybar: ybar)
            report += "ybar \(ybar): (a) worst centre-scale step \(c.worstScaleStep) | (b) coverage rise \(c.maxCoverageRise) pt, max drop \(c.maxCoverageDrop) pt per 0.5%, "
                + "98->60 in \(c.lines98to60) wheel lines, min coverage to s=0.3 \(c.minCoverageUpTo03) | (c) p99 displacement \(c.p99Displacement) px per 0.5% (\(c.steps) steps)\n"
        }
        try report.write(to: out.appendingPathComponent("criteria.txt"), atomically: true, encoding: .utf8)
        print(report)
    }
}

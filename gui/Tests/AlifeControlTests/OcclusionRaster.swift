import Foundation

/// One organism as a disc of the drawn picture, for the occlusion raster: the centre and radius in drawable pixels, and the depth
/// along the view axis (the nearer disc is drawn over the farther).
struct OrganismDisc {
    var x: Float, y: Float, r: Float, depth: Float
}

/// What the occlusion raster says about a set of discs.
struct Occlusion {
    /// The union of the discs, as a share of the picture.
    var coverage: Double
    /// The share of the discs wholly inside the picture that keep at least half of their area visible, and that are hidden.
    var halfVisible: Double
    var hidden: Double
    /// The discs counted, and each one's visible share (indexed like the input; NaN for a disc not wholly inside).
    var counted: Int
    var visible: [Float]
}


/// The occlusion raster of the single-organism acceptance (DESIGN §36.7, spec 1.9; the pure form of `review1/occl_probe.py`): the
/// discs are painted far to near into an id buffer, and each disc's visible share is the share of its pixels that still show its
/// id. Used by `DiagProbe` (the recorded planet) and the GPU legibility tests (a synthetic frame).
enum OcclusionRaster {
    /// The discs painted far to near into an id buffer; a disc wholly inside the picture counts, with the share of its pixels
    /// that still show its id.
    static func measure(_ discs: [OrganismDisc], width: Int, height: Int) -> Occlusion {
        var ids = [Int32](repeating: -1, count: width * height)
        let order = discs.indices.sorted { discs[$0].depth > discs[$1].depth }  // far first, near last = on top
        for i in order {
            let d = discs[i]
            let x0 = max(0, Int((d.x - d.r).rounded(.down))), x1 = min(width, Int((d.x + d.r).rounded(.up)) + 1)
            let y0 = max(0, Int((d.y - d.r).rounded(.down))), y1 = min(height, Int((d.y + d.r).rounded(.up)) + 1)
            guard x0 < x1, y0 < y1 else { continue }
            let r2 = d.r * d.r
            for y in y0..<y1 {
                let dy = Float(y) + 0.5 - d.y
                for x in x0..<x1 {
                    let dx = Float(x) + 0.5 - d.x
                    if dx * dx + dy * dy <= r2 { ids[y * width + x] = Int32(i) }
                }
            }
        }
        var visiblePixels = [Int](repeating: 0, count: discs.count)
        var covered = 0
        for id in ids where id >= 0 {
            visiblePixels[Int(id)] += 1
            covered += 1
        }
        var shares = [Float](repeating: .nan, count: discs.count)
        var counted = 0, half = 0, hidden = 0
        for (i, d) in discs.enumerated() where d.r <= d.x && d.x <= Float(width) - d.r && d.r <= d.y && d.y <= Float(height) - d.r {
            let share = Float(visiblePixels[i]) / (Float.pi * d.r * d.r)
            shares[i] = share
            counted += 1
            if share >= 0.5 { half += 1 }
            if visiblePixels[i] == 0 { hidden += 1 }
        }
        return Occlusion(
            coverage: Double(covered) / Double(width * height),
            halfVisible: counted > 0 ? Double(half) / Double(counted) : .nan,
            hidden: counted > 0 ? Double(hidden) / Double(counted) : .nan,
            counted: counted, visible: shares)
    }

    /// Whether no other disc comes within reach of each disc: the distance of the centres is over `r + 1.25 r' + 2` pixels (the
    /// nose of a neighbour reaches `1.25 r'` from its centre).
    static func isolated(_ discs: [OrganismDisc]) -> [Bool] {
        let cell: Float = 64
        var grid: [Int: [Int]] = [:]
        func key(_ x: Float, _ y: Float) -> Int { Int((x / cell).rounded(.down)) * 100_000 + Int((y / cell).rounded(.down)) }
        for (i, d) in discs.enumerated() { grid[key(d.x, d.y), default: []].append(i) }
        let largest = discs.map(\.r).max() ?? 0
        return discs.enumerated().map { i, d in
            let reach = Int(((d.r + 1.25 * largest + 2) / cell).rounded(.up))
            let cx = Int((d.x / cell).rounded(.down)), cy = Int((d.y / cell).rounded(.down))
            for gx in (cx - reach)...(cx + reach) {
                for gy in (cy - reach)...(cy + reach) {
                    for j in grid[gx * 100_000 + gy] ?? [] where j != i {
                        let o = discs[j]
                        let dx = o.x - d.x, dy = o.y - d.y
                        let need = d.r + 1.25 * o.r + 2
                        if dx * dx + dy * dy < need * need { return false }
                    }
                }
            }
            return true
        }
    }
}

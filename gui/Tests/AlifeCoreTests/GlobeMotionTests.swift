import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

@Suite("The motion that outlives the input, and the glow's grid")
struct GlobeMotionTests {
    private func drag(_ m: inout GlobeMotion, steps: Int, dx: CGFloat, dt: Double, hold: Double = 0, allowed: Bool = true) -> Bool {
        var t = 10.0
        m.begin(at: .zero, time: t)
        for k in 1...steps {
            t += dt
            m.move(to: CGPoint(x: dx * CGFloat(k), y: 0), time: t)
        }
        return m.end(time: t + hold, allowed: allowed)
    }

    @Test func aFlickStartsAnInertiaWithThePointersSpeedThatDecaysByTheTimeConstant() {
        var m = GlobeMotion()
        #expect(drag(&m, steps: 6, dx: 12, dt: 0.03))
        #expect(abs(m.velocity.x - 400) < 1e-3 && m.velocity.y == 0)
        var total: Float = 0
        for _ in 0..<600 { total += m.coast(dt: 1.0 / 60)?.x ?? 0 }
        #expect(!m.isCoasting)
        // the distance is v * tau (the exponential's integral), whatever the frame rate
        #expect(abs(total - 400 * GlobeMotion.timeConstant) < 4, "\(total)")
        var other = GlobeMotion()
        _ = drag(&other, steps: 6, dx: 12, dt: 0.03)
        var total30: Float = 0
        for _ in 0..<300 { total30 += other.coast(dt: 1.0 / 30)?.x ?? 0 }
        #expect(abs(total30 - total) < 4)
    }

    @Test func noFlickFromAShortASlowOrARestedDragOrWhereItIsNotAllowed() {
        var m = GlobeMotion()
        #expect(!drag(&m, steps: 2, dx: 40, dt: 0.03))
        #expect(!drag(&m, steps: 30, dx: 0.2, dt: 0.02))
        #expect(!drag(&m, steps: 6, dx: 12, dt: 0.03, hold: 0.2))
        #expect(!drag(&m, steps: 6, dx: 12, dt: 0.03, allowed: false))
        #expect(!m.isCoasting && m.coast(dt: 0.016) == nil)
    }

    @Test func theSpinRestsAfterAnInputAndGoesWestAtHalfADegreeASecond() {
        var m = GlobeMotion()
        m.stopCoasting(time: 50)
        #expect(!m.spins(enabled: true, time: 52, busy: false))
        #expect(m.spins(enabled: true, time: 50 + GlobeMotion.spinRest, busy: false))
        #expect(!m.spins(enabled: true, time: 60, busy: true) && !m.spins(enabled: false, time: 60, busy: false))
        #expect(abs(GlobeMotion.spinStep(dt: 1, size: 360) + 0.5) < 1e-6)
    }

    @Test func theGlowGridBinsTheRowsByLogCountAndWraps() {
        // 5 rows in one cell, 1 in another, 1 at the far corner wrapping to the first cell
        var pos: [Float] = []
        for _ in 0..<5 { pos += [1, 1, 0] }
        pos += [3.5, 1, 0]
        pos += [321, 1, 0]  // wraps to x = 1
        let grid = GlowHistogram.bin(positions: pos, n: 7, size: 320)
        #expect(grid.count == 160 * 160)
        #expect(grid[0] == GlowHistogram.byte(count: 6))  // cell (0, 0): the 5 and the wrapped one
        #expect(grid[1] == GlowHistogram.byte(count: 1))  // cell (1, 0): x in [2, 4)
        #expect(grid[5] == 0)
        #expect(GlowHistogram.byte(count: 16) == 255 && GlowHistogram.byte(count: 400) == 255)
        #expect(GlowHistogram.light(byte: GlowHistogram.byte(count: 4)) > 0.5, "a crowd of 4 keeps a floor of light")
        #expect(GlowHistogram.light(byte: 0) == 0)
        // complementary with the instances: glow + instances = 1 at every radius
        for r in stride(from: Float(0), through: 6, by: 0.5) {
            #expect(abs(OrganismSizePolicy.glowAlpha(radius: r) + OrganismSizePolicy.instanceAlpha(radius: r) - 1) < 1e-6)
        }
    }
}

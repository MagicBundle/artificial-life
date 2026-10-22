import Foundation
import Testing

@testable import AlifeCore

/// The wall-clock fade between two maps, with an injected clock (plain seconds), and the upload plan of a buffer set.
@Suite struct MapFadeTests {
    @Test func withNothingDeliveredTheMapIsShownWhole() {
        let fade = MapFade()
        #expect(fade.blend(at: 0) == 1 && fade.blend(at: 1000) == 1)
        #expect(!fade.isAnimating(at: 0))
    }

    @Test func aDeliveryStartsAFadeThatRunsForTheDuration() {
        var fade = MapFade()
        #expect(fade.deliver(at: 10) == .current, "none was running")
        #expect(fade.blend(at: 10) == 0)
        #expect(fade.isAnimating(at: 10.2))
        let quarter = fade.blend(at: 10 + MapFade.duration / 4)
        let half = fade.blend(at: 10 + MapFade.duration / 2)
        let threeQuarters = fade.blend(at: 10 + MapFade.duration * 0.75)
        #expect(0 < quarter && quarter < half && half < threeQuarters && threeQuarters < 1)
        #expect(abs(half - 0.5) < 1e-6)
        #expect(fade.blend(at: 10 + MapFade.duration) == 1 && !fade.isAnimating(at: 10 + MapFade.duration))
        #expect(MapFade.duration == 0.4)
    }

    @Test func aPausedWorldDeliversNothingAndTheFadeStaysFinished() {
        var fade = MapFade()
        _ = fade.deliver(at: 5)
        for t in stride(from: 6.0, to: 600.0, by: 37) {
            #expect(fade.blend(at: t) == 1 && !fade.isAnimating(at: t))
        }
    }

    @Test func aDeliveryDuringAFadeRestartsItFromTheNearerEnd() {
        var fade = MapFade()
        _ = fade.deliver(at: 0)
        // A third of the way: the picture is still nearer the previous map.
        #expect(fade.deliver(at: 0.3 * MapFade.duration) == .previous)
        #expect(fade.blend(at: 0.3 * MapFade.duration) == 0, "and the new fade starts at the beginning")
        // More than halfway: nearer the current one.
        #expect(fade.deliver(at: 0.3 * MapFade.duration + 0.8 * MapFade.duration) == .current)
        // A delivery after the fade has finished starts from the current map.
        #expect(fade.deliver(at: 100) == .current)
    }

    @Test func aNewWorldHasNoFade() {
        var fade = MapFade()
        _ = fade.deliver(at: 0)
        fade.reset()
        #expect(fade.blend(at: 0.1) == 1 && !fade.isAnimating(at: 0.1))
    }

    // MARK: Upload plan

    @Test func aSetThatHoldsWhatIsWantedWritesNothing() {
        #expect(MapUploadPlan.make(have: (5, 4), want: (5, 4)).isEmpty)
    }

    @Test func theFirstMapIsWrittenToBothTextureSets() {
        let plan = MapUploadPlan.make(have: (-1, -1), want: (1, 1))
        #expect(!plan.swap && plan.uploadCur && plan.uploadPrev)
    }

    @Test func aNewMapMovesTheOldCurrentToPreviousAndWritesOnlyTheNewOne() {
        // The set held (4, 3); the world now shows 5 over 4: swap, then write 5 into what held 3.
        let plan = MapUploadPlan.make(have: (4, 3), want: (5, 4))
        #expect(plan == MapUploadPlan(swap: true, uploadCur: true, uploadPrev: false))
    }

    @Test func aSetThatFellBehindWritesBoth() {
        let plan = MapUploadPlan.make(have: (2, 1), want: (9, 8))
        #expect(!plan.swap && plan.uploadCur && plan.uploadPrev)
    }

    @Test func aFadeRestartedFromThePreviousKeepsTheSameTexturesAndWritesTheNewOne() {
        // Delivery 6 landed early in the fade from 4 to 5: previous stays 4, current becomes 6.
        let plan = MapUploadPlan.make(have: (5, 4), want: (6, 4))
        #expect(!plan.swap && plan.uploadCur && !plan.uploadPrev)
    }

    @Test(arguments: 0..<200)
    func theCheapestPlanIsChosenAndLeavesTheSetHoldingWhatIsWanted(seed: Int) {
        var generator = SeededGenerator(seed: UInt64(seed))
        let have = (cur: Int.random(in: -1...6, using: &generator), prev: Int.random(in: -1...6, using: &generator))
        let want = (cur: Int.random(in: 0...6, using: &generator), prev: Int.random(in: 0...6, using: &generator))
        let plan = MapUploadPlan.make(have: have, want: want)
        // Apply it.
        var cur = have.cur, prev = have.prev
        if plan.swap { swap(&cur, &prev) }
        if plan.uploadCur { cur = want.cur }
        if plan.uploadPrev { prev = want.prev }
        #expect(cur == want.cur && prev == want.prev)
        let writes = (plan.uploadCur ? 1 : 0) + (plan.uploadPrev ? 1 : 0)
        let direct = (have.cur != want.cur ? 1 : 0) + (have.prev != want.prev ? 1 : 0)
        #expect(writes <= direct, "never worse than not swapping")
    }
}

/// A small deterministic generator for the randomised cases.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

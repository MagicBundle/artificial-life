import Foundation
import Testing

@testable import AlifeCore

@Suite struct WorldKeysTests {
    @Test func theTableIsTheDocumentedOne() {
        let expected: [(String, WorldAction)] = [
            (" ", .togglePause), (".", .step), ("c", .cycleColor), ("1", .toggleAgents), ("2", .toggleBonds),
            ("3", .cycleTerritory), ("5", .toggleStructures), ("6", .toggleSoil), ("f", .followSelection), ("[", .slower), ("]", .faster), ("r", .home),
            ("h", .home), ("n", .northUp), ("i", .toggleMinimap),
            ("t", .viewPreset(.top)), ("o", .viewPreset(.oblique)), ("l", .viewPreset(.low)), ("s", .viewPreset(.side)),
            ("\u{1B}", .escape), ("g", .continueRun), ("p", .showPowers), ("a", .flyToAlert), ("k", .acknowledgeAlert),
            ("\r", .confirmPower), ("\u{3}", .confirmPower),
        ]
        for (key, action) in expected { #expect(WorldKeys.action(for: key) == action, "\(key.debugDescription)") }
        #expect(Set(WorldKeys.characters) == Set(expected.map(\.0)))
        for key in WorldKeys.characters { #expect(WorldKeys.action(for: key) != nil) }
    }

    @Test func theStandardViewsHaveOneKeyEachAndNoKeyOfTheTableIsTakenTwice() {
        for preset in ViewPreset.allCases {
            #expect(WorldKeys.action(for: preset.key) == .viewPreset(preset), "\(preset)")
            #expect(WorldKeys.action(for: preset.key.uppercased()) == .viewPreset(preset), "a capital is the same key")
        }
        #expect(ViewPreset.allCases.map(\.key) == ["t", "o", "l", "s"])
        #expect(Set(WorldKeys.characters).count == WorldKeys.characters.count, "no key is listed twice")
        // The keys of A1 and before still do what they did (the new ones clash with none of them).
        for (key, action) in [(" ", WorldAction.togglePause), ("h", .home), ("r", .home), ("n", .northUp), ("i", .toggleMinimap), ("c", .cycleColor)] {
            #expect(WorldKeys.action(for: key) == action)
        }
    }

    @Test func aCapitalLetterIsTheSameKey() {
        #expect(WorldKeys.action(for: "F") == .followSelection && WorldKeys.action(for: "C") == .cycleColor)
        #expect(WorldKeys.action(for: "R") == .home && WorldKeys.action(for: "H") == .home)
        #expect(WorldKeys.action(for: "N") == .northUp && WorldKeys.action(for: "I") == .toggleMinimap)
    }

    @Test func theLabsFieldKeysAreNotTheWorldsKeys() {
        // `f` is the Lab's field slice, `[` and `]` its field layers, `4` its other field key: here follow and the rate, and
        // `4` does nothing.
        #expect(ViewerKeysLike.labBinds("f") && WorldKeys.action(for: "f") == .followSelection)
        #expect(WorldKeys.action(for: "4") == nil && WorldKeys.action(for: "x") == nil)
        #expect(WorldKeys.action(for: "+") == nil && WorldKeys.action(for: "") == nil && WorldKeys.action(for: "ab") == nil)
    }

    @Test func theDecideKeysAreTheSpecsAndEscNeverResumesOrAcknowledges() {
        // `Esc` has one meaning: one level back (DESIGN §34.7). It is none of Continue, Confirm or Acknowledge, which have keys of
        // their own, so a press meant to dismiss something cannot commit the player to the next 6,000 steps.
        #expect(WorldKeys.action(for: "\u{1B}") == .escape)
        for action in [WorldAction.continueRun, .confirmPower, .acknowledgeAlert, .togglePause] {
            #expect(WorldKeys.action(for: "\u{1B}") != action)
        }
        #expect(WorldKeys.action(for: "G") == .continueRun && WorldKeys.action(for: "P") == .showPowers)
        #expect(WorldKeys.action(for: "g", command: true) == nil && WorldKeys.action(for: "k", control: true) == nil)
    }

    @Test func aCommandOrControlChordIsNotOurs() {
        for key in WorldKeys.characters {
            #expect(WorldKeys.action(for: key, command: true) == nil)
            #expect(WorldKeys.action(for: key, control: true) == nil)
        }
    }

    @Test func colourModesCycleAndWrap() {
        let modes = ["tag", "energy", "clade", "house"]
        #expect(WorldKeys.nextColorMode(after: "tag", in: modes) == "energy")
        #expect(WorldKeys.nextColorMode(after: "house", in: modes) == "tag", "wraps")
        #expect(WorldKeys.nextColorMode(after: nil, in: modes) == "tag")
        #expect(WorldKeys.nextColorMode(after: "nope", in: modes) == "tag")
        #expect(WorldKeys.nextColorMode(after: "tag", in: []) == nil)
        #expect(WorldKeys.nextColorMode(after: "tag", in: ["tag"]) == "tag")
    }

    @Test func theRateKeysWalkTheRungs() {
        let rungs = PlayRate.rungs
        #expect(rungs == [.multiplier(0.25), .multiplier(0.5), .multiplier(1), .multiplier(2), .max])
        for (i, rung) in rungs.enumerated() {
            #expect(RateLadder.faster(than: rung) == (i + 1 < rungs.count ? rungs[i + 1] : nil), "faster than \(rung)")
            #expect(RateLadder.slower(than: rung) == (i > 0 ? rungs[i - 1] : nil), "slower than \(rung)")
        }
    }

    @Test func aRateBetweenTheRungsMovesToTheNeighbouringOne() {
        #expect(RateLadder.faster(than: .multiplier(0.75)) == .multiplier(1))
        #expect(RateLadder.slower(than: .multiplier(0.75)) == .multiplier(0.5))
        #expect(RateLadder.faster(than: .multiplier(8)) == .max)
        #expect(RateLadder.slower(than: .multiplier(8)) == .multiplier(2))
        #expect(RateLadder.slower(than: .multiplier(0.05)) == nil)
        #expect(RateLadder.faster(than: .multiplier(16)) == .max)
    }
}

/// What the Lab's table binds, for the comparison above (the Lab's own table is in the app target).
private enum ViewerKeysLike {
    static func labBinds(_ key: String) -> Bool { ["f", "4", "[", "]"].contains(key) }
}

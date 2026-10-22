import Foundation
import Testing

@testable import AlifeCore

/// The control lines of `alife play` (DESIGN §33.2.6). `tests/data/play_command_lines.txt` lists every line the window can
/// send; `tests/test_gui_sync.py` parses each with the engine's own parser, and this suite checks that `PlayCommand`
/// produces exactly those lines, so the two grammars cannot drift apart.
@Suite struct PlayCommandTests {
    private static let cases: [(PlayCommand, String)] = [
        (.pause, "pause"), (.run, "run"), (.step(count: 1), "step"), (.step(count: 25), "step 25"),
        (.rate(.multiplier(0)), "rate 0"), (.rate(.multiplier(0.05)), "rate 0.05"),
        (.rate(.multiplier(0.25)), "rate 0.25"), (.rate(.multiplier(0.5)), "rate 0.5"),
        (.rate(.multiplier(1)), "rate 1"), (.rate(.multiplier(2)), "rate 2"), (.rate(.multiplier(16)), "rate 16"),
        (.rate(.max), "rate max"),
        (.color("tag"), "color tag"), (.color("energy"), "color energy"), (.color("clade"), "color clade"),
        (.color("house"), "color house"), (.color("kin"), "color kin"), (.color("archetype"), "color archetype"),
        (.color("intent"), "color intent"),
        (.every(1), "every 1"), (.every(64), "every 64"), (.every(nil), "every auto"),
        (.focus(0), "focus 0"), (.focus(123_456_789_012), "focus 123456789012"),
        (.focus(Int64.max), "focus 9223372036854775807"), (.focus(nil), "focus off"),
        (.follow(7), "follow 7"), (.follow(nil), "follow off"),
        (.idnt, "idnt"), (.labels(true), "labels on"), (.labels(false), "labels off"), (.quit, "quit"),
    ]

    @Test(arguments: cases)
    func everyCommandIsItsLine(command: PlayCommand, line: String) throws {
        #expect(try command.line() == line)
        #expect(try command.encoded() == Data((line + "\n").utf8))
    }

    @Test func theLinesAreExactlyTheOnesTheEngineIsTestedWith() throws {
        let text = try String(contentsOf: Repo.streamData.appendingPathComponent("play_command_lines.txt"), encoding: .utf8)
        let shared = text.split(separator: "\n").map(String.init).filter { !$0.hasPrefix("#") && !$0.isEmpty }
        #expect(Set(shared) == Set(Self.cases.map(\.1)))
        #expect(shared.count == Self.cases.count, "a line is listed twice")
    }

    @Test(arguments: [
        PlayCommand.step(count: 0), .step(count: -3), .rate(.multiplier(-1)), .rate(.multiplier(0.04)),
        .rate(.multiplier(16.5)), .rate(.multiplier(.nan)), .rate(.multiplier(.infinity)),
        .every(0), .every(65), .every(-1), .focus(-1), .follow(-5),
        .color(""), .color("Tag"), .color("two words"), .color("kin\nquit"), .color("a;b"), .color("1kin"),
    ])
    func aCommandTheGrammarCannotExpressIsRefused(command: PlayCommand) {
        #expect(throws: PlayCommand.Invalid.self) { try command.line() }
    }

    @Test func aLineIsNeverLongerThanTheWorldAccepts() throws {
        let longest = PlayCommand.color(String(repeating: "a", count: PlayCommand.maxLineBytes - 6))
        #expect(try longest.line().utf8.count == PlayCommand.maxLineBytes)
        #expect(throws: PlayCommand.Invalid.self) {
            try PlayCommand.color(String(repeating: "a", count: PlayCommand.maxLineBytes - 5)).line()
        }
    }

    @Test func aDecimalHasNoExponentAndNoTrailingZeros() {
        #expect(PlayCommand.decimal(1) == "1")
        #expect(PlayCommand.decimal(0.25) == "0.25")
        #expect(PlayCommand.decimal(0.05) == "0.05")
        #expect(PlayCommand.decimal(16) == "16")
        #expect(PlayCommand.decimal(2.5) == "2.5")
        #expect(PlayCommand.decimal(0) == "0")
        #expect(PlayCommand.decimal(0.1 + 0.2) == "0.3")
    }

    @Test func theRungsAreTheOnesTheWindowOffers() {
        #expect(PlayRate.rungs == [.multiplier(0.25), .multiplier(0.5), .multiplier(1), .multiplier(2), .max])
        #expect(PlayRate.rungs.map(\.label) == ["0.25x", "0.5x", "1x", "2x", "max"])
        for rung in PlayRate.rungs { #expect(throws: Never.self) { try PlayCommand.rate(rung).line() } }
    }

    @Test func theClocksRateIsARate() {
        #expect(PlayRate(clock: -1) == .max)
        #expect(PlayRate(clock: 0) == .multiplier(0))
        #expect(PlayRate(clock: 2) == .multiplier(2))
        #expect(PlayRate(clock: .nan) == nil)
    }
}

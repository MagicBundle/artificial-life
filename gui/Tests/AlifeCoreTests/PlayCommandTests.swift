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
        (.autopause(.all(true)), "autopause on"), (.autopause(.all(false)), "autopause off"),
        (.autopause(.chapter(true)), "autopause chapter on"), (.autopause(.chapter(false)), "autopause chapter off"),
        (.autopause(.alert("pop_fall", true)), "autopause alert pop_fall on"),
        (.autopause(.alert("storm_named", false)), "autopause alert storm_named off"),
        (.watch(12), "watch 12"), (.watch(nil), "watch off"), (.unwatch(12), "unwatch 12"),
        (.ack(466_250_600), "ack 466250600"),
        (.power(PowerCommand(.preview, .food, id: 3, [.init(.mult, 1.3), .init(.steps, 6000)])),
         "power preview food id=3 mult=1.3 steps=6000"),
        (.power(PowerCommand(.apply, .food, id: 3, [.init(.mult, 1.3), .init(.steps, 6000)])),
         "power food id=3 mult=1.3 steps=6000"),
        (.power(PowerCommand(.apply, .bloom, id: 4, [.init(.patch, 2), .init(.gain, 1.5), .init(.steps, 3000)])),
         "power bloom id=4 patch=2 gain=1.5 steps=3000"),
        (.power(PowerCommand(.apply, .blight, id: 5, [.init(.patch, 0), .init(.steps, 12000)])),
         "power blight id=5 patch=0 steps=12000"),
        (.power(PowerCommand(.apply, .strike, id: 6,
                             [.init(.x, 100.5), .init(.y, 20), .init(.radius, 12.5), .init(.kill, 0.4)])),
         "power strike id=6 x=100.5 y=20 radius=12.5 kill=0.4"),
        (.power(PowerCommand(.cancel, .food, id: 3)), "power cancel food id=3"),
        (.intent("make more food"), "intent make more food"),
        (.saves(.all), "saves all"), (.saves(.newest), "saves newest"), (.saves(.single), "saves single"),
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
        .watch(-1), .unwatch(-2), .ack(-3),
        .autopause(.alert("", true)), .autopause(.alert("Pop", true)), .autopause(.alert("pop fall", false)),
        .autopause(.alert("pop\nquit", true)), .autopause(.alert("1pop", true)),
        .power(PowerCommand(.apply, .food, id: -1)),
        .power(PowerCommand(.apply, .food, id: 1, [.init(.mult, -1.3)])),
        .power(PowerCommand(.apply, .food, id: 1, [.init(.mult, .nan)])),
        .power(PowerCommand(.apply, .food, id: 1, [.init(.mult, .infinity)])),
        .power(PowerCommand(.apply, .food, id: 1, [.init(.steps, 6000.5)])),
        .power(PowerCommand(.apply, .food, id: 1, [.init(.patch, -1)])),
        // a whole number the line cannot hold: `Int64(_:)` trapped here (third review, stage 8)
        .power(PowerCommand(.apply, .food, id: 1, [.init(.steps, 1e30)])),
        .power(PowerCommand(.apply, .blight, id: 1, [.init(.patch, Double(Int64.max))])),
        .power(PowerCommand(.apply, .food, id: 1, [.init(.mult, 1.3), .init(.mult, 1.4)])),
        .power(PowerCommand(.cancel, .food, id: 1, [.init(.mult, 1.3)])),
        .intent(""), .intent("   "), .intent("two\nlines"), .intent("quit\rrun"),
    ])
    func aCommandTheGrammarCannotExpressIsRefused(command: PlayCommand) {
        #expect(throws: PlayCommand.Invalid.self) { try command.line() }
    }

    @Test func aWholeNumberJustUnderTwoToTheSixtyThirdIsWrittenWhole() throws {
        let top = Double(Int64.max).nextDown  // 9223372036854774784, the largest Double below 2^63
        #expect(try PlayCommand.power(PowerCommand(.apply, .food, id: 1, [.init(.steps, top)])).line()
            == "power food id=1 steps=9223372036854774784")
    }

    @Test func aLineIsNeverLongerThanTheWorldAccepts() throws {
        let longest = PlayCommand.color(String(repeating: "a", count: PlayCommand.maxLineBytes - 6))
        #expect(try longest.line().utf8.count == PlayCommand.maxLineBytes)
        #expect(throws: PlayCommand.Invalid.self) {
            try PlayCommand.color(String(repeating: "a", count: PlayCommand.maxLineBytes - 5)).line()
        }
    }

    @Test func aGameLineIsNeverLongerThanTheWorldAccepts() throws {
        let limit = PlayCommand.maxLineBytes - "intent ".utf8.count
        #expect(try PlayCommand.intent(String(repeating: "a", count: limit)).line().utf8.count == PlayCommand.maxLineBytes)
        #expect(throws: PlayCommand.Invalid.self) { try PlayCommand.intent(String(repeating: "a", count: limit + 1)).line() }
        #expect(try PlayCommand.intent("  padded  ").line() == "intent padded", "the text is trimmed, never altered inside")
        #expect(try PlayCommand.intent("Keep CASE").line() == "intent Keep CASE", "the free text keeps its case")
    }

    @Test func theSaveModesAreTheEnginesWords() {
        #expect(SaveMode.allCases.map(\.rawValue) == ["all", "newest", "single"])
        #expect(SaveMode(rawValue: "newest") == .newest && SaveMode(rawValue: "single") == .single && SaveMode(rawValue: "some") == nil)
    }

    @Test func theNamesAndKeysOfAPowerAreTheEnginesGrammar() {
        #expect(PowerCommand.Name.allCases.map(\.rawValue) == ["food", "bloom", "blight", "strike"])
        #expect(PowerCommand.Key.allCases.map(\.rawValue) == ["mult", "steps", "patch", "gain", "x", "y", "radius", "kill"])
        #expect(PowerCommand.Key.allCases.filter(\.isWhole).map(\.rawValue) == ["steps", "patch"])
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

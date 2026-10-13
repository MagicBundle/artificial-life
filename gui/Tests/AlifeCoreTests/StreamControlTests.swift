import Foundation
import Testing

@testable import AlifeCore

/// The control-line encoder against the grammar of DESIGN §27.11.
@Suite struct StreamControlTests {
    @Test(arguments: [
        (StreamCommand.pause, "pause"),
        (.run, "run"),
        (.step(count: 1), "step"),
        (.step(count: 2), "step 2"),
        (.step(count: 1_000_000), "step 1000000"),
        (.every(1), "every 1"),
        (.every(25), "every 25"),
        (.color("tag"), "color tag"),
        (.color("energy"), "color energy"),
        (.color("clade"), "color clade"),
        (.fields(true), "fields on"),
        (.fields(false), "fields off"),
        (.fieldLayer(0), "fields z 0"),
        (.fieldLayer(19), "fields z 19"),
        (.quit, "quit"),
    ])
    func linesFollowTheGrammar(command: StreamCommand, line: String) throws {
        #expect(try command.line() == line)
        #expect(try command.encoded() == Data((line + "\n").utf8))
    }

    @Test(arguments: [
        StreamCommand.step(count: 0), .step(count: -3), .every(0), .every(-1), .fieldLayer(-1), .color(""),
        .color("Tag"), .color("two words"), .color("tag\nquit"), .color("tag\r"), .color("é"), .color("a;b"),
    ])
    func commandsTheGrammarCannotExpressAreRefused(command: StreamCommand) {
        #expect(throws: StreamCommand.Invalid.self) { try command.line() }
        #expect(throws: StreamCommand.Invalid.self) { try command.encoded() }
    }

    @Test func aLineIsOneLineWithinTheLimit() throws {
        let commands: [StreamCommand] = [.pause, .run, .step(count: 9), .every(3), .color("energy"), .fields(true), .fieldLayer(5), .quit]
        for command in commands {
            let bytes = try command.encoded()
            #expect(bytes.last == 0x0A && bytes.filter { $0 == 0x0A }.count == 1)
            #expect(bytes.count - 1 <= StreamCommand.maxLineBytes)
        }
        let long = String(repeating: "a", count: StreamCommand.maxLineBytes)
        #expect(throws: StreamCommand.Invalid.self) { try StreamCommand.color(long).line() }
    }

    @Test func theColourModesOfAHeaderAreAcceptedAsTheyAre() throws {
        let fixture = try StreamFixture(kind: "3d")
        for case .header(let header) in try fixture.events() {
            for mode in header.colorModes { #expect(try StreamCommand.color(mode).line() == "color \(mode)") }
        }
    }
}

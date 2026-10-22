import Foundation
import Testing

@testable import AlifeCore

/// A played world's `config.json` says `run.director = "player"` (DESIGN §34.5): the Lab reads it as the director
/// being off, with the rest of the director's settings as written, and still refuses a mode the engine does not know.
@Suite struct DirectorPlayModeTests {
    @Test func thePlayerModeReadsAsOffAndKeepsTheOtherSettings() throws {
        let json = #"{"run": {"director": "player"}, "director": {"enabled": true, "persona": "trickster", "rule_every": 500}}"#
        let source = try DirectorSource(configJSON: Data(json.utf8))
        #expect(source.settings.mode == .off)
        #expect(source.enabled)
        #expect(source.settings.persona == .trickster)
        #expect(source.settings.ruleEvery == "500")
    }

    @Test func anUnknownModeIsStillRefused() {
        #expect(throws: LeverConfigError.self) {
            try DirectorSource(configJSON: Data(#"{"run": {"director": "nonsense"}}"#.utf8))
        }
        #expect(throws: Never.self) {
            try DirectorSource(configJSON: Data(#"{"run": {"director": "replay"}}"#.utf8))
        }
    }
}

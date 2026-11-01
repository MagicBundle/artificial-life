import Foundation
import Testing

@testable import AlifeCore

/// R5 of docs/APP.md: the sheet that stops the open world before another is played or a planet is made, and the time it says the save takes.
@Suite("Stopping one world to play another (R5)") struct WorldSwitchTests {
    @Test func theSheetSaysWhatIsStoppedWhyAndForHowLong() {
        let play = SwitchRequest(open: "mini3s_a", goal: .play("planet_b"), estimateSeconds: 10)
        #expect(play.title == "Stop \u{2018}mini3s_a\u{2019} to play \u{2018}planet_b\u{2019}?")
        #expect(play.reason == "Only one world can be played at a time.")
        #expect(play.detail == "\u{2018}mini3s_a\u{2019} is saved first, which takes about 10 seconds. Nothing is lost: you can play it again from Worlds whenever you like.")
        #expect(play.confirmLabel == "Stop and Play" && play.keepLabel == "Keep Playing")
        let planet = SwitchRequest(open: "a", goal: .makePlanet("p"))
        #expect(planet.title == "Stop \u{2018}a\u{2019} to make the planet \u{2018}p\u{2019}?")
        #expect(planet.confirmLabel == "Stop and Make Planet")
        #expect(planet.reason.contains("Only one world can be played at a time"))
        #expect(planet.estimateSeconds == SwitchWording.minimumSeconds)
    }

    @Test func whileItSavesTheSheetSaysWhatComesNext() {
        let play = SwitchRequest(open: "a", goal: .play("b"), estimateSeconds: 15)
        #expect(play.savingTitle == "Saving \u{2018}a\u{2019}\u{2026}")
        #expect(play.savingDetail(elapsed: 4) == "\u{2018}b\u{2019} opens as soon as it is saved. Waited 4 s; expected about 15 seconds.")
        #expect(SwitchRequest(open: "a", goal: .makePlanet("p")).savingDetail(elapsed: -3).hasPrefix("The planet \u{2018}p\u{2019} starts as soon as it is saved. Waited 0 s"))
        #expect(SwitchRequest.Goal.play("x").target == "x" && SwitchRequest.Goal.makePlanet("y").target == "y")
    }

    @Test func theEstimateFollowsTheSizeOfTheSaveAndIsAnEstimate() {
        #expect(SwitchWording.estimateSeconds(newestSaveBytes: nil, compressed: false) == 5, "nothing to measure: the minimum")
        #expect(SwitchWording.estimateSeconds(newestSaveBytes: 0, compressed: true) == 5)
        #expect(SwitchWording.estimateSeconds(newestSaveBytes: 10_000, compressed: false) == 5)
        // 686 MB plain: 3 + 6.86 = 9.9 s, up to a multiple of five
        #expect(SwitchWording.estimateSeconds(newestSaveBytes: 686_000_000, compressed: false) == 10)
        // 233 MB compressed: 3 + 11.65 = 14.65 s
        #expect(SwitchWording.estimateSeconds(newestSaveBytes: 233_000_000, compressed: true) == 15)
        // the same bytes cost more compressed than plain
        #expect(SwitchWording.estimateSeconds(newestSaveBytes: 400_000_000, compressed: true) > SwitchWording.estimateSeconds(newestSaveBytes: 400_000_000, compressed: false))
        // never past what a played world's own ladder allows
        #expect(SwitchWording.estimateSeconds(newestSaveBytes: 50_000_000_000, compressed: true) == SwitchWording.maximumSeconds)
        #expect(SwitchWording.estimateText(10) == "about 10 seconds" && SwitchWording.estimateText(0) == "about 1 seconds")
    }

    @Test func theSizeOfTheNewestSaveIsReadFromTheWorldsFolder() throws {
        let runs = FileManager.default.temporaryDirectory.appendingPathComponent("switch-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: runs) }
        let checkpoints = runs.appendingPathComponent("w/checkpoints")
        try FileManager.default.createDirectory(at: checkpoints, withIntermediateDirectories: true)
        try Data(count: 100).write(to: checkpoints.appendingPathComponent("step_0000006000.safetensors"))
        try Data(count: 300).write(to: checkpoints.appendingPathComponent("step_0000012000.safetensors.zst"))
        try Data(count: 900).write(to: checkpoints.appendingPathComponent("step_0000018000.safetensors.tmp"))  // being written: not a save
        let newest = try #require(SwitchWording.newestSave(of: "w", in: runs))
        #expect(newest.bytes == 300 && newest.compressed, "the newest by step, in whichever form")
        try Data(count: 50).write(to: checkpoints.appendingPathComponent("step_0000024000.safetensors"))
        let plain = try #require(SwitchWording.newestSave(of: "w", in: runs))
        #expect(plain.bytes == 50 && !plain.compressed)
        #expect(SwitchWording.newestSave(of: "none", in: runs) == nil)
        try FileManager.default.createDirectory(at: runs.appendingPathComponent("empty/checkpoints"), withIntermediateDirectories: true)
        #expect(SwitchWording.newestSave(of: "empty", in: runs) == nil)
    }
}

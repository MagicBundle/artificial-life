@testable import AlifeCore
import Foundation
import Testing

@testable import AlifeControl

/// What the controller adds for the decide half: the origin of the chapter numbering, the hooks for reports and plans, and the
/// acknowledgement of an alert (DESIGN §34.7). A stand-in script plays `alife play` with the G3b fixture.
@MainActor
@Suite struct PlayControllerDecideTests {
    private func standIn(_ scratch: Scratch) throws -> (URL, commands: URL) {
        let commands = scratch.url.appendingPathComponent("commands.txt")
        return (try scratch.script("alife-play-stand-in", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done"), commands)
    }

    private func commandLines(_ url: URL) -> [String] {
        ((try? String(contentsOf: url, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init)
    }

    private func writePlayFile(_ scratch: Scratch, run: String, madeAt: Int) throws {
        let dir = scratch.url.appendingPathComponent("runs/\(run)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let json = """
        {"play_version": 1, "base_rate": 14.4, "cadence": {"metrics_every": 250, "heavy_metrics_every": 1000, "checkpoint_every": 6000,
         "keep_checkpoints": 3}, "tick": 25, "planet": {"source": "s", "step": \(madeAt), "houses": "exact", "made_at_step": \(madeAt)}}
        """
        try json.write(to: dir.appendingPathComponent("play.json"), atomically: true, encoding: .utf8)
    }

    @Test func theChapterNumberingStartsAfterTheChapterAPlanetWasMadeIn() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        try writePlayFile(scratch, run: "r", madeAt: 450_000)
        let (script, commands) = try standIn(scratch)
        let controller = PlayController()
        defer { controller.shutdown() }
        controller.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                         arguments: [Repo.playFixtureURL("3d_g3b").path, commands.path])
        #expect(await eventually { controller.phase == .live })
        #expect(controller.originChapter == 75, "made_at_step / the header's chapter (6,000)")
        controller.stop()
        #expect(controller.originChapter == 0)
    }

    @Test func aWorldWithNoPlayFileOrNoPlanetCountsFromOne() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, commands) = try standIn(scratch)
        let controller = PlayController()
        defer { controller.shutdown() }
        controller.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                         arguments: [Repo.playFixtureURL("3d_g3b").path, commands.path])
        #expect(await eventually { controller.phase == .live })
        #expect(controller.originChapter == 0)
    }

    @Test func theHeaderChapterLengthIsWhatTheOriginIsDividedBy() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, commands) = try standIn(scratch)
        let controller = PlayController()
        defer { controller.shutdown() }
        var seen: [(URL, Int)] = []
        controller.readOrigin = { url, chapter in
            seen.append((url, chapter))
            return 7
        }
        controller.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                         arguments: [Repo.playFixtureURL("3d_g3b").path, commands.path])
        #expect(await eventually { controller.phase == .live })
        #expect(controller.originChapter == 7 && seen.count == 1 && seen[0].1 == 6000)
        #expect(seen[0].0.path.hasSuffix("runs/r/play.json"))
    }

    @Test func reportsAndPlansAreAnnouncedInOrderBeforeTheDeliveryThatCarriesThem() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, commands) = try standIn(scratch)
        let controller = PlayController()
        defer { controller.shutdown() }
        var reports: [PlayReport] = []
        var plans: [PlayPlan] = []
        controller.onReports = { reports += $0 }
        controller.onPlans = { plans += $0 }
        controller.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                         arguments: [Repo.playFixtureURL("3d_g3b").path, commands.path])
        #expect(await eventually { controller.phase == .live && controller.store.plans.records.count == 7 })
        // (chapter 1 as backlog, and chapter 2 in modes 0, 1 and 2)
        #expect(reports.count == 4 && reports.filter(\.isBacklog).count == 1)
        #expect(plans.map(\.plan.id) == [3, 3, 3, 3, 4, 3, 9] && plans.map(\.plan.status.rawValue)
                == ["preview", "queued", "applied", "since", "refused", "cancelled", "refused"])
    }

    @Test func acknowledgingAnAlertTakesItOffTheStackAndTellsTheWorld() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, commands) = try standIn(scratch)
        let controller = PlayController()
        defer { controller.shutdown() }
        controller.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                         arguments: [Repo.playFixtureURL("3d_g3b").path, commands.path])
        #expect(await eventually { controller.phase == .live && controller.store.alerts.alerts.count == 6 })
        let before = controller.revision
        #expect(controller.acknowledge(alert: 12_000_601))
        #expect(controller.revision == before + 1 && controller.store.alerts.acknowledged == [12_000_601])
        #expect(await eventually { commandLines(commands) == ["ack 12000601"] })
        #expect(!controller.acknowledge(alert: 7), "an alert it does not know is not acknowledged")
        try await Task.sleep(for: .milliseconds(100))
        #expect(commandLines(commands) == ["ack 12000601"])
    }
}

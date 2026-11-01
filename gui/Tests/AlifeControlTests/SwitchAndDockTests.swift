import AlifeCore
import AppKit
import Foundation
import Testing

@testable import AlifeControl

/// R5 (the sheet that stops one world to play another) and the Dock icon of a run (D3), with stand-in engines and a recording Dock. Nothing
/// here touches the real Dock tile or puts a window on a display.
@MainActor
@Suite("Switching worlds, and the Dock icon of a run", .serialized)
struct SwitchAndDockTests {
    // MARK: R5: a world is open, another is asked for

    /// A live world "a" in a stand-in alife folder whose engine takes `quitSeconds` to leave after `quit`, as a save does.
    @MainActor private final class SwitchRig {
        let scratch: Scratch
        let play = PlayController()
        let switcher: WorldSwitcher
        let repo: URL
        var ran = 0
        var order: [String] = []

        init(quitSeconds: Double = 0.4) throws {
            scratch = try Scratch()
            repo = scratch.url.appendingPathComponent("alife")
            let bin = repo.appendingPathComponent(".venv/bin")
            try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
            let checkpoints = repo.appendingPathComponent("runs/a/checkpoints")
            try FileManager.default.createDirectory(at: checkpoints, withIntermediateDirectories: true)
            // a 686 MB save point, sparse: the size is what the estimate reads
            let save = checkpoints.appendingPathComponent("step_0000006000.safetensors")
            FileManager.default.createFile(atPath: save.path, contents: nil)
            let handle = try FileHandle(forWritingTo: save)
            try handle.truncate(atOffset: 686_000_000)
            try handle.close()
            let engine = bin.appendingPathComponent("alife")
            let body = "#!/bin/bash\ncat \"\(Repo.playFixtureURL("3d_n70").path)\"\nwhile IFS= read -r line; do if [ \"$line\" = quit ]; then sleep \(quitSeconds); exit 0; fi; done\n"
            try body.write(to: engine, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: engine.path)
            let runs = repo.appendingPathComponent("runs")
            switcher = WorldSwitcher(play: play, runsDirectory: { runs })
            switcher.isQuitting = { false }
            switcher.poll = .milliseconds(20)
        }

        func openA() async -> Bool {
            play.start(.resume(PlayResume(run: "a")), executable: repo.appendingPathComponent(".venv/bin/alife"), directory: repo, arguments: [])
            return await eventually { play.phase == .live }
        }

        func finish() {
            play.shutdown()
            scratch.cleanup()
        }
    }

    @Test func noLiveWorldNeedsNoSheet() async throws {
        let rig = try SwitchRig()
        defer { rig.finish() }
        #expect(!rig.switcher.ask(.play("b")) { rig.ran += 1 }, "nothing is open")
        #expect(rig.switcher.request == nil && rig.ran == 0)
        #expect(await rig.openA())
        rig.play.stop()
        #expect(!rig.switcher.ask(.play("b")) { rig.ran += 1 }, "closed again")
    }

    @Test func aWorldThatIsStillStartingIsNotStoppedBySheet() async throws {
        let rig = try SwitchRig()
        defer { rig.finish() }
        let slow = try rig.scratch.script("slow", "sleep 5")
        rig.play.start(.resume(PlayResume(run: "a")), executable: slow, directory: rig.repo, arguments: [])
        #expect(rig.play.phase == .starting)
        #expect(!rig.switcher.ask(.play("b")) { rig.ran += 1 }, "the page says it is still starting; a sheet is for a live world")
        rig.play.stop()
    }

    @Test func askingPutsOneSheetUpWithTheWordsAndTheEstimateOfTheSave() async throws {
        let rig = try SwitchRig()
        defer { rig.finish() }
        #expect(await rig.openA())
        #expect(rig.switcher.ask(.play("b")) { rig.ran += 1 })
        let request = try #require(rig.switcher.request)
        #expect(request.open == "a" && request.goal == .play("b"))
        #expect(request.estimateSeconds == 10, "a 686 MB save point: 3 s and 6.9 s, up to a multiple of five")
        #expect(rig.switcher.phase == .asking && rig.ran == 0, "nothing runs until the person presses the button")
        #expect(rig.play.phase == .live, "and nothing was stopped")
        #expect(rig.switcher.ask(.play("c")) { rig.ran += 100 }, "a second request is taken by the sheet that is up")
        #expect(rig.switcher.request?.goal == .play("b") && rig.ran == 0, "and does not replace the first")
    }

    @Test func keepPlayingChangesNothing() async throws {
        let rig = try SwitchRig()
        defer { rig.finish() }
        #expect(await rig.openA())
        _ = rig.switcher.ask(.makePlanet("p")) { rig.ran += 1 }
        rig.switcher.keepPlaying()
        #expect(rig.switcher.request == nil && !rig.switcher.isBusy)
        #expect(rig.play.phase == .live && rig.play.world == "a")
        try await Task.sleep(for: .milliseconds(200))
        #expect(rig.ran == 0)
        // Escape on the asking sheet is the same
        _ = rig.switcher.ask(.play("b")) { rig.ran += 1 }
        rig.switcher.sheetDismissed()
        #expect(rig.switcher.request == nil && rig.play.phase == .live && rig.ran == 0)
    }

    @Test func confirmingSavesTheOpenWorldFirstAndOnlyThenStartsTheOther() async throws {
        let rig = try SwitchRig(quitSeconds: 0.6)
        defer { rig.finish() }
        #expect(await rig.openA())
        _ = rig.switcher.ask(.play("b")) { [unowned rig] in
            rig.order.append("start b, a still running: \(rig.play.hasProcess)")
            rig.ran += 1
        }
        rig.switcher.confirm()
        #expect(rig.switcher.phase == .saving && rig.play.retiredWorlds == ["a"], "it is being saved")
        #expect(rig.play.phase == .idle)
        try await Task.sleep(for: .milliseconds(250))
        #expect(rig.ran == 0, "not while a is still writing")
        #expect(await eventually { rig.ran == 1 })
        #expect(rig.order == ["start b, a still running: false"], "a had gone when b was started")
        #expect(rig.switcher.request == nil && !rig.switcher.isBusy)
        // confirming twice does nothing more
        rig.switcher.confirm()
        #expect(rig.ran == 1)
    }

    @Test func theWaitIsShownInSeconds() async throws {
        let rig = try SwitchRig(quitSeconds: 1.6)
        defer { rig.finish() }
        #expect(await rig.openA())
        _ = rig.switcher.ask(.play("b")) { rig.ran += 1 }
        rig.switcher.confirm()
        #expect(await eventually { rig.switcher.elapsed >= 1 })
        #expect(rig.switcher.request?.savingDetail(elapsed: rig.switcher.elapsed).contains("Waited \(rig.switcher.elapsed) s") == true)
        #expect(await eventually { rig.ran == 1 })
    }

    @Test func hidingTheSheetWhileItSavesStillOpensTheOtherWorld() async throws {
        let rig = try SwitchRig(quitSeconds: 0.5)
        defer { rig.finish() }
        #expect(await rig.openA())
        _ = rig.switcher.ask(.play("b")) { rig.ran += 1 }
        rig.switcher.confirm()
        rig.switcher.sheetDismissed()  // Escape, or the sheet ended by the quit's handling
        #expect(rig.switcher.request == nil && rig.switcher.isBusy, "the sheet is gone, the switch is not")
        #expect(await eventually { rig.ran == 1 })
        #expect(!rig.switcher.isBusy)
    }

    @Test func nothingStartsWhileTheAppIsQuitting() async throws {
        let rig = try SwitchRig(quitSeconds: 0.3)
        defer { rig.finish() }
        #expect(await rig.openA())
        rig.switcher.isQuitting = { true }
        _ = rig.switcher.ask(.play("b")) { rig.ran += 1 }
        rig.switcher.confirm()
        #expect(await eventually { !rig.switcher.isBusy })
        try await Task.sleep(for: .milliseconds(200))
        #expect(rig.ran == 0, "the quit's own wait is ending the world; what was asked for does not follow")
    }

    @Test func aMakePlanetFromTheSetUpPageWithAWorldOpenGoesThroughTheSheet() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try rig.makeSource("src")
        try LibraryTestFolders.played(rig.runs, "mini3s_a")
        await rig.scan()
        let switcher = WorldSwitcher.forApp(controller: rig.controller, play: rig.play)
        switcher.isQuitting = { false }
        rig.setup.switcher = switcher
        rig.play.start(.resume(PlayResume(run: "mini3s_a")), executable: rig.controller.alifeURL, directory: rig.repo)
        #expect(await patiently { rig.play.phase == .live })
        await rig.scan()
        rig.setup.openMakePlanet(from: "src", name: "planet-x")
        #expect(rig.setup.reasons.isEmpty, "a live world is no reason to grey the button any more: \(rig.setup.reasons)")
        rig.setup.start()
        #expect(switcher.request?.goal == .makePlanet("planet-x") && switcher.request?.open == "mini3s_a")
        #expect(rig.coordinator.plan == nil && rig.play.world == "mini3s_a" && rig.play.phase == .live, "nothing was stopped or started yet")
        switcher.confirm()
        #expect(await patiently { rig.play.world == "planet-x" && rig.play.isActive }, "the planet starts once the open world is saved")
        #expect(rig.coordinator.plan?.mode == .makePlanet)
        #expect(rig.playerStarts == 1)
        #expect(await rig.waitForState { $0 == .live })
    }

    @Test func aParkedPlanetPressedWhileAWorldIsOpenAsksToStopItAndThenMakesThePlanet() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        let switcher = WorldSwitcher.forApp(controller: rig.controller, play: rig.play)
        switcher.isQuitting = { false }
        rig.coordinator.hooks.switchWorld = { goal, then in switcher.ask(goal, then: then) }
        try LibraryTestFolders.played(rig.runs, "mini3s_a")
        await rig.scan()
        rig.play.start(.resume(PlayResume(run: "mini3s_a")), executable: rig.controller.alifeURL, directory: rig.repo)
        #expect(await patiently { rig.play.phase == .live })
        #expect(await rig.create("planet-1").isEmpty)
        #expect(await rig.waitForState { if case .parked = $0 { return true } else { return false } })
        #expect(rig.coordinator.plan?.state == .parked(.worldOpen("mini3s_a"), step: 30_000), "the second stage never replaces a world by itself")
        rig.coordinator.press(.makePlanet)
        #expect(switcher.request?.goal == .makePlanet("planet-1"), "the button asks")
        #expect(rig.play.world == "mini3s_a")
        switcher.keepPlaying()
        #expect(rig.coordinator.plan?.state == .parked(.worldOpen("mini3s_a"), step: 30_000) && rig.play.world == "mini3s_a")
        rig.coordinator.press(.makePlanet)
        switcher.confirm()
        #expect(await rig.waitForState { $0 == .live })
        #expect(rig.play.world == "planet-1")
    }

    // MARK: D3: the Dock icon

    @MainActor private final class RecordingDock: DockTileSurface {
        var progress: [(fraction: Double?, text: String)] = []
        var clears = 0
        var badges: [String?] = []
        var bounces = 0
        func showProgress(fraction: Double?, text: String) { progress.append((fraction, text)) }
        func clearProgress() { clears += 1 }
        func setBadge(_ text: String?) { badges.append(text) }
        func bounceIfInactive() { bounces += 1 }
    }

    private func exit(serial: Int, status: Int32 = 0, stopRequested: Bool = false, forced: Bool = false, byTimer: Bool = false,
                      signalled: Bool = false) -> RunExit {
        RunExit(run: "r", status: status, signalled: signalled, stopRequested: stopRequested, forced: forced, problem: nil, serial: serial, byTimer: byTimer)
    }

    @Test func aRunGoingShowsItsProgressAndTheBarGoesWhenItEnds() async throws {
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        let dock = RecordingDock()
        let controller = DockTileController(controller: rig.controller, surface: dock) { 30_000 }
        var step: Int? = nil
        controller.step = { step }
        controller.windowShown = { false }
        #expect(await rig.create("exp", goal: .research).isEmpty)
        #expect(await rig.waitForGrowing("exp"))
        #expect(await eventually { dock.progress.last?.text == "Growing" }, "the bar is up while it runs")
        #expect(dock.progress.last?.fraction == nil, "the run has said nothing yet: no number is made up")
        step = 15_000
        controller.refresh()
        #expect(dock.progress.last?.fraction == 0.5 && dock.progress.last?.text == "50%")
        step = 90_000
        controller.refresh()
        #expect(dock.progress.last?.fraction == 1 && dock.progress.last?.text == "100%")
        // the person stops it: the bar goes, and their own Stop is not news
        let before = dock.clears
        rig.controller.stop()
        #expect(await patiently { rig.controller.job == nil })
        #expect(await eventually { dock.clears > before })
        #expect(dock.badges.isEmpty && dock.bounces == 0)
    }

    @Test func aRunThatEndsOnItsOwnWithNoWindowBouncesOnceAndLeavesABadge() async throws {
        let rig = try PipelineRig(mode: "brief:ok")
        defer { rig.finish() }
        let dock = RecordingDock()
        let controller = DockTileController(controller: rig.controller, surface: dock) { 30_000 }
        var shown = false
        controller.windowShown = { shown }
        #expect(await rig.create("exp", goal: .research).isEmpty)
        #expect(await rig.waitForGrowing("exp"))
        #expect(await patiently { rig.controller.job == nil })
        #expect(await eventually { dock.bounces == 1 }, "one bounce")
        #expect(dock.badges.last == "Done" && dock.clears >= 1)
        try await Task.sleep(for: .milliseconds(300))
        #expect(dock.bounces == 1, "once")
        // the window comes back: the badge goes
        shown = true
        NotificationCenter.default.post(name: NSApplication.didBecomeActiveNotification, object: NSApp)
        #expect(await eventually { dock.badges.last == .some(nil) })
        NotificationCenter.default.post(name: NSApplication.didBecomeActiveNotification, object: NSApp)
        try await Task.sleep(for: .milliseconds(150))
        #expect(dock.badges.count == 2, "and is not cleared twice")
        _ = controller
    }

    @Test func aFailedRunSaysFailedAndOneEndedWithTheWindowShownLeavesNoBadge() async throws {
        let rig = try PipelineRig(mode: "refuse:ok")
        defer { rig.finish() }
        let dock = RecordingDock()
        let controller = DockTileController(controller: rig.controller, surface: dock) { nil }
        controller.windowShown = { false }
        #expect(await rig.create("exp", goal: .research).isEmpty)
        #expect(await patiently { rig.controller.lastRunExit != nil })
        #expect(await eventually { dock.badges.last == "Failed" })
        let shownDock = RecordingDock()
        let again = DockTileController(controller: rig.controller, surface: shownDock) { nil }
        again.windowShown = { true }
        again.ended(exit(serial: 99))
        #expect(shownDock.badges.isEmpty, "the person is looking at the window: no badge")
        #expect(shownDock.bounces == 1, "the bounce is asked for all the same; the system ignores it for an app that is in front")
        _ = controller
    }

    @Test func theEndsThatAreTheirOwnOrNewsAreTold() {
        let controller = RunController(storedPath: "", savePath: { _ in }, candidates: [], isApp: false)
        let dock = RecordingDock()
        let tile = DockTileController(controller: controller, surface: dock)
        tile.windowShown = { false }
        tile.ended(exit(serial: 1, status: 130, stopRequested: true))
        #expect(dock.badges.isEmpty && dock.bounces == 0, "Stop: their own")
        tile.ended(exit(serial: 1, status: 130, byTimer: true))
        #expect(dock.badges.isEmpty && dock.bounces == 0, "an exit is counted once, whatever it says the second time")
        tile.ended(exit(serial: 2, status: 130, stopRequested: true, byTimer: true))
        #expect(dock.badges == ["Done"] && dock.bounces == 1, "the stop-after timer is not theirs")
        tile.ended(exit(serial: 3, status: 15, stopRequested: true, forced: true, signalled: true))
        #expect(dock.badges == ["Done", "Failed"] && dock.bounces == 2, "a Force stop ends a run without a clean save: that is told")
        tile.ended(exit(serial: 4, status: 4))
        #expect(dock.badges.last == "Failed")
        tile.ended(exit(serial: 5, status: 0, signalled: true))
        #expect(dock.badges.last == "Failed")
    }

    @Test func theRunsTargetIsReadFromItsOwnConfig() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let dir = scratch.url.appendingPathComponent("runs/exp")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        #expect(DockTileController.configTarget(run: "exp", repo: scratch.url) == nil, "no config yet")
        try #"{"run": {"steps": 40000}}"#.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        #expect(DockTileController.configTarget(run: "exp", repo: scratch.url) == 40_000)
        try #"{"run": {"steps": 0}}"#.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        #expect(DockTileController.configTarget(run: "exp", repo: scratch.url) == nil)
        try "not json".write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        #expect(DockTileController.configTarget(run: "exp", repo: scratch.url) == nil)
    }

    @Test func theRealDockTileViewDrawsABarThatGrows() throws {
        let view = DockProgressView(frame: NSRect(x: 0, y: 0, width: 128, height: 128))
        func pixels(_ fraction: Double?) throws -> NSBitmapImageRep {
            view.fraction = fraction
            view.text = "x"  // (the same in every picture, so that only the bar differs)
            let rep = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
            view.cacheDisplay(in: view.bounds, to: rep)
            if let out = ProcessInfo.processInfo.environment["ALIFE_SHELL_SHOTS_OUT"].map({ URL(fileURLWithPath: $0) }) {
                try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
                let name = fraction.map { "e-dock-tile-\(Int($0 * 100))" } ?? "e-dock-tile-unknown"
                try rep.representation(using: .png, properties: [:])?.write(to: out.appendingPathComponent("\(name).png"))
            }
            return rep
        }
        let empty = try pixels(0), half = try pixels(0.5), full = try pixels(1), unknown = try pixels(nil)
        /// How many pixels of the row along the bar's foot differ from the empty bar's.
        func changed(_ rep: NSBitmapImageRep) -> Int {
            let scale = rep.pixelsHigh / 128
            let y = rep.pixelsHigh - 10 * scale  // the view's origin is at the bottom: ten points up is inside the bar, under its text
            var n = 0
            for x in 0..<rep.pixelsWide {
                guard let a = rep.colorAt(x: x, y: y)?.usingColorSpace(.sRGB), let b = empty.colorAt(x: x, y: y)?.usingColorSpace(.sRGB) else { continue }
                if abs(a.redComponent - b.redComponent) + abs(a.greenComponent - b.greenComponent) + abs(a.blueComponent - b.blueComponent) > 0.2 { n += 1 }
            }
            return n
        }
        #expect(changed(half) > 10 && changed(full) > changed(half), "\(changed(half)) \(changed(full))")
        #expect(changed(unknown) == 0, "no fraction, no fill")
    }
}

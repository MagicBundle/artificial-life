import AlifeCore
import AppKit
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// The new dialogs and states of PR-E, drawn by AppKit in windows that are never shown, in light and dark. With `ALIFE_SHELL_SHOTS_OUT=DIR`
/// the pictures are also written to DIR as PNG, for looking at.
@MainActor
@Suite("The safety rules, drawn", .serialized)
struct SessionViewTests {
    private var out: URL? { ProcessInfo.processInfo.environment["ALIFE_SHELL_SHOTS_OUT"].map { URL(fileURLWithPath: $0) } }
    private var everything: Bool { out != nil }

    private func spin() { RunLoop.current.run(until: Date().addingTimeInterval(0.01)) }

    private func colours(_ rep: NSBitmapImageRep) -> Int {
        var seen = Set<UInt32>()
        let step = max(1, rep.pixelsWide / 200)
        for y in stride(from: 0, to: rep.pixelsHigh, by: step) {
            for x in stride(from: 0, to: rep.pixelsWide, by: step) {
                guard let c = rep.colorAt(x: x, y: y)?.usingColorSpace(.sRGB) else { continue }
                seen.insert(UInt32(c.redComponent * 31) << 10 | UInt32(c.greenComponent * 31) << 5 | UInt32(c.blueComponent * 31))
            }
        }
        return seen.count
    }

    private func save(_ rep: NSBitmapImageRep, _ name: String, dark: Bool) throws {
        guard let out else { return }
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        try rep.representation(using: .png, properties: [:])?.write(to: out.appendingPathComponent("\(name)-\(dark ? "dark" : "light").png"))
    }

    /// A view drawn at its size in a borderless window far off every display.
    private func draw(_ view: some View, size: CGSize, dark: Bool, name: String, background: Bool = true) throws -> NSBitmapImageRep {
        let root = view.frame(width: size.width, height: size.height).background(Color(nsColor: .windowBackgroundColor))
        let host = NSHostingView(rootView: root)
        let window = QuietWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: true)
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.contentView = host
        host.frame = NSRect(origin: .zero, size: size)
        window.alphaValue = 0
        window.ignoresMouseEvents = true
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        window.orderFrontRegardless()
        defer { window.orderOut(nil) }
        let rep = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        for turn in 0..<20 {
            host.layoutSubtreeIfNeeded()
            window.displayIfNeeded()
            spin()
            guard turn >= 4 else { continue }
            host.cacheDisplay(in: host.bounds, to: rep)
            if colours(rep) > 12 { break }
        }
        try save(rep, name, dark: dark)
        return rep
    }

    // MARK: The switching sheet

    @Test(arguments: [false, true])
    func theSwitchingSheetAsksAndThenWaits(dark: Bool) async throws {
        guard everything || !dark else { return }
        let play = PlayController()
        let switcher = WorldSwitcher(play: play)
        let asking = SwitchRequest(open: "mini3s_a", goal: .play("planet_b"), estimateSeconds: 10)
        let a = try draw(SwitchSheet(switcher: switcher, request: asking), size: CGSize(width: 470, height: 190), dark: dark, name: "e-switch-asking")
        #expect(colours(a) > 8)
        let planet = SwitchRequest(open: "mini3s_a", goal: .makePlanet("planet_c"), estimateSeconds: 15)
        let p = try draw(SwitchSheet(switcher: switcher, request: planet), size: CGSize(width: 470, height: 190), dark: dark, name: "e-switch-planet")
        #expect(colours(p) > 8)
    }

    @Test(arguments: [false, true])
    func theSheetWhileTheWorldSavesShowsTheWait(dark: Bool) async throws {
        guard everything || !dark else { return }
        let rig = try SwitchRigLite()
        defer { rig.finish() }
        #expect(await rig.openA())
        _ = rig.switcher.ask(.play("planet_b")) {}
        rig.switcher.confirm()
        #expect(rig.switcher.phase == .saving)
        let request = try #require(rig.switcher.request)
        let rep = try draw(SwitchSheet(switcher: rig.switcher, request: request), size: CGSize(width: 470, height: 150), dark: dark, name: "e-switch-saving")
        #expect(colours(rep) > 8)
        #expect(await eventually(timeout: .seconds(20)) { !rig.switcher.isBusy })
    }

    /// A live world "a" whose engine takes a moment to leave after `quit`.
    @MainActor private final class SwitchRigLite {
        let scratch: Scratch
        let play = PlayController()
        let switcher: WorldSwitcher
        init() throws {
            scratch = try Scratch()
            let engine = try scratch.script("alife", "cat \"\(Repo.playFixtureURL("3d_n70").path)\"\nwhile IFS= read -r line; do if [ \"$line\" = quit ]; then sleep 2; exit 0; fi; done")
            _ = engine
            switcher = WorldSwitcher(play: play)
            switcher.isQuitting = { false }
            switcher.windowShown = { true }
        }
        func openA() async -> Bool {
            play.start(.resume(PlayResume(run: "a")), executable: scratch.url.appendingPathComponent("alife"), directory: scratch.url, arguments: [])
            return await eventually { play.phase == .live }
        }
        func finish() {
            play.shutdown()
            scratch.cleanup()
        }
    }

    // MARK: The close question

    @Test(arguments: [false, true])
    func theCloseQuestionDrawsWithItsThreeButtonsAndTheCheckbox(dark: Bool) throws {
        guard everything || !dark else { return }
        for (name, facts) in [("e-close-run", CloseFacts(runGoing: true, runName: "exp-1")),
                              ("e-close-run-world", CloseFacts(runGoing: true, runName: "planet-1-grow", world: "mini3s_a"))] {
            let alert = QuitPanel.closeAlert(SessionRules.question(facts))
            #expect(alert.buttons.map(\.title) == ["Keep Running", "Stop and Quit", "Cancel"], "the first is the default, the last answers Escape")
            #expect(alert.buttons[0].keyEquivalent == "\r" && alert.buttons[2].keyEquivalent == "\u{1b}")
            #expect(alert.buttons[1].hasDestructiveAction && !alert.buttons[0].hasDestructiveAction)
            #expect(alert.showsSuppressionButton && alert.suppressionButton?.title == "Remember Keep Running (change it in Settings)")
            alert.layout()
            let window = alert.window
            window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
            guard let content = window.contentView else { Issue.record("no content"); return }
            content.layoutSubtreeIfNeeded()
            let rep = try #require(content.bitmapImageRepForCachingDisplay(in: content.bounds))
            content.cacheDisplay(in: content.bounds, to: rep)
            try save(rep, name, dark: dark)
            #expect(colours(rep) > 4)
        }
    }

    @Test(arguments: [false, true])
    func theQuestionForAPlanetBeingMadeHasTwoButtonsAndNoCheckbox(dark: Bool) throws {
        guard everything || !dark else { return }
        let facts = CloseFacts(runGoing: false, world: "planet-2", worldStarting: true, planetBeingMade: true)
        guard case .ask(let question) = SessionRules.decideClose(facts, preference: .keep) else {
            Issue.record("a planet being made asks, whatever was remembered")
            return
        }
        let alert = QuitPanel.closeAlert(question)
        #expect(alert.buttons.map(\.title) == ["Cancel", "Stop and Quit"], "Cancel is the default and answers Escape; no Keep Running exists here")
        #expect(alert.buttons[1].hasDestructiveAction && !alert.showsSuppressionButton, "and nothing here can be remembered")
        alert.layout()
        let window = alert.window
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        guard let content = window.contentView else { Issue.record("no content"); return }
        content.layoutSubtreeIfNeeded()
        let rep = try #require(content.bitmapImageRepForCachingDisplay(in: content.bounds))
        content.cacheDisplay(in: content.bounds, to: rep)
        try save(rep, "e-close-making", dark: dark)
        #expect(colours(rep) > 4)
    }

    // MARK: Settings

    @Test(arguments: [false, true])
    func settingsShowTheTwoNewChoicesWithTheirEffectLines(dark: Bool) throws {
        guard everything || !dark else { return }
        let controller = RunController(storedPath: "", savePath: { _ in }, candidates: [], isApp: false)
        for (suffix, value) in [("ask", "ask"), ("keep", "keep")] {
            UserDefaults.standard.set(value, forKey: AppShell.closeWithRunKey)
            UserDefaults.standard.set(suffix == "ask", forKey: AppShell.pickUpKey)
            defer {
                UserDefaults.standard.removeObject(forKey: AppShell.closeWithRunKey)
                UserDefaults.standard.removeObject(forKey: AppShell.pickUpKey)
            }
            let rep = try draw(GeneralSettingsView(controller: controller), size: CGSize(width: 540, height: 620), dark: dark, name: "e-settings-\(suffix)")
            #expect(colours(rep) > 10)
        }
    }

    // MARK: Pages

    /// The shell over a stand-in alife folder with one played world open.
    private func drawShell(_ page: ShellPage, dark: Bool, name: String, closedNote: String? = nil, away: Bool = false, run: Bool = false) async throws {
        let rig = try PipelineRig(mode: run ? "long:ok" : "ok:ok", playFixture: "3d_n70")
        defer { rig.finish() }
        try LibraryTestFolders.played(rig.runs, "mini3s_a")
        try LibraryTestFolders.played(rig.runs, "planet_b")
        await rig.scan()
        let suite = "alife-sessionview-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let session = WorldSession(play: rig.play, defaults: defaults)
        let shell = ShellModel(session: session, defaults: defaults)
        if away || run {
            if away {
                rig.play.start(.resume(PlayResume(run: "mini3s_a")), executable: rig.controller.alifeURL, directory: rig.repo)
                #expect(await patiently { rig.play.phase == .live })
            }
            if run {
                #expect(await rig.create("exp-1", goal: .research).isEmpty)
                #expect(await rig.waitForGrowing("exp-1"))
            }
            await rig.scan()
        }
        shell.select(page)
        if away {
            // the world is running and the player has gone to another page: the app paused it
            var running = AwayPause.Facts(world: "mini3s_a", live: true, rate: 1, reason: .running)
            running.requestedPause = false
            _ = session.model.away.leftWatch(running)
            session.model.awayHold = session.model.away.hold
            shell.select(.worlds)
        }
        shell.closedNote = closedNote
        let size = CGSize(width: AppShell.minWidth, height: AppShell.minHeight)
        let host = NSHostingView(rootView: AppShellView(shell: shell, controller: rig.controller, library: rig.library, setup: rig.setup, arguments: []))
        let window = QuietWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: true)
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.contentView = host
        host.frame = NSRect(origin: .zero, size: size)
        window.alphaValue = 0
        window.ignoresMouseEvents = true
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        window.orderFrontRegardless()
        defer { window.orderOut(nil) }
        let rep = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        for turn in 0..<40 {
            host.layoutSubtreeIfNeeded()
            window.displayIfNeeded()
            spin()
            guard turn >= 6 else { continue }
            host.cacheDisplay(in: host.bounds, to: rep)
            if colours(rep) > 25 { break }
        }
        try save(rep, name, dark: dark)
        #expect(colours(rep) > 25, "\(name): the page is blank")
        if away {
            let cue = session.model.watchCue(startFailed: false)
            #expect(cue.text == "mini3s_a · paused: you left", "\(cue.text) phase \(rig.play.phase) hold \(session.model.awayHold)")
        }
        // (the run ends before the rig goes: nothing may answer to a rig that is gone)
        rig.controller.forceStop()
        rig.play.shutdown()
        #expect(await patiently { rig.controller.job == nil })
    }

    @Test(arguments: [false, true])
    func theWorldsPageSaysWhatBecameOfTheWorldWhenTheWindowWasClosed(dark: Bool) async throws {
        guard everything || !dark else { return }
        try await drawShell(.worlds, dark: dark, name: "e-worlds-closed-note",
                            closedNote: SessionRules.closedWorldNote("mini3s_a"))
    }

    @Test(arguments: [false, true])
    func theWorldsPageAndTheSidebarSayTheWorldIsPausedBecauseThePlayerLeft(dark: Bool) async throws {
        guard everything || !dark else { return }
        try await drawShell(.worlds, dark: dark, name: "e-worlds-paused-away", away: true, run: true)
    }

    @Test(arguments: [false, true])
    func activityShowsThePausedWorldNextToTheRunAndTheGPUNote(dark: Bool) async throws {
        guard everything || !dark else { return }
        try await drawShell(.activity, dark: dark, name: "e-activity-paused-away", away: true, run: true)
    }

    @Test(arguments: [false, true])
    func watchSaysItIsPausedBecauseThePlayerLeftAndOffersRun(dark: Bool) async throws {
        guard everything || !dark else { return }
        // (the map is a Metal view and is not drawn here: the notes over it are)
        let rig = try PipelineRig(mode: "long:ok")
        defer { rig.finish() }
        try LibraryTestFolders.played(rig.runs, "mini3s_a")
        await rig.scan()
        let suite = "alife-sessionview-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let session = WorldSession(play: rig.play, defaults: defaults)
        let shell = ShellModel(session: session, defaults: defaults)
        rig.play.start(.resume(PlayResume(run: "mini3s_a")), executable: rig.controller.alifeURL, directory: rig.repo)
        #expect(await patiently { rig.play.phase == .live })
        #expect(await rig.create("exp-1", goal: .research).isEmpty)
        #expect(await rig.waitForGrowing("exp-1"))
        session.model.away = AwayPause()
        _ = session.model.away.leftWatch(AwayPause.Facts(world: "mini3s_a", live: true, rate: 1, reason: .running))
        _ = session.model.away.returned(AwayPause.Facts(world: "mini3s_a", live: true, rate: 0, reason: .userPause), pickUp: false)
        session.model.awayHold = session.model.away.hold
        #expect(session.model.awayHold == .left(world: "mini3s_a"))
        let rep = try draw(
            WatchChrome(shell: shell, play: rig.play, model: session.model, controller: rig.controller)
                .background(Color(nsColor: .underPageBackgroundColor)),
            size: CGSize(width: 900, height: 120), dark: dark, name: "e-watch-notes")
        #expect(colours(rep) > 8)
        rig.controller.forceStop()
        rig.play.shutdown()
        #expect(await patiently { rig.controller.job == nil })
    }
}

@testable import AlifeCore
import AppKit
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// The pages of PR-F, drawn by AppKit in windows that are never shown, at the window's minimum size (1,100 x 700), in light and dark: Details
/// of a finished research world, of a growing one, of a played world and of nothing; the research Watch with its Run panel in each state that
/// does not need a Metal device (the picture itself is the running app's); Help; Settings > Autopilot. With `ALIFE_SHELL_SHOTS_OUT=DIR` the
/// pictures are also written to DIR as PNG, for looking at.
@MainActor
@Suite("Watch and Details, drawn", .serialized)
struct WatchDetailsViewTests {
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

    // MARK: Folders

    private static let metricsHeavy = #"{"step": 30000, "n_alive": 992.0, "steps_per_s": 347.1, "clades_alive": 25, "generation_max": 44, "ledger_error": 0, "height_mean": 6.07, "grounded_frac": 0.248, "surface_frac": 0.082, "arms_hits": 12, "arms_kills": 3, "bond_total": 449.5, "build_blocks": 40}"#

    private static let config = """
        {"run": {"seed": 1, "steps": 30000}, "world": {"dims": 3, "geometry": "sphere", "radius": 36, "height": 16, "nside": 8, "layers": 8, "size": 1024},
         "mutation": {"rate_min": 0.001, "sigma_min": 0.002}, "physics": {"gravity": 0.03, "buoyancy_max": 0.04},
         "weather": {"enabled": false},
         "director": {"persona": "gardener", "rule_every": 1000, "llm_every": 0, "enabled": true}}
        """

    private func world(_ runs: URL, _ name: String, parent: String? = nil, step: Int = 30_000, status: String = "done", autopilot: Bool = true,
                       metrics: Bool = true, flat: Bool = false) throws {
        let dir = runs.appendingPathComponent(name)
        var meta = #"{"status": "\#(status)", "start_step": 0, "created_utc": "2026-11-01T16:29:19Z""#
        if let parent { meta += #", "parent": "\#(parent)""# }
        try LibraryTestFolders.write(meta + "}", "meta.json", in: dir)
        try LibraryTestFolders.write(flat ? #"{"run": {"seed": 5, "steps": 20000}, "world": {"dims": 2, "geometry": "box", "size": 256}, "mutation": {"rate_min": 0.001}}"# : Self.config,
                                     "config.json", in: dir)
        try LibraryTestFolders.write("x", "checkpoints/step_\(String(format: "%010d", step)).safetensors", in: dir)
        try LibraryTestFolders.write("{}", "events/chunk_0.json", in: dir)
        if metrics { try LibraryTestFolders.write(Self.metricsHeavy + "\n", "metrics.jsonl", in: dir) }
        if autopilot {
            let log = #"{"kind": "header", "persona": "gardener"}"# + "\n"
                + #"{"step_applied": 12000, "source": "rule", "action": "regen_pulse", "params": {"gain": 1.5, "radius": 8.25}, "reason": "the population fell by 18%"}"# + "\n"
                + #"{"step_applied": 21000, "source": "llm", "action": "season", "params": {"period": 4000}, "reason": "a slow season keeps the lineages apart"}"# + "\n"
                + #"{"step_applied": 24000, "source": "revert", "action": "season", "params": {}, "reason": "the energy check was off"}"# + "\n"
            try LibraryTestFolders.write(log, "director.jsonl", in: dir)
            let journal = #"{"step": 12000, "text": "regen_pulse: the population fell 18% below its 5,000-step mean"}"# + "\n"
                + #"{"step": 21000, "kind": "llm_call", "latency_s": 2.4, "outcome": "ok", "model": "m", "journal": "A slow season should keep the families apart.", "accepted": ["season"], "rejected": [], "messages": [{"role": "user", "content": "state"}], "response": "{}"}"# + "\n"
            try LibraryTestFolders.write(journal, "journal.jsonl", in: dir)
        }
    }

    /// A folder with the worlds each page of this PR meets.
    private func folder(_ scratch: Scratch) throws -> String {
        let root = scratch.url.appendingPathComponent("the-alife-folder")
        let engine = root.appendingPathComponent(".venv/bin")
        try FileManager.default.createDirectory(at: engine, withIntermediateDirectories: true)
        let alife = try scratch.script("alife", "sleep 300")
        try FileManager.default.moveItem(at: alife, to: engine.appendingPathComponent("alife"))
        let runs = root.appendingPathComponent("runs")
        try FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        try world(runs, "tiny3s_seed1")  // a finished research planet with an autopilot
        try world(runs, "planet-2-branch-1", parent: "runs/tiny3s_seed1:30000", autopilot: false)
        try world(runs, "planet-3-grow", step: 12_300, status: "running")  // (the run this app grows in the growing pictures)
        try world(runs, "planet-4-grow")
        try LibraryTestFolders.played(runs, "planet-4", source: "runs/planet-4-grow:30000")
        try world(runs, "flat_1", autopilot: false, flat: true)
        try world(runs, "died_out", step: 700, status: "extinct")
        try LibraryTestFolders.write("{}", "events/chunk_0.json", in: runs.appendingPathComponent("planet-4"))
        try Self.config.write(to: runs.appendingPathComponent("planet-4/config.json"), atomically: true, encoding: .utf8)
        try (Self.metricsHeavy + "\n").write(to: runs.appendingPathComponent("planet-4/metrics.jsonl"), atomically: true, encoding: .utf8)
        // a world with a save point and no files besides: no numbers, no autopilot
        try LibraryTestFolders.research(runs, "bare_run")
        return root.path
    }

    struct Stage {
        var page: ShellPage
        var selected: String?
        var observe: String?
        var growing: String?
        var openWorld = false
        var dark: Bool
        var name: String
        var minimumColours = 25
        var size = CGSize(width: AppShell.minWidth, height: AppShell.minHeight)
    }

    /// The window at `stage`, drawn.
    private func drawShell(_ stage: Stage) async throws -> NSBitmapImageRep {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let path = try folder(scratch)
        let controller = RunController(storedPath: path, savePath: { _ in }, candidates: [], isApp: false)
        if let growing = stage.growing {
            // A real run job (a stand-in engine that sleeps): Keep growing on the world. It is ended before the picture is returned.
            controller.settings.mode = .resume
            controller.settings.sourceRun = growing
            controller.settings.steps = ""
            controller.start()
            #expect(controller.runningRunName == growing)
        }
        defer {
            controller.forceStop()
            for _ in 0..<400 where controller.job != nil { spin() }
        }
        // The pictures to look at need the dock open on the right; the test process's own preferences may say otherwise (other suites leave the
        // dock collapsed or on a side). Only a run that writes pictures sets them, and puts them back: a normal run touches nothing that a
        // suite running beside it reads.
        let dockKeys = ["world.dockCollapsed", "world.dockSide"]
        let kept = dockKeys.map { UserDefaults.standard.object(forKey: $0) }
        if everything {
            UserDefaults.standard.set(false, forKey: "world.dockCollapsed")
            UserDefaults.standard.set(DockSide.right.rawValue, forKey: "world.dockSide")
        }
        defer { if everything { for (key, value) in zip(dockKeys, kept) { UserDefaults.standard.set(value, forKey: key) } } }
        let suite = "alife-wd-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let play = PlayController()
        let session = WorldSession(play: play, defaults: defaults)
        let shell = ShellModel(session: session, defaults: defaults)
        let stream = StreamController()
        let observer = ObserverSession(controller: controller, stream: stream)
        observer.startStream = { _, _ in }  // (the engine is not started: the picture is the running app's)
        observer.runningWorld = { controller.runningRunName }
        let runs: () -> URL? = { controller.repoURL.appendingPathComponent("runs") }
        let watchReader = WorldReader(runsDirectory: runs, interval: .milliseconds(100))
        let detailsReader = WorldReader(runsDirectory: runs, interval: .milliseconds(100))
        let library = WorldLibrary(runsDirectory: runs, observesApp: false, lockHeld: { _ in false })
        library.refresh()
        _ = await eventually { library.hasScanned }
        if let growing = stage.growing { library.update(live: LiveFacts(growing: growing)) }
        shell.shapeOf = { library.state.row(named: $0)?.shape }
        if let selected = stage.selected { library.select(selected, scroll: false) }
        if let observe = stage.observe { shell.observe(observe) }
        shell.select(stage.page)
        let host = NSHostingView(rootView: AppShellView(shell: shell, controller: controller, library: library, observer: observer,
                                                        watchReader: watchReader, detailsReader: detailsReader, stream: stream, arguments: []))
        let window = QuietWindow(contentRect: NSRect(origin: .zero, size: stage.size), styleMask: [.borderless], backing: .buffered, defer: true)
        window.appearance = NSAppearance(named: stage.dark ? .darkAqua : .aqua)
        window.contentView = host
        host.frame = NSRect(origin: .zero, size: stage.size)
        window.alphaValue = 0
        window.ignoresMouseEvents = true
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        window.orderFrontRegardless()
        defer { window.orderOut(nil) }
        let rep = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        // the readers poll every 0.1 s: give the page time to have read the files
        let readers = [watchReader, detailsReader]
        _ = await eventually(timeout: .seconds(5)) { readers.contains { $0.reading?.metrics != nil } || stage.page == .watch && stage.observe == "bare_run" }
        for turn in 0..<40 {
            host.layoutSubtreeIfNeeded()
            window.displayIfNeeded()
            spin()
            guard turn >= 6 else { continue }
            host.cacheDisplay(in: host.bounds, to: rep)
            if colours(rep) > stage.minimumColours { break }
        }
        try save(rep, stage.name, dark: stage.dark)
        return rep
    }

    private func save(_ rep: NSBitmapImageRep, _ name: String, dark: Bool) throws {
        guard let out else { return }
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        try rep.representation(using: .png, properties: [:])?.write(to: out.appendingPathComponent("\(name)-\(dark ? "dark" : "light").png"))
    }

    private func drawView(_ view: some View, size: CGSize, dark: Bool, name: String) throws -> NSBitmapImageRep {
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

    // MARK: Details

    @Test(arguments: [false, true])
    func detailsOfAFinishedResearchWorldHasItsNumbersItsJournalAndItsSettings(dark: Bool) async throws {
        guard everything || !dark else { return }
        let rep = try await drawShell(Stage(page: .details, selected: "tiny3s_seed1", dark: dark, name: "f-details-research-saved"))
        #expect(colours(rep) > 20)
    }

    @Test(arguments: [false, true])
    func detailsOfABranchAndOfAWorldWithNoNumbers(dark: Bool) async throws {
        guard everything else { return }  // (a normal run draws four pictures: these tests hold the main actor, and other suites' timed waits run on it)
        #expect(colours(try await drawShell(Stage(page: .details, selected: "planet-2-branch-1", dark: dark, name: "f-details-branch"))) > 20)
        #expect(colours(try await drawShell(Stage(page: .details, selected: "bare_run", dark: dark, name: "f-details-bare", minimumColours: 20))) > 20)
        #expect(colours(try await drawShell(Stage(page: .details, selected: "died_out", dark: dark, name: "f-details-diedout"))) > 25)
        #expect(colours(try await drawShell(Stage(page: .details, selected: "flat_1", dark: dark, name: "f-details-flat"))) > 25)
    }

    @Test(arguments: [false, true])
    func detailsOfAPlayedWorld(dark: Bool) async throws {
        guard everything else { return }
        #expect(colours(try await drawShell(Stage(page: .details, selected: "planet-4", dark: dark, name: "f-details-played"))) > 20)
    }

    @Test(arguments: [false, true])
    func detailsOfAGrowingWorldHasItsClock(dark: Bool) async throws {
        guard everything else { return }
        #expect(colours(try await drawShell(Stage(page: .details, selected: "planet-3-grow", growing: "planet-3-grow", dark: dark, name: "f-details-growing"))) > 20)
    }

    @Test(arguments: [false, true])
    func detailsWithNothingChosenOrAWorldThatIsGone(dark: Bool) async throws {
        guard everything || !dark else { return }
        #expect(colours(try await drawShell(Stage(page: .details, dark: dark, name: "f-details-nothing", minimumColours: 6))) > 6)
        guard everything else { return }
        #expect(colours(try await drawShell(Stage(page: .details, selected: "gone_world", dark: dark, name: "f-details-gone", minimumColours: 6))) > 6)
    }

    // MARK: Watch, the research form

    @Test(arguments: [false, true])
    func theResearchWatchWithItsRunPanelForAFinishedWorld(dark: Bool) async throws {
        guard everything || !dark else { return }
        let rep = try await drawShell(Stage(page: .watch, observe: "tiny3s_seed1", dark: dark, name: "f-watch-research-saved"))
        #expect(colours(rep) > 25)
    }

    @Test(arguments: [false, true])
    func theResearchWatchWaitsForTheFirstSavePointOfAWorldThatGrows(dark: Bool) async throws {
        guard everything else { return }
        let scratchless = try await drawShell(Stage(page: .watch, observe: "planet-3-grow", growing: "planet-3-grow", dark: dark, name: "f-watch-research-growing"))
        #expect(colours(scratchless) > 25)
    }

    // MARK: Help and Settings

    @Test(arguments: [false, true])
    func helpDrawsEveryTopicAndBothKeyTables(dark: Bool) throws {
        guard everything || !dark else { return }
        let suite = "alife-help-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let shell = ShellModel(session: WorldSession(play: PlayController(), defaults: defaults), defaults: defaults)
        let help = HelpModel()
        for topic in HelpTopics.all where everything || topic.id == HelpTopics.controlsID || topic.id == "watch" {
            help.show(topic.id)
            let rep = try drawView(HelpView(model: help, shell: shell), size: CGSize(width: 860, height: 700), dark: dark, name: "f-help-\(topic.id)")
            #expect(colours(rep) > 15, "\(topic.id)")
        }
        guard everything else { return }
        shell.observe("tiny3s_seed1")
        help.show(HelpTopics.controlsID)
        #expect(colours(try drawView(HelpView(model: help, shell: shell), size: CGSize(width: 860, height: 700), dark: dark, name: "f-help-controls-research")) > 15)
    }

    @Test(arguments: [false, true])
    func settingsAutopilotDrawsInEveryState(dark: Bool) async throws {
        guard everything || !dark else { return }
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let path = try folder(scratch)
        let controller = RunController(storedPath: path, savePath: { _ in }, candidates: [], isApp: false)
        let suite = "alife-ap-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let settings = AutopilotSettings(controller: controller, defaults: defaults)
        settings.runCheck = { _, _, _ in LLMCheckResult.parse(status: 0, reason: .exit, output: "OK the model answered in 1.2 s") }
        let size = CGSize(width: 540, height: 360)
        #expect(colours(try drawView(AutopilotSettingsView(settings: settings), size: size, dark: dark, name: "f-settings-autopilot")) > 10)
        guard everything else { return }
        settings.test()
        _ = await eventually { settings.check != .running }
        #expect(colours(try drawView(AutopilotSettingsView(settings: settings), size: size, dark: dark, name: "f-settings-autopilot-ok")) > 10)
        settings.runCheck = { _, _, _ in LLMCheckResult.parse(status: 1, reason: .exit, output: "FAIL connection refused (127.0.0.1:9)") }
        settings.url = "http://127.0.0.1:9/v1/chat/completions"
        settings.test()
        _ = await eventually { settings.check != .running }
        #expect(colours(try drawView(AutopilotSettingsView(settings: settings), size: size, dark: dark, name: "f-settings-autopilot-failed")) > 10)
        settings.url = "nonsense"
        #expect(colours(try drawView(AutopilotSettingsView(settings: settings), size: size, dark: dark, name: "f-settings-autopilot-invalid")) > 10)
    }

    // MARK: The Worlds page, with the cards' new buttons

    @Test(arguments: [false, true])
    func theWorldsPageCardsHaveWatchAndDetails(dark: Bool) async throws {
        guard everything else { return }
        #expect(colours(try await drawShell(Stage(page: .worlds, dark: dark, name: "f-worlds-cards"))) > 30)
    }

    @Test(arguments: [false, true])
    func theSetUpAndActivityPagesStillDraw(dark: Bool) async throws {
        guard everything else { return }
        #expect(colours(try await drawShell(Stage(page: .activity, growing: "planet-3-grow", dark: dark, name: "f-activity", minimumColours: 12))) > 12)
    }
}

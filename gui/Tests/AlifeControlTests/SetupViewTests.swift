import AlifeCore
import AppKit
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// The set-up page and the pipeline's bar, drawn by AppKit in a window that is never shown, at the minimum size (1,100 x 700), in light and
/// dark, over a stand-in alife folder, in every state of the form and of the pipeline. With `ALIFE_SHELL_SHOTS_OUT=DIR` the pictures are also
/// written to DIR as PNG, for looking at.
@MainActor
@Suite("The set-up page, drawn", .serialized)
struct SetupViewTests {
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

    /// Draws the shell on `page` over the rig's models.
    private func draw(_ rig: PipelineRig, page: ShellPage, dark: Bool, name: String, advanced: Bool = false, command: Bool = false,
                      inWindow: Bool = false) async throws -> NSBitmapImageRep {
        let suite = "alife-setupview-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        UserDefaults.standard.set(advanced, forKey: "setup.advanced")
        UserDefaults.standard.set(command, forKey: "setup.showCommand")
        defer { UserDefaults.standard.removeObject(forKey: "setup.advanced"); UserDefaults.standard.removeObject(forKey: "setup.showCommand") }
        let session = WorldSession(play: rig.play, defaults: defaults)
        let shell = ShellModel(session: session, defaults: defaults)
        shell.select(page)
        let size = CGSize(width: AppShell.minWidth, height: AppShell.minHeight)
        let root: AnyView
        if inWindow {
            root = AnyView(SetupWindowView(setup: rig.setup, controller: rig.controller, library: rig.library).frame(width: size.width, height: size.height))
        } else {
            root = AnyView(AppShellView(shell: shell, controller: rig.controller, library: rig.library, setup: rig.setup, arguments: []))
        }
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
        for turn in 0..<40 {
            host.layoutSubtreeIfNeeded()
            window.displayIfNeeded()
            spin()
            guard turn >= 6 else { continue }
            host.cacheDisplay(in: host.bounds, to: rep)
            if colours(rep) > 30 { break }
        }
        if let out {
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            try rep.representation(using: .png, properties: [:])?.write(to: out.appendingPathComponent("\(name)-\(dark ? "dark" : "light").png"))
        }
        return rep
    }

    private func check(_ rep: NSBitmapImageRep, _ name: String) { #expect(colours(rep) > 25, "\(name): the page is blank") }

    // MARK: The form

    @Test(arguments: [false, true])
    func theFormDrawsInEveryStateOfTheDefaultPath(dark: Bool) async throws {
        guard everything || !dark else { return }
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try LibraryTestFolders.research(rig.runs, "planet-0")
        await rig.scan()
        let m = rig.setup
        m.openCreate()
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-create"), "create")
        guard everything else { return }
        m.name = "planet-0"
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-create-name-taken"), "taken")
        m.name = "planet-1"
        m.choose(.recipe(Recipes.flatWorld.id))
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-create-flat"), "flat")
        m.choose(.recipe(Recipes.biggerPlanet.id))
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-create-bigger"), "bigger")
        m.choose(.recipe(Recipes.smallPlanet.id))
        m.setGrowSteps("")
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-create-no-steps"), "no steps")
    }

    @Test(arguments: [false, true])
    func advancedAndTheChangesDraw(dark: Bool) async throws {
        guard everything else { return }
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        await rig.scan()
        let m = rig.setup
        m.openCreate()
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-create-advanced", advanced: true), "advanced")
        rig.controller.settings.levers.world3D.gravity = "0.09"
        rig.controller.settings.levers.seasonPeriod = "5000"
        rig.controller.settings.levers.seasonAmplitude = "2"
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-create-changes", advanced: true, command: true), "changes")
    }

    @Test(arguments: [false, true])
    func theRememberedChipsAndTheOtherIntentsDraw(dark: Bool) async throws {
        guard everything else { return }
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try LibraryTestFolders.research(rig.runs, "src", steps: [10_000, 20_000], status: "interrupted")
        await rig.scan()
        let m = rig.setup
        m.openCreate()
        rig.controller.settings.levers.world3D.gravity = "0.09"
        m.name = "mine"
        m.goal = .research
        m.start()
        _ = await rig.waitForState { $0 == .finished(stopped: false) }
        rig.coordinator.press(.done)
        await rig.scan()
        m.openCreate()
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-remembered", advanced: true), "remembered")
        m.openBranch(from: "src")
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-branch"), "branch")
        m.openKeepGrowing(world: "src")
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-keep-growing"), "keep growing")
        m.openMakePlanet(from: "src")
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-make-planet"), "make planet")
        m.openBranch(from: "src")
        m.name = "my-branch"
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-branch-in-window", inWindow: true), "branch in the window of its own")
        m.openCreate()
        check(try await draw(rig, page: .setup, dark: dark, name: "setup-in-window", inWindow: true), "window")
    }

    // MARK: The pipeline

    @Test(arguments: [false, true])
    func eachStateOfThePipelineDraws(dark: Bool) async throws {
        guard everything else { return }
        // growing, with numbers
        let growing = try PipelineRig(mode: "long:slow")
        defer { growing.finish() }
        #expect(await growing.create("planet-1").isEmpty)
        #expect(await growing.waitForGrowing())
        growing.tickMetrics()
        _ = await patiently { growing.coordinator.plan?.step ?? 0 > 0 }
        check(try await draw(growing, page: .setup, dark: dark, name: "pipeline-growing"), "growing")
        check(try await draw(growing, page: .worlds, dark: dark, name: "pipeline-growing-worlds"), "growing, worlds page")
        check(try await draw(growing, page: .activity, dark: dark, name: "pipeline-growing-activity"), "growing, activity page")
        // stopped: parked
        growing.coordinator.press(.stop)
        _ = await growing.waitForState { if case .parked = $0 { return true } else { return false } }
        await growing.scan()
        check(try await draw(growing, page: .setup, dark: dark, name: "pipeline-parked-stopped"), "parked")
        check(try await draw(growing, page: .worlds, dark: dark, name: "pipeline-parked-worlds"), "parked, worlds")
        // making the planet
        growing.mode = "ok:slow"
        growing.coordinator.press(.makePlanetFromHere)
        _ = await growing.waitForState { $0 == .making(cancelling: false) }
        _ = await patiently { growing.coordinator.plan?.making.phase != .starting }
        check(try await draw(growing, page: .setup, dark: dark, name: "pipeline-making"), "making")
        growing.coordinator.press(.cancel)
        _ = await growing.waitForState { $0 == .cancelled }
        check(try await draw(growing, page: .setup, dark: dark, name: "pipeline-cancelled"), "cancelled")
        // a refused run
        let failed = try PipelineRig(mode: "refuse:ok")
        defer { failed.finish() }
        #expect(await failed.create("planet-1").isEmpty)
        _ = await failed.waitForState { if case .failed = $0 { return true } else { return false } }
        check(try await draw(failed, page: .setup, dark: dark, name: "pipeline-failed"), "failed")
        // a planet that could not be opened
        let noOpen = try PipelineRig(mode: "ok:fail")
        defer { noOpen.finish() }
        #expect(await noOpen.create("planet-1").isEmpty)
        _ = await noOpen.waitForState { if case .failed = $0 { return true } else { return false } }
        check(try await draw(noOpen, page: .setup, dark: dark, name: "pipeline-failed-planet"), "failed planet")
        // a planet that is live
        let live = try PipelineRig(mode: "ok:ok")
        defer { live.finish() }
        #expect(await live.create("planet-1").isEmpty)
        _ = await live.waitForState { $0 == .live }
        check(try await draw(live, page: .setup, dark: dark, name: "pipeline-live"), "live")
        // a world that died out
        let died = try PipelineRig(mode: "extinct:ok")
        defer { died.finish() }
        #expect(await died.create("planet-1").isEmpty)
        _ = await died.waitForState { if case .diedOut = $0 { return true } else { return false } }
        await died.scan()
        check(try await draw(died, page: .setup, dark: dark, name: "pipeline-died-out"), "died out")
        check(try await draw(died, page: .worlds, dark: dark, name: "pipeline-died-out-worlds"), "died out, worlds page")
        // waiting while nobody looked
        let away = try PipelineRig(mode: "ok:ok")
        defer { away.finish() }
        away.looking = false
        #expect(await away.create("planet-1").isEmpty)
        _ = await away.waitForState { if case .parked = $0 { return true } else { return false } }
        check(try await draw(away, page: .worlds, dark: dark, name: "pipeline-parked-away-worlds"), "away")
    }

    @Test func theFormAndTheBarDrawInTheFirstPictureEvenWithNoFolder() async throws {
        let rig = try PipelineRig(mode: "ok:ok")
        defer { rig.finish() }
        try FileManager.default.removeItem(at: rig.repo.appendingPathComponent(".venv/bin/alife"))
        rig.controller.refreshRepo()
        await rig.scan()
        rig.setup.openCreate()
        check(try await draw(rig, page: .setup, dark: false, name: "setup-no-engine"), "no engine")
        #expect(rig.setup.reasons.contains("Set up the engine first."))
    }
}

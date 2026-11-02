@testable import AlifeCore
import AppKit
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// The pages of the `alife` window, drawn by AppKit in a window that is never shown, at the minimum size (1,100 x 700), in light and
/// dark, over scratch alife folders in each state a newcomer meets. With `ALIFE_SHELL_SHOTS_OUT=DIR` the pictures are also written to DIR as
/// PNG, for looking at; without it nothing is written. The Metal canvas of the Watch page is not drawn here (no world is open): the
/// running app's pictures show it.
@MainActor
@Suite("The pages of the alife window", .serialized)
struct ShellViewTests {
    private var out: URL? { ProcessInfo.processInfo.environment["ALIFE_SHELL_SHOTS_OUT"].map { URL(fileURLWithPath: $0) } }

    private static let playJSON = """
        {"play_version": 1, "base_rate": 14.4, "cadence": {"metrics_every": 250, "heavy_metrics_every": 250,
         "checkpoint_every": 6000, "keep_checkpoints": 3}, "tick": 25, "weather": {"dc_ref": 0.5}, "planet": null}
        """

    private func write(_ text: String, _ name: String, in dir: URL) throws {
        try FileManager.default.createDirectory(at: dir.appendingPathComponent(name).deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8)
    }

    enum Folder: String, CaseIterable {
        case missing, noEngine, noRuns, oneRun, worlds, five, sixty
    }

    /// A scratch alife folder in the given state.
    private func folder(_ state: Folder, _ scratch: Scratch) throws -> String {
        let root = scratch.url.appendingPathComponent("the-alife-folder")
        if state == .missing { return root.path + "-missing" }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if state == .noEngine { return root.path }
        let engine = root.appendingPathComponent(".venv/bin")
        try FileManager.default.createDirectory(at: engine, withIntermediateDirectories: true)
        let alife = try scratch.script("alife", "exit 0")
        try FileManager.default.moveItem(at: alife, to: engine.appendingPathComponent("alife"))
        let runs = root.appendingPathComponent("runs")
        try FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        if state == .oneRun || state == .worlds {
            let research = runs.appendingPathComponent("tiny3s_seed1")
            try write(#"{"world": {"dims": 3, "geometry": "sphere", "radius": 36, "size": 64}, "weather": {"enabled": false}}"#, "config.json", in: research)
            try write("x", "checkpoints/step_0000030000.safetensors", in: research)
            try write("{}", "events/chunk_0.json", in: research)
        }
        if state == .five { try LibraryTestFolders.many(runs, count: 5) }
        if state == .sixty {
            try LibraryTestFolders.many(runs, count: 60)
            // an archive on a drive that is not connected
            try FileManager.default.createSymbolicLink(
                atPath: runs.appendingPathComponent("archived_old").path, withDestinationPath: "/Volumes/Archive/runs/archived_old")
        }
        if state == .worlds {
            for name in ["mini3s_a", "planet_b"] {
                let world = runs.appendingPathComponent(name)
                try write(Self.playJSON, "play.json", in: world)
                try write(#"{"world": {"dims": 3, "geometry": "sphere", "radius": 36}, "weather": {"enabled": true}}"#, "config.json", in: world)
                try write("x", "checkpoints/step_0000450000.safetensors", in: world)
            }
        }
        return root.path
    }

    private func draw(_ page: ShellPage, state: Folder, dark: Bool, name: String, minimumColours: Int = 25,
                      failure: StartFailure? = nil, configure: (WorldLibrary) -> Void = { _ in }) async throws -> NSBitmapImageRep {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let path = try folder(state, scratch)
        let controller = RunController(storedPath: path, savePath: { _ in }, candidates: [], isApp: false)
        let suite = "alife-shellview-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let play = PlayController()
        let session = WorldSession(play: play, defaults: defaults)
        let shell = ShellModel(session: session, defaults: defaults)
        shell.select(page)
        shell.startFailure = failure
        let size = CGSize(width: AppShell.minWidth, height: AppShell.minHeight)
        // (world_002 is held by another app, world_003 is the one this window has open, world_004 is growing)
        let library = WorldLibrary(
            runsDirectory: { controller.repoReady ? controller.repoURL.appendingPathComponent("runs") : nil }, observesApp: false,
            lockHeld: { $0.lastPathComponent == "world_002" })
        library.refresh()
        _ = await eventually { library.hasScanned }
        configure(library)
        let host = NSHostingView(rootView: AppShellView(shell: shell, controller: controller, library: library, arguments: []))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: true)
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.contentView = host
        host.frame = NSRect(origin: .zero, size: size)
        window.alphaValue = 0
        window.ignoresMouseEvents = true
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        window.orderFrontRegardless()
        defer { window.orderOut(nil) }
        // Short turns, and only as many as the picture needs: these tests hold the main actor, and other suites' waits run on it.
        let rep = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        for turn in 0..<40 {
            host.layoutSubtreeIfNeeded()
            window.displayIfNeeded()
            spin()
            guard turn >= 5 else { continue }
            host.cacheDisplay(in: host.bounds, to: rep)
            if colours(rep) > minimumColours { break }
        }
        if let out {
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            try rep.representation(using: .png, properties: [:])?.write(to: out.appendingPathComponent("\(name)-\(dark ? "dark" : "light").png"))
        }
        return rep
    }

    private func spin() { RunLoop.current.run(until: Date().addingTimeInterval(0.01)) }

    /// The number of distinct colours among a sparse grid of pixels: a blank picture has one or two.
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

    /// A normal run draws five pictures (these tests hold the main actor, and other suites' waits run on it); with the shots directory
    /// set, every state in light and dark.
    private var everything: Bool { out != nil }

    @Test(arguments: [false, true])
    func theWorldsPageDrawsInEveryStateAFolderCanBeIn(dark: Bool) async throws {
        guard everything || !dark else { return }
        for state in Folder.allCases where everything || state == .missing || state == .worlds || state == .sixty {
            let rep = try await draw(.worlds, state: state, dark: dark, name: "worlds-\(state.rawValue)")
            #expect(colours(rep) > 25, "worlds / \(state.rawValue): the page is blank")
        }
    }

    @Test(arguments: [false, true])
    func aWorldThatDidNotOpenIsToldOnTheWorldsPage(dark: Bool) async throws {
        guard everything || !dark else { return }
        let failure = StartFailure(world: "mini3s_a", message: "alife play found no usable checkpoint (exit code 4)")
        let rep = try await draw(.worlds, state: .worlds, dark: dark, name: "worlds-failed", failure: failure)
        #expect(colours(rep) > 25)
    }

    @Test(arguments: [false, true])
    func theOtherPagesDraw(dark: Bool) async throws {
        guard everything || !dark else { return }
        for (page, state) in [(ShellPage.watch, Folder.worlds), (.activity, .worlds), (.lab, .oneRun)] {
            let rep = try await draw(page, state: state, dark: dark, name: page.rawValue, minimumColours: page == .watch ? 2 : 12)
            // (Watch with no world is a card on a plain page: few colours, but not one)
            let n = colours(rep)
            #expect(n > (page == .watch ? 2 : 12), "\(page.rawValue) dark=\(dark): \(n) colours")
        }
    }

    @Test(arguments: [false, true])
    func theFiltersAndTheSearchDraw(dark: Bool) async throws {
        guard everything || !dark else { return }
        let unfinished = try await draw(.worlds, state: .sixty, dark: dark, name: "worlds-sixty-unfinished") { $0.filter = .unfinished }
        #expect(colours(unfinished) > 25)
        guard everything else { return }
        let found = try await draw(.worlds, state: .sixty, dark: dark, name: "worlds-sixty-search") {
            $0.query = "research box"
            $0.select("world_001")
        }
        #expect(colours(found) > 25)
        let none = try await draw(.worlds, state: .sixty, dark: dark, name: "worlds-sixty-nomatch") { $0.query = "zzz" }
        #expect(colours(none) > 25)
    }
}

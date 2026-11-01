import AlifeCore
import AppKit
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// The two-second listing of `runs/` on the main thread belongs to the Lab's pickers: the `alife` window turns it off away from the Lab and
/// lists once on the way in (docs/APP.md §5.3); the two old apps never turn it off.
@MainActor
@Suite("The run list's polling", .serialized) struct RunListPollingTests {
    private func folder(_ scratch: Scratch) throws -> URL {
        let root = scratch.url.appendingPathComponent("folder")
        let bin = root.appendingPathComponent(".venv/bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        let alife = bin.appendingPathComponent("alife")
        try "#!/bin/bash\nexit 0\n".write(to: alife, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: alife.path)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("runs/first"), withIntermediateDirectories: true)
        return root
    }

    private func newRun(_ root: URL, _ name: String) throws {
        try FileManager.default.createDirectory(at: root.appendingPathComponent("runs/\(name)"), withIntermediateDirectories: true)
    }

    @Test func theTickerListsRunsEveryTwoSecondsByDefault() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let root = try folder(scratch)
        let controller = RunController(storedPath: root.path, savePath: { _ in }, candidates: [], isApp: false)
        #expect(controller.pollsRunList && controller.runs == ["first"])
        try newRun(root, "second")
        controller.tick()
        controller.tick()
        #expect(Set(controller.runs) == ["first", "second"], "the old apps' Lab sees a new run within two seconds")
    }

    @Test func withPollingOffTheTickerDoesNotListAndTheLabListsWhenItAsks() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let root = try folder(scratch)
        let controller = RunController(storedPath: root.path, savePath: { _ in }, candidates: [], isApp: false)
        controller.pollsRunList = false
        try newRun(root, "second")
        controller.tick()
        controller.tick()
        controller.tick()
        controller.tick()
        #expect(controller.runs == ["first"], "the main thread is not listing a folder nobody on screen reads")
        controller.refreshRuns()
        #expect(Set(controller.runs) == ["first", "second"])
    }

    @Test func theFirstPlanetsRunListIsFreshWhateverPageTheUserCameFrom() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let root = try folder(scratch)
        let controller = RunController(storedPath: root.path, savePath: { _ in }, candidates: [], isApp: false)
        controller.pollsRunList = false
        try newRun(root, "made_a_moment_ago")
        controller.prepareSmallPlanetRun()
        #expect(controller.runs.contains("made_a_moment_ago"), "the name that Start will use is chosen against the folder as it is")
    }

    @Test func theWindowPollsOnlyWhileTheSetUpPageIsShownAndListsOnTheWayIn() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let root = try folder(scratch)
        let controller = RunController(storedPath: root.path, savePath: { _ in }, candidates: [], isApp: false)
        let suite = "alife-poll-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let shell = ShellModel(session: WorldSession(play: PlayController(), defaults: defaults), defaults: defaults)
        let library = WorldLibrary(runsDirectory: { root.appendingPathComponent("runs") }, observesApp: false)
        let size = CGSize(width: AppShell.minWidth, height: AppShell.minHeight)
        let host = NSHostingView(rootView: AppShellView(shell: shell, controller: controller, library: library, arguments: []))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: true)
        window.contentView = host
        host.frame = NSRect(origin: .zero, size: size)
        window.alphaValue = 0
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        window.orderFrontRegardless()
        defer { window.orderOut(nil) }
        #expect(await eventually(timeout: .seconds(10)) { !controller.pollsRunList }, "the window opens on Worlds: the ticker leaves the list alone")
        try newRun(root, "made_while_away")
        shell.select(.setup)
        #expect(await eventually(timeout: .seconds(10)) { controller.pollsRunList }, "on the set-up page it polls")
        #expect(controller.runs.contains("made_while_away"), "and the list is read once on the way in")
        shell.select(.worlds)
        #expect(await eventually(timeout: .seconds(10)) { !controller.pollsRunList })
    }
}

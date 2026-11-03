@testable import AlifeCore
import AppKit
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// Welcome drawn by AppKit in a window that is never shown, at the window's minimum size (1,100 x 700), in light and dark, in every state a
/// person without a working engine can be in. With `ALIFE_WELCOME_SHOTS_OUT=DIR` the pictures are also written to DIR as PNG, for looking at.
@MainActor
@Suite("Welcome, drawn", .serialized)
struct WelcomeViewTests {
    private var out: URL? { ProcessInfo.processInfo.environment["ALIFE_WELCOME_SHOTS_OUT"].map { URL(fileURLWithPath: $0) } }

    enum Situation: String, CaseIterable {
        case noFolder, missingFolder, notAlife, setUp, noUV, settingUp, setupFailed, checking, checkFailed
    }

    /// The whole window on the Worlds page, over a scratch folder in `situation`.
    private func draw(_ situation: Situation, dark: Bool) async throws -> NSBitmapImageRep {
        let uvBody: String? = switch situation {
        case .noUV: nil
        case .settingUp: "echo 'Resolved 41 packages in 12ms'; echo 'Downloading mlx (38.1MiB)'; echo 'Downloading numpy (5.2MiB)'; trap 'exit 130' INT; while true; do sleep 0.1; done"
        case .setupFailed: "echo 'Resolved 41 packages in 12ms'; echo 'error: Failed to download `mlx==0.30.1`' >&2; echo '  Caused by: error sending request for url (https://files.pythonhosted.org/mlx)' >&2; exit 2"
        default: "exit 0"
        }
        let rig = try WelcomeTests.Rig(engine: situation == .checking || situation == .checkFailed, checkout: situation != .noFolder && situation != .notAlife,
                                       uvBody: uvBody, folderExists: situation != .noFolder && situation != .missingFolder)
        defer { rig.cleanup() }
        let controller = situation == .noFolder
            ? RunController(storedPath: "", savePath: { _ in }, candidates: [], isApp: false) : rig.controller
        if situation == .noFolder { controller.uvLocator = { nil } }
        let model = rig.model
        var release: CheckedContinuation<Void, Never>?
        defer { release?.resume() }
        switch situation {
        case .settingUp:
            controller.setupEnvironment()
            _ = await eventually { WelcomeModel.progressLines(controller).contains { $0.contains("numpy") } }
        case .setupFailed:
            controller.setupEnvironment()
            _ = await eventually { controller.job == nil && controller.setupTrouble != nil }
        case .checking:
            // (Welcome was shown before the engine appeared: the check is then told)
            try FileManager.default.moveItem(at: rig.folder.appendingPathComponent(".venv"), to: rig.scratch.url.appendingPathComponent("venv-aside"))
            controller.refreshRepo()
            model.refresh(controller)
            try FileManager.default.moveItem(at: rig.scratch.url.appendingPathComponent("venv-aside"), to: rig.folder.appendingPathComponent(".venv"))
            controller.refreshRepo()
            model.check = { _, _ in
                await withCheckedContinuation { release = $0 }
                return .passed
            }
        case .checkFailed:
            model.check = { _, _ in
                .failed(WelcomeGuide.healthFailure(
                    status: 1, timedOut: false,
                    lines: ["Traceback (most recent call last):", "  File \".venv/bin/alife\", line 3, in <module>", "ModuleNotFoundError: No module named 'mlx'"]))
            }
            model.refresh(controller)
            _ = await eventually { if case .failed = model.health { true } else { false } }
        default: break
        }
        let suite = "alife-welcomeview-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let session = WorldSession(play: PlayController(), defaults: defaults)
        let shell = ShellModel(session: session, defaults: defaults, welcome: model)
        shell.select(.worlds)
        let size = CGSize(width: AppShell.minWidth, height: AppShell.minHeight)
        let library = WorldLibrary(runsDirectory: { controller.repoReady ? controller.repoURL.appendingPathComponent("runs") : nil }, observesApp: false)
        let host = NSHostingView(rootView: AppShellView(shell: shell, controller: controller, library: library, arguments: []))
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
            guard turn >= 8 else { continue }
            host.cacheDisplay(in: host.bounds, to: rep)
        }
        host.cacheDisplay(in: host.bounds, to: rep)
        if situation == .settingUp {
            controller.cancelSetup()
            _ = await eventually { controller.job == nil }
        }
        if let out {
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            try rep.representation(using: .png, properties: [:])?.write(to: out.appendingPathComponent("welcome-\(situation.rawValue)-\(dark ? "dark" : "light").png"))
        }
        return rep
    }

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

    @Test(arguments: [false, true])
    func everyStateOfWelcomeDraws(dark: Bool) async throws {
        // a normal run draws two states; with the shots directory set, all of them in the mode asked
        for situation in Situation.allCases where out != nil || situation == .setUp || situation == .setupFailed {
            let rep = try await draw(situation, dark: dark)
            #expect(colours(rep) > 25, "\(situation.rawValue) dark=\(dark): the page is blank")
        }
    }
}

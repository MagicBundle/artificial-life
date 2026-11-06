import AlifeTestSupport
import Foundation

@testable import AlifeControl

/// Where the repository is, found from this file's own path: `gui/Tests/AlifeControlTests/Support.swift`.
enum Repo {
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()

    static let streamData = root.appendingPathComponent("tests/data", isDirectory: true)

    static func streamFixture(_ kind: String) throws -> [UInt8] {
        [UInt8](try Data(contentsOf: streamData.appendingPathComponent("stream_tiny_\(kind).bin")))
    }

    static func streamFixtureURL(_ kind: String) -> URL {
        streamData.appendingPathComponent("stream_tiny_\(kind).bin")
    }

    /// A committed play stream (`play_tiny3d`, `play_tiny2d`, `play_tiny3d_n70`).
    static func playFixtureURL(_ kind: String) -> URL {
        streamData.appendingPathComponent("play_tiny\(kind).bin")
    }
}

/// The test process is a background tool with windows that nobody sees; with the display awake and someone working in another app, macOS
/// throttles such a process (App Nap): a 20 ms run-loop turn was seen to last seconds, a hosted world took 150 s to come up, and the tests
/// that wait on timers failed together (two of fourteen full runs). An activity that is user-initiated switches the throttling off for the
/// whole process; it is taken by the first test helper that is used (a scratch folder, a window, a wait) and never ended.
enum TestProcess {
    nonisolated(unsafe) static let activity: NSObjectProtocol = ProcessInfo.processInfo.beginActivity(
        options: [.userInitiated, .idleSystemSleepDisabled], reason: "alife tests: a throttled test process stalls its main thread for minutes")
    static func unthrottled() { _ = activity }
}

/// A fresh empty directory under the system temporary directory, removed by `cleanup()`.
struct Scratch {
    let url: URL

    init() throws {
        TestProcess.unthrottled()
        url = FileManager.default.temporaryDirectory.appendingPathComponent("alifecontrol-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    func cleanup() { try? FileManager.default.removeItem(at: url) }

    /// An executable shell script `name` with `body` (a stand-in for `alife`).
    func script(_ name: String, _ body: String) throws -> URL {
        let file = url.appendingPathComponent(name)
        try ("#!/bin/bash\n" + body + "\n").write(to: file, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: file.path)
        return file
    }
}

/// Polls `condition` on the main actor until it holds or `timeout` (times `TestTime.scale`) passes. True if it held.
@MainActor
func eventually(timeout: Duration = .seconds(20), _ condition: @MainActor () -> Bool) async -> Bool {
    TestProcess.unthrottled()
    let deadline = ContinuousClock.now + TestTime.limit(timeout)
    while !condition() {
        if ContinuousClock.now > deadline {
            // A stall (the main actor held for seconds by a drawing suite) can use up the whole wait while the work the condition waits for sits
            // queued behind it: look once more after letting it run.
            await Task.yield()
            try? await Task.sleep(for: .milliseconds(100))
            return condition()
        }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return true
}

import AppKit

extension QuitPanel {
    /// The real quit panel, placed off every screen: a test run must never put anything on the display of whoever is working (it still
    /// draws, updates and answers, in a window nobody sees).
    @MainActor static func offscreen() -> QuitPanel {
        QuitPanel(place: { $0.alphaValue = 0; $0.setFrameOrigin(NSPoint(x: -20_000, y: -20_000)) }, keepsOnScreen: false)
    }
}

/// Every window a test makes is one of these (or a subclass): **no fade animation**. AppKit fades a window out when it is closed or ordered out,
/// as a blocking `NSAnimation` on a global-queue thread; while the display sleeps or the session is locked the animation never ends, each
/// window leaks one blocked worker, and a full run makes enough of them (the pool's soft limit is 90) to starve everything that waits on the
/// dispatch queues. `TestWindowLintTests` fails a test file that makes a window any other way.
class QuietWindow: NSWindow {
    override init(contentRect: NSRect, styleMask style: NSWindow.StyleMask, backing backingStoreType: NSWindow.BackingStoreType, defer flag: Bool) {
        TestProcess.unthrottled()
        super.init(contentRect: contentRect, styleMask: style, backing: backingStoreType, defer: flag)
        animationBehavior = .none
    }
}

/// A titled window that stays where it is put. AppKit moves a titled window back onto a screen when it is ordered in (a borderless one it
/// leaves alone), so a test that needs the title bar (a sheet, a key equivalent) uses this at -20000 and nothing of it is ever on a display.
final class OffscreenWindow: QuietWindow {
    override func constrainFrameRect(_ frameRect: NSRect, to screen: NSScreen?) -> NSRect { frameRect }
}

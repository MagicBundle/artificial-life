@testable import AlifeCore
import AppKit
import Foundation
import Testing

@testable import AlifeControl

/// Review fixes of PR-B: the cue off Watch, the dock badge, the key gate of the planet's view, the reopen of the window and what closing it
/// takes with it.
@MainActor
@Suite("The alife window: attention, the planet's keys, reopening") struct ShellAttentionTests {
    @MainActor private struct Rig {
        let scratch: Scratch
        let play = PlayController()
        let session: WorldSession
        let suite: String
        let script: URL
        let commands: URL
        var sent: [String] { ((try? String(contentsOf: commands, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init) }

        init() throws {
            scratch = try Scratch()
            suite = "alife-attention-\(UUID().uuidString)"
            commands = scratch.url.appendingPathComponent("commands.txt")
            script = try scratch.script("alife", "cat \"\(Repo.playFixtureURL("3d_g3b").path)\"\nwhile IFS= read -r line; do echo \"$line\" >> \"\(commands.path)\"; done")
            session = WorldSession(play: play, defaults: try #require(UserDefaults(suiteName: suite)))
        }

        func start() { play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url, arguments: []) }

        func finish() {
            play.shutdown()
            session.model.observeWindow(nil)
            UserDefaults(suiteName: suite)?.removePersistentDomain(forName: suite)
            scratch.cleanup()
        }

        func alerts() throws -> [AlertChange] {
            try [12_000_600, 12_000_601, 12_000_602, 11_500_603, 11_750_605].map {
                AlertChange(kind: .raised, alert: try #require(play.store.alerts.alert(id: UInt64($0))))
            }
        }
    }

    // MARK: M1: a world that stops while the player is on another page

    @Test func aStopWhileAnotherPageIsShownAsksForAttentionInTheKeyWindowAndTheSidebarSaysSo() async throws {
        let rig = try Rig()
        defer { rig.finish() }
        var asked = 0
        rig.session.model.windowIsKey = { true }  // the app is active and the window is the key one: the player is on the Lab, say
        rig.session.model.requestAttention = { asked += 1 }
        #expect(!rig.session.model.watchActive)
        rig.start()
        #expect(await eventually { rig.play.store.header != nil && rig.play.store.clock?.reason == .alert })
        // (the fixture ends on an alert stop at rate 0) The stop is the player's business, and they are not on Watch:
        #expect(asked == 1, "off Watch a stop is not 'looking', even when the window is key")
        let cue = rig.session.model.watchCue(startFailed: false)
        #expect(cue.needsPlayer && cue.text.contains("stopped for an alert"), "\(cue.text)")
        // on Watch, in the key window, the player is looking: the row says only that the world is live
        rig.session.model.setWatchActive(true)
        let looking = rig.session.model.watchCue(startFailed: false)
        #expect(!looking.needsPlayer && looking.text.hasSuffix("live"), "\(looking.text)")
    }

    @Test func unreadAlertsAreCountedInTheRowUntilWatchIsShown() async throws {
        let rig = try Rig()
        defer { rig.finish() }
        rig.session.model.windowIsKey = { true }
        rig.session.model.setBadge = { _ in }
        rig.start()
        #expect(await eventually { rig.play.store.header != nil })
        rig.session.model.handle(alerts: try rig.alerts(), backlog: false)
        #expect(rig.session.model.watchCue(startFailed: false).text.contains("2 unread alerts"))
        rig.session.model.setWatchActive(true)
        #expect(!rig.session.model.watchCue(startFailed: false).text.contains("unread"))
    }

    // MARK: M5: the Lab coming forward is not the player looking at the world

    @Test func theAppBecomingActiveClearsTheBadgeOnlyWithWatchShown() async throws {
        let rig = try Rig()
        defer { rig.finish() }
        var badge: String?
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 200, height: 100), styleMask: [.borderless], backing: .buffered, defer: true)
        rig.session.model.observeWindow(window)  // installs the observers (and the dock tile's setBadge, which the next lines replace)
        rig.session.model.windowIsKey = { true }
        rig.session.model.requestAttention = {}
        rig.session.model.setBadge = { badge = $0 }
        rig.start()
        #expect(await eventually { rig.play.store.header != nil })
        rig.session.model.handle(alerts: try rig.alerts(), backlog: false)
        #expect(badge == "2")
        NotificationCenter.default.post(name: NSApplication.didBecomeActiveNotification, object: NSApp)
        try await Task.sleep(for: .milliseconds(200))
        #expect(badge == "2", "the app came forward on another page: the alerts are still unread")
        rig.session.model.setWatchActive(true)
        #expect(badge == nil)
        rig.session.model.attention.raised(2, windowIsKey: false)
        badge = "2"
        NotificationCenter.default.post(name: NSApplication.didBecomeActiveNotification, object: NSApp)
        #expect(await eventually { badge == nil }, "with Watch shown it is the player looking")
    }

    // MARK: The planet's view has the same gate

    @Test func theSphereViewsKeysFallThroughOffWatchAndActOnIt() async throws {
        let rig = try Rig()
        defer { rig.finish() }
        rig.start()
        #expect(await eventually { rig.play.store.header != nil })
        let view = SphereMTKView(frame: CGRect(x: 0, y: 0, width: 900, height: 700), device: nil)
        view.model = rig.session.model
        rig.session.model.setWatchActive(false)
        func key(_ c: String) -> NSEvent {
            NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0, context: nil,
                             characters: c, charactersIgnoringModifiers: c, isARepeat: false, keyCode: 0)!
        }
        for c in [" ", ".", "g", "k", "1"] { #expect(!view.handleKey(key(c)), "'\(c)' off Watch") }
        #expect(!view.acceptsFirstResponder)
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.sent.isEmpty, "nothing reached the world: \(rig.sent)")
        rig.session.model.setWatchActive(true)
        #expect(view.handleKey(key(" ")))
        #expect(await eventually { !rig.sent.isEmpty })
    }

    // MARK: M4: the window comes back, and closing it closes the app's other windows

    private func delegate(_ plan: LaunchMode) -> AppDelegate {
        let delegate = AppDelegate(run: RunController(storedPath: "", savePath: { _ in }, candidates: [], isApp: false), play: PlayController())
        delegate.plan = plan.plan
        delegate.replyToTerminate = {}
        delegate.terminate = {}
        return delegate
    }

    @Test func aClickOnTheDockIconBringsTheClosedWindowBack() {
        let delegate = delegate(.alife)
        var opened = 0
        delegate.reopenMainWindow = { opened += 1 }
        #expect(delegate.applicationShouldHandleReopen(NSApp, hasVisibleWindows: false) == false)
        #expect(opened == 1)
        #expect(delegate.applicationShouldHandleReopen(NSApp, hasVisibleWindows: true), "with a window showing, AppKit does as it does")
        #expect(opened == 1)
    }

    @Test func theOldAppsKeepAppKitsOwnReopen() {
        for mode in [LaunchMode.lab, .world] {
            let delegate = delegate(mode)
            var opened = 0
            delegate.reopenMainWindow = { opened += 1 }
            #expect(delegate.applicationShouldHandleReopen(NSApp, hasVisibleWindows: false))
            #expect(opened == 0)
        }
    }

    private func window(_ id: String?) -> NSWindow {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 100, height: 100), styleMask: [.titled], backing: .buffered, defer: true)
        window.isReleasedWhenClosed = false
        window.identifier = id.map { NSUserInterfaceItemIdentifier($0) }
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        window.alphaValue = 0
        window.orderFrontRegardless()
        return window
    }

    @Test func closingTheMainWindowClosesSettingsAndTheViewerSoTheAppEndsThroughTheQuit() {
        let delegate = delegate(.alife)
        let main = window(AppShell.mainWindowID), settings = window("settings"), viewer = window("viewer3d")
        defer { for w in [main, settings, viewer] { w.close() } }
        main.close()  // (as the player's close did; the observer then calls this)
        delegate.mainWindowClosed(windows: { [main, settings, viewer] })
        #expect(!settings.isVisible && !viewer.isVisible)
        // a window that is the main one is left to the close that is under way
        let other = window(AppShell.mainWindowID)
        defer { other.close() }
        delegate.mainWindowClosed(windows: { [other] })
        #expect(other.isVisible)
    }

    @Test func theOldAppsDoNotCloseOtherWindowsWithAMainOne() {
        let delegate = delegate(.lab)
        let settings = window("settings")
        defer { settings.close() }
        delegate.mainWindowClosed(windows: { [settings] })
        #expect(settings.isVisible)
    }
}

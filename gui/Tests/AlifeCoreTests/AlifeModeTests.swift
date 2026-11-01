import Foundation
import Testing

@testable import AlifeCore

/// The third mode, `alife`: one window with a sidebar (docs/APP.md §2.4, PR-B). The two old modes are pinned by `LaunchModeTests`.
@Suite struct AlifeModeTests {
    @Test func theBundleValueAndTheArgumentSelectIt() {
        #expect(LaunchMode.resolve(info: "alife", arguments: []) == .alife)
        #expect(LaunchMode.resolve(info: " Alife ", arguments: []) == .alife)
        #expect(LaunchMode.resolve(info: nil, arguments: ["--launch-mode", "alife"]) == .alife)
        #expect(LaunchMode.resolve(info: "alife", arguments: ["--launch-mode=lab"]) == .lab)
        // the default is still the Lab: a bundle from before the key and `swift run` behave as they always did
        #expect(LaunchMode.resolve(info: nil, arguments: []) == .lab)
        #expect(LaunchMode.resolve(info: "preview", arguments: []) == .lab)
    }

    @Test func itIsOneWindowThatTheTwoOldWindowsDoNotShare() {
        let plan = LaunchMode.alife.plan
        #expect(plan.appName == "alife")
        #expect(plan.isShell)
        #expect(!LaunchMode.lab.plan.isShell && !LaunchMode.world.plan.isShell)
        #expect(plan.windowsAtLaunch().isEmpty)
        #expect(plan.windowsAtLaunch(worldArgument: "planet_a").isEmpty)  // the argument starts the world in the shell
        #expect(!plan.labIsMainWindow)
        #expect(plan.windowMenu.isEmpty)
        // the safety of PR-A holds in every mode
        #expect(plan.confirmsRunOnQuit && plan.quitWaitsForRun && plan.quitsAfterLastWindow)
    }

    @Test func theOldModesAreAsTheyWere() {
        #expect(LaunchMode.lab.plan.appName == "Alife Control")
        #expect(LaunchMode.world.plan.appName == "Storm Worlds")
        #expect(LaunchMode.lab.plan.windowsAtLaunch() == [.lab])
        #expect(LaunchMode.world.plan.windowsAtLaunch() == [.world])
        #expect(AppWindow.allCases.map(\.rawValue) == ["lab", "world", "viewer3d"])
    }

    @Test func theMainWindowIsNotAnOldWindow() {
        #expect(AppShell.mainWindowID == "main")
        #expect(!AppWindow.allCases.map(\.rawValue).contains(AppShell.mainWindowID))
        #expect(AppShell.minWidth == 1100 && AppShell.minHeight == 700)
    }
}

@Suite struct ShellPageTests {
    @Test func theCommandKeysFollowTheSidebarOrder() {
        #expect(ShellPage.allCases.map(\.key) == ["1", "2", "3", "4"])
        #expect(ShellPage.allCases == [.worlds, .watch, .activity, .setup])
        #expect(Set(ShellPage.allCases.map(\.key)).count == ShellPage.allCases.count)
        #expect(Set(ShellPage.allCases.map(\.title)).count == ShellPage.allCases.count)
    }

    @Test func theNamesAreTheSpecsOnes() {
        #expect(ShellPage.allCases.map(\.title) == ["Worlds", "Watch", "Activity", "New world"])
        // Resume, Home and Continue are not page names; "Watch" is the picture
        for page in ShellPage.allCases { #expect(!["Resume", "Home", "World", "3D Viewer"].contains(page.title)) }
    }

    @Test func aRememberedPageIsRestoredExceptWatch() {
        #expect(ShellPage.restored(from: "setup") == .setup)
        #expect(ShellPage.restored(from: "activity") == .activity)
        #expect(ShellPage.restored(from: "watch") == .worlds)  // opening the app starts nothing, so no world is open to watch
        #expect(ShellPage.restored(from: nil) == .worlds)
        #expect(ShellPage.restored(from: "nonsense") == .worlds)
    }
}

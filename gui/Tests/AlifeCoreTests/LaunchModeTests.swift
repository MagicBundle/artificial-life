import Foundation
import Testing

@testable import AlifeCore

/// Which app the bundle is (Storm Worlds or the Lab) and what each does with its windows (DESIGN §33.4.10).
@Suite struct LaunchModeTests {
    // MARK: Parsing

    @Test func noKeyAnEmptyOrUnknownValueAndAnythingButAStringAreTheLab() {
        #expect(LaunchMode.resolve(info: nil, arguments: []) == .lab)
        #expect(LaunchMode.resolve(info: "", arguments: []) == .lab)
        #expect(LaunchMode.resolve(info: "storm", arguments: []) == .lab)
        #expect(LaunchMode.resolve(info: 1, arguments: []) == .lab)
        #expect(LaunchMode.resolve(info: true, arguments: []) == .lab)
        #expect(LaunchMode.resolve(info: ["world"], arguments: []) == .lab)
    }

    @Test func theBundleValuesAreParsedLoosely() {
        #expect(LaunchMode.resolve(info: "world", arguments: []) == .world)
        #expect(LaunchMode.resolve(info: " World ", arguments: []) == .world)
        #expect(LaunchMode.resolve(info: "lab", arguments: []) == .lab)
        #expect(LaunchMode.resolve(info: "LAB", arguments: []) == .lab)
    }

    @Test func theArgumentOverridesTheBundleAndTheLastOneWins() {
        #expect(LaunchMode.resolve(info: nil, arguments: ["--launch-mode", "world"]) == .world)
        #expect(LaunchMode.resolve(info: nil, arguments: ["--launch-mode=world"]) == .world)
        #expect(LaunchMode.resolve(info: "world", arguments: ["--launch-mode", "lab"]) == .lab)
        #expect(LaunchMode.resolve(info: nil, arguments: ["--launch-mode=world", "--launch-mode=lab"]) == .lab)
        #expect(LaunchMode.resolve(info: nil, arguments: ["--launch-mode=lab", "--launch-mode", "world"]) == .world)
    }

    @Test func anArgumentThatIsNotAModeLeavesTheBundleInCharge() {
        #expect(LaunchMode.resolve(info: "world", arguments: ["--launch-mode", "storm"]) == .world)
        #expect(LaunchMode.resolve(info: "world", arguments: ["--launch-mode"]) == .world)  // no value
        #expect(LaunchMode.resolve(info: nil, arguments: ["--launch-mode", "storm"]) == .lab)
        #expect(LaunchMode.resolve(info: "world", arguments: ["--world", "planet_a"]) == .world)
    }

    @Test func theKeyAndTheArgumentAreWhatTheScriptWrites() {
        #expect(LaunchMode.infoKey == "AlifeLaunchMode")
        #expect(LaunchMode.argumentName == "--launch-mode")
        #expect(LaunchMode.allCases.map(\.rawValue) == ["lab", "world"])
    }

    /// A bundle on disk with an Info.plist, as `make-app.sh` writes it.
    private func bundle(_ scratch: ScratchDirectory, mode: String?) throws -> Bundle {
        let contents = scratch.url.appendingPathComponent("T-\(mode ?? "none").app/Contents")
        try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
        var plist: [String: Any] = ["CFBundleIdentifier": "io.example.test", "CFBundleName": "T"]
        if let mode { plist[LaunchMode.infoKey] = mode }
        try (plist as NSDictionary).write(to: contents.appendingPathComponent("Info.plist"))
        return try #require(Bundle(url: contents.deletingLastPathComponent()))
    }

    @Test func theModeIsReadFromTheBundle() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        #expect(LaunchMode.current(bundle: try bundle(scratch, mode: "world"), arguments: []) == .world)
        #expect(LaunchMode.current(bundle: try bundle(scratch, mode: "lab"), arguments: []) == .lab)
        #expect(LaunchMode.current(bundle: try bundle(scratch, mode: nil), arguments: []) == .lab)
        #expect(LaunchMode.current(bundle: try bundle(scratch, mode: "world"), arguments: ["--launch-mode=lab"]) == .lab)
    }

    @Test func theTestRunnerIsTheLab() {
        // The test bundle has no key: `swift test` and `swift run` keep today's behaviour.
        #expect(LaunchMode.current(arguments: []) == .lab)
    }

    // MARK: What each mode does with its windows

    @Test func theWindowIdentifiersAreTheOnesTheScenesUse() {
        #expect(AppWindow.allCases.map(\.rawValue) == ["lab", "world", "viewer3d"])
    }

    @Test func theLabOpensItsMainWindowAndTheWorldOnlyForTheArgument() {
        let plan = LaunchMode.lab.plan
        #expect(plan.windowsAtLaunch() == [.lab])
        #expect(plan.windowsAtLaunch(worldArgument: "planet_a") == [.lab, .world])
        #expect(plan.firstScene == .lab)
        #expect(plan.appName == "Alife Control")
        #expect(plan.labIsMainWindow)
        #expect(plan.windowsClosedWithLab == [.viewer3d, .world])
        #expect(plan.windowMenu.isEmpty)
    }

    @Test func stormWorldsOpensOnlyTheWorldWindowAndNeverTheLab() {
        let plan = LaunchMode.world.plan
        #expect(plan.windowsAtLaunch() == [.world])
        #expect(plan.windowsAtLaunch(worldArgument: "planet_a") == [.world])  // the argument starts a world in it
        #expect(plan.firstScene == .world)
        #expect(plan.appName == "Storm Worlds")
        #expect(!plan.labIsMainWindow)
        #expect(!plan.windowsAtLaunch().contains(.lab))
    }

    @Test func closingTheLabInStormWorldsLeavesTheWorldAlone() {
        #expect(LaunchMode.world.plan.windowsClosedWithLab == [.viewer3d])
    }

    @Test func stormWorldsHasMenuItemsForBothWindowsWithDistinctKeys() {
        let items = LaunchMode.world.plan.windowMenu
        #expect(items.map(\.window) == [.world, .lab])
        #expect(items.map(\.title) == ["World", "Lab"])
        #expect(Set(items.map(\.key)).count == items.count)
        #expect(items.map(\.key) == ["1", "2"])
    }

    @Test func bothAppsQuitWhenTheirLastWindowCloses() {
        // The Lab always did; a single-window game does the same, and the Window menu reopens a window while one is open.
        for mode in LaunchMode.allCases { #expect(mode.plan.quitsAfterLastWindow) }
    }
}

import Foundation
import Testing

@testable import AlifeCore

/// What `alife` is now that it is the only app (docs/APP.md §6, PR-H): its names, the argument of the old apps that is accepted and
/// ignored, and the window it opens. This replaced `LaunchModeTests` and `AlifeModeTests`, which pinned the Lab and Storm Worlds.
@Suite struct AppPlanTests {
    @Test func theNamesAreTheOnesTheBundleUses() {
        #expect(AppPlan.name == "alife")
        #expect(AppPlan.bundleID == "io.github.magicbundle.alife")
        #expect(AppPlan.infoKey == "AlifeLaunchMode")
        #expect(AppPlan.ignoredArgument == "--launch-mode")
        // the bundle id is the preferences domain the old apps shared, which is where the folder they stored already is
        #expect(RepoPathStore.sharedSuite == AppPlan.bundleID)
    }

    @Test func theMainWindowIsOneWindowOfItsOwn() {
        #expect(AppShell.mainWindowID == "main")
        #expect(AppShell.minWidth == 1100 && AppShell.minHeight == 700)
        #expect(Set([AppShell.mainWindowID, AppShell.setupWindowID, AppShell.helpWindowID]).count == 3)
    }

    @Test func theOldLaunchModeArgumentChangesNothing() {
        // `--launch-mode lab|world` of the old apps is read by nothing; `--world` works with it in front, behind or between
        #expect(WorldLaunchArgument.parse(["--launch-mode", "lab", "--world", "planet_a"]) == "planet_a")
        #expect(WorldLaunchArgument.parse(["--world", "planet_a", "--launch-mode=world"]) == "planet_a")
        #expect(WorldLaunchArgument.parse(["--launch-mode", "world"]) == nil)
        #expect(WorldLaunchArgument.parse(["--launch-mode=lab", "--world=planet_b", "--launch-mode", "world"]) == "planet_b")
    }

    @Test func theKeyIsWhatMarksABundleForThePreferenceStores() {
        // A process without the key (swift run, the tests) keeps its own preferences and opens no other domain...
        var asked: [String] = []
        _ = RepoPathStore.forApp(info: nil, environment: [:], standard: MemoryDomain(), suite: { asked.append($0); return nil })
        #expect(asked.isEmpty)
        // ...a bundle that has it reads the preview's and the two old apps' domains, by name, and never its own identifier as a suite.
        _ = RepoPathStore.forApp(info: "alife", environment: [:], standard: MemoryDomain(), suite: { asked.append($0); return nil })
        #expect(asked == [RepoPathStore.previewDomain, RepoPathStore.stormDomain, RepoPathStore.labDomain])
        #expect(!asked.contains(AppPlan.bundleID))
    }

    /// A bundle on disk with an Info.plist, as `make-app.sh` writes it.
    @Test func theKeyIsReadFromABundleOnDisk() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let contents = scratch.url.appendingPathComponent("T.app/Contents")
        try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
        let plist: [String: Any] = ["CFBundleIdentifier": "io.example.test", "CFBundleName": "T", AppPlan.infoKey: "alife"]
        try (plist as NSDictionary).write(to: contents.appendingPathComponent("Info.plist"))
        let bundle = try #require(Bundle(url: contents.deletingLastPathComponent()))
        #expect(bundle.object(forInfoDictionaryKey: AppPlan.infoKey) as? String == "alife")
        // the test runner has none
        #expect(Bundle.main.object(forInfoDictionaryKey: AppPlan.infoKey) == nil)
    }
}

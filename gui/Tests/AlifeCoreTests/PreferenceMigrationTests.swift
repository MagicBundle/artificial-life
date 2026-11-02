import Foundation
import Testing

@testable import AlifeCore

/// `PreferenceMigration` (docs/APP.md §6): precedence, "only where unset", idempotence, never deleting, never writing to a source.
@Suite struct PreferenceMigrationTests {
    /// `own` is the app's domain, which is also the shared one the old apps wrote their `repoPath` into.
    private func domains() -> (own: MemoryDomain, storm: MemoryDomain, lab: MemoryDomain) {
        (MemoryDomain(), MemoryDomain(), MemoryDomain())
    }

    private func migration(_ d: (own: MemoryDomain, storm: MemoryDomain, lab: MemoryDomain)) -> PreferenceMigration {
        PreferenceMigration(own: d.own, storm: d.storm, lab: d.lab)
    }

    @Test func theFolderComesFromTheAppsOwnDomainThenStormThenTheLab() {
        let d = domains()
        d.lab.set("/lab", forKey: "repoPath")
        migration(d).run()
        #expect(d.own.string(forKey: "repoPath") == "/lab")

        let e = domains()
        e.lab.set("/lab", forKey: "repoPath")
        e.storm.set("/storm", forKey: "repoPath")
        migration(e).run()
        #expect(e.own.string(forKey: "repoPath") == "/storm")

        let f = domains()
        f.lab.set("/lab", forKey: "repoPath")
        f.storm.set("/storm", forKey: "repoPath")
        f.own.set("/shared", forKey: "repoPath")  // the shared domain is the app's own
        migration(f).run()
        #expect(f.own.string(forKey: "repoPath") == "/shared")
    }

    @Test func anEmptyFolderIsNotAFolder() {
        let d = domains()
        d.own.set("", forKey: "repoPath")
        d.storm.set("/storm", forKey: "repoPath")
        migration(d).run()
        #expect(d.own.string(forKey: "repoPath") == "/storm")
    }

    @Test func theFourteenWorldKeysComeFromStormThenTheLabAndOnlyWhereUnset() {
        let d = domains()
        #expect(AppPreferenceKeys.world.count == 14 && Set(AppPreferenceKeys.world).count == 14)
        d.lab.set("lab-side", forKey: "world.dockSide")
        d.lab.set(0.7, forKey: "world.exposure")
        d.lab.set(true, forKey: "world.minimap")
        d.storm.set("storm-side", forKey: "world.dockSide")
        d.storm.set("{\"x\":1}", forKey: "world.autopause")
        d.own.set(false, forKey: "world.minimap")  // the new app already has one: it wins
        migration(d).run()
        #expect(d.own.string(forKey: "world.dockSide") == "storm-side")
        #expect(d.own.object(forKey: "world.exposure") as? Double == 0.7)
        #expect(d.own.object(forKey: "world.minimap") as? Bool == false)
        #expect(d.own.string(forKey: "world.autopause") == "{\"x\":1}")  // the JSON string arrives as a string
        #expect(d.own.object(forKey: "world.globe") == nil)  // nothing invented
    }

    @Test func typesAreKept() {
        let d = domains()
        d.storm.set(true, forKey: "world.dockCollapsed")
        d.storm.set(412.0, forKey: "world.dockExtent")
        d.storm.set(true, forKey: "world.hintShown")
        migration(d).run()
        #expect(d.own.object(forKey: "world.dockCollapsed") as? Bool == true)
        #expect(d.own.object(forKey: "world.dockExtent") as? Double == 412)
        #expect(d.own.object(forKey: "world.hintShown") as? Bool == true)
    }

    @Test func itRunsOnceAndASecondRunChangesNothing() {
        let d = domains()
        d.storm.set("/storm", forKey: "repoPath")
        d.storm.set("left", forKey: "world.dockSide")
        #expect(migration(d).run())
        #expect(d.own.object(forKey: "migrated.v1") as? Bool == true)
        let after = d.own.all.keys.sorted()
        // the sources change afterwards; a second run must not look at them
        d.storm.set("/elsewhere", forKey: "repoPath")
        d.storm.set("right", forKey: "world.dockSide")
        d.storm.set(1.0, forKey: "world.exposure")
        #expect(!migration(d).run())
        #expect(d.own.all.keys.sorted() == after)
        #expect(d.own.string(forKey: "repoPath") == "/storm")
        #expect(d.own.string(forKey: "world.dockSide") == "left")
    }

    @Test func nothingIsDeletedFromOrWrittenToASource() {
        let d = domains()
        d.storm.set("/storm", forKey: "repoPath")
        d.storm.set("left", forKey: "world.dockSide")
        d.lab.set(true, forKey: "world.spin")
        let before = (d.storm.all.mapValues { "\($0)" }, d.lab.all.mapValues { "\($0)" })
        migration(d).run()
        migration(d).run()
        #expect(d.storm.all.mapValues { "\($0)" } == before.0 && d.lab.all.mapValues { "\($0)" } == before.1, "values and keys of the sources are as they were")
        #expect(d.storm.string(forKey: "world.dockSide") == "left" && d.own.string(forKey: "repoPath") == "/storm")
    }

    @Test func aNewUserGetsOnlyTheMark() {
        let d = domains()
        migration(d).run()
        #expect(d.own.all.keys.sorted() == ["migrated.v1"])
    }

    @Test func missingDomainsAreFine() {
        let own = MemoryDomain()
        #expect(PreferenceMigration(own: own, storm: nil, lab: nil).run())
        #expect(own.all.keys.sorted() == ["migrated.v1"])
    }

    @Test func underTheSuiteOverrideThereIsNoMigrationAndNothingIsWrittenOrRead() {
        let own = MemoryDomain()
        var asked: [String] = []
        let m = PreferenceMigration.forApp(
            environment: [RepoPathStore.suiteOverrideEnvironment: "trial-suite"], standard: own,
            suite: { name in asked.append(name); return MemoryDomain() })
        #expect(m == nil, "a trial launch must not mark the real domain as migrated")
        #expect(asked.isEmpty && own.all.isEmpty, "no domain is opened, and the app's own is untouched")
    }

    @Test func withoutAnOverrideTheTwoOldAppsDomainsAreOpenedByNameAndNeverTheAppsOwn() {
        var asked: [String] = []
        let m = PreferenceMigration.forApp(environment: [:], standard: MemoryDomain(), suite: { asked.append($0); return nil })
        #expect(m != nil)
        #expect(asked == ["io.github.magicbundle.stormworlds", "io.github.magicbundle.alifecontrol"])
        #expect(!asked.contains(AppPlan.bundleID), "a bundle cannot open its own identifier as a suite")
    }

    @Test func aWholeLaunchOfTheFinalAppImportsFromBothOldAppsOnceAndLeavesThemAlone() {
        // The app's domain already holds the folder the old apps shared; Storm Worlds' and the Lab's display settings come in.
        let own = MemoryDomain(), storm = MemoryDomain(), lab = MemoryDomain()
        own.set("/shared/folder", forKey: "repoPath")
        storm.set("left", forKey: "world.dockSide")
        lab.set("bottom", forKey: "world.dockSide")
        lab.set(0.5, forKey: "world.exposure")
        func launch() -> Bool {
            PreferenceMigration.forApp(environment: [:], standard: own, suite: { $0 == RepoPathStore.stormDomain ? storm : ($0 == RepoPathStore.labDomain ? lab : nil) })?.run() ?? false
        }
        #expect(launch())
        #expect(own.string(forKey: "repoPath") == "/shared/folder")
        #expect(own.string(forKey: "world.dockSide") == "left" && own.object(forKey: "world.exposure") as? Double == 0.5)
        let after = own.all.mapValues { "\($0)" }
        storm.set("right", forKey: "world.dockSide")  // changed since: the second launch must not look
        #expect(!launch() && own.all.mapValues { "\($0)" } == after, "idempotent")
        #expect(storm.all.count == 1 && lab.all.count == 2, "read only")
    }
}

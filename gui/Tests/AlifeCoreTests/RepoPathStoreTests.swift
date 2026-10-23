import Foundation
import Testing

@testable import AlifeCore

/// The repository folder shared by the Lab app and Storm Worlds, and moved from the Lab's own preferences. Every domain
/// here is in memory: the real preferences are never read or written, and no property list is left behind.
@Suite struct RepoPathStoreTests {
    private let key = RepoPathStore.key

    @Test func nothingStoredIsNil() {
        let store = RepoPathStore(shared: MemoryDomain(), own: MemoryDomain(), legacy: [MemoryDomain()])
        #expect(store.load() == nil)
    }

    @Test func theLabsStoredFolderIsCopiedIntoTheSuiteAndLeftWhereItWas() {
        let shared = MemoryDomain(), own = MemoryDomain()
        own.set("/work/artificial-life", forKey: key)
        let store = RepoPathStore(shared: shared, own: own)
        #expect(store.load() == "/work/artificial-life")
        #expect(shared.string(forKey: key) == "/work/artificial-life")
        #expect(own.string(forKey: key) == "/work/artificial-life")
    }

    @Test func stormWorldsTakesTheFolderOfTheLabsDomain() {
        let shared = MemoryDomain(), lab = MemoryDomain()
        lab.set("/work/artificial-life", forKey: key)
        // Storm Worlds: its own preferences are empty, the Lab's are a legacy source.
        let storm = RepoPathStore(shared: shared, own: MemoryDomain(), legacy: [lab])
        #expect(storm.load() == "/work/artificial-life")
        #expect(shared.string(forKey: key) == "/work/artificial-life")
        #expect(lab.string(forKey: key) == "/work/artificial-life")
    }

    @Test func theSuiteWinsOverAnOlderValueInTheLabsDomain() {
        let shared = MemoryDomain(), lab = MemoryDomain()
        shared.set("/new", forKey: key)
        lab.set("/old", forKey: key)
        let store = RepoPathStore(shared: shared, own: MemoryDomain(), legacy: [lab])
        #expect(store.load() == "/new")
        #expect(lab.string(forKey: key) == "/old")
    }

    @Test func itsOwnPreferencesComeBeforeTheLabs() {
        let own = MemoryDomain(), lab = MemoryDomain()
        own.set("/mine", forKey: key)
        lab.set("/labs", forKey: key)
        #expect(RepoPathStore(shared: MemoryDomain(), own: own, legacy: [lab]).load() == "/mine")
    }

    @Test func aFolderChosenInOneAppIsTheOtherOnesAtItsNextStart() {
        let shared = MemoryDomain(), labOwn = MemoryDomain(), stormOwn = MemoryDomain()
        let lab = RepoPathStore(shared: shared, own: labOwn)
        let storm = RepoPathStore(shared: shared, own: stormOwn, legacy: [labOwn])
        storm.save("/chosen/in/storm")
        #expect(lab.load() == "/chosen/in/storm")
        lab.save("/chosen/in/lab")
        #expect(storm.load() == "/chosen/in/lab")
    }

    @Test func aSaveIsMirroredToTheAppsOwnPreferencesForAnOlderBuild() {
        let shared = MemoryDomain(), own = MemoryDomain()
        RepoPathStore(shared: shared, own: own).save("/x")
        #expect(shared.string(forKey: key) == "/x")
        #expect(own.string(forKey: key) == "/x")
    }

    @Test func anEmptyStoredValueCountsAsNone() {
        let shared = MemoryDomain(), lab = MemoryDomain()
        shared.set("", forKey: key)
        lab.set("/labs", forKey: key)
        #expect(RepoPathStore(shared: shared, own: MemoryDomain(), legacy: [lab]).load() == "/labs")
    }

    @Test func aMissingFolderIsStillReturnedAsStored() {
        // Whether it exists is `RepoFolder.initialPath`'s business (the fallback of #36), not the store's.
        let store = RepoPathStore(shared: MemoryDomain(), own: MemoryDomain())
        store.save("/does/not/exist")
        #expect(store.load() == "/does/not/exist")
    }

    // MARK: The store of a process

    /// Hands out the named domains and records which names were asked for.
    private final class Suites: @unchecked Sendable {
        var domains: [String: MemoryDomain] = [:]
        var asked: [String] = []

        func open(_ name: String) -> (any PreferenceDomain)? {
            asked.append(name)
            let domain = domains[name] ?? MemoryDomain()
            domains[name] = domain
            return domain
        }
    }

    @Test func anUnbundledProcessKeepsItsOwnPreferencesAlone() {
        let standard = MemoryDomain(), suites = Suites()
        standard.set("/worktree", forKey: key)
        let store = RepoPathStore.forApp(info: nil, environment: [:], standard: standard, suite: suites.open)
        #expect(store.load() == "/worktree")
        store.save("/other")
        #expect(standard.string(forKey: key) == "/other")
        #expect(suites.asked.isEmpty, "no shared suite is opened: swift run stays separate from the installed apps")
    }

    @Test func aBundleWithTheKeyUsesTheSharedSuiteAndMigratesFromTheLabsDomain() {
        let standard = MemoryDomain(), suites = Suites()
        let lab = MemoryDomain()
        lab.set("/labs", forKey: key)
        suites.domains[RepoPathStore.labDomain] = lab
        let store = RepoPathStore.forApp(info: "world", environment: [:], standard: standard, suite: suites.open)
        #expect(suites.asked == [RepoPathStore.sharedSuite, RepoPathStore.labDomain])
        #expect(store.load() == "/labs")
        #expect(suites.domains[RepoPathStore.sharedSuite]?.string(forKey: key) == "/labs")
        store.save("/chosen")
        #expect(suites.domains[RepoPathStore.sharedSuite]?.string(forKey: key) == "/chosen")
        #expect(lab.string(forKey: key) == "/labs", "the Lab's own preferences are not written from the game")
    }

    @Test func theOverrideSuiteReplacesTheSharedOneAndTheLabsDomain() {
        let standard = MemoryDomain(), suites = Suites()
        standard.set("/own", forKey: key)
        let store = RepoPathStore.forApp(
            info: "world", environment: [RepoPathStore.suiteOverrideEnvironment: "test-suite"], standard: standard,
            suite: suites.open)
        #expect(suites.asked == ["test-suite"])
        #expect(store.load() == "/own")  // migrated from the app's own preferences into the override suite
        #expect(suites.domains["test-suite"]?.string(forKey: key) == "/own")
        store.save("/saved")
        #expect(suites.domains["test-suite"]?.string(forKey: key) == "/saved")
    }

    @Test func theNamesAreTheOnesTheBundlesUse() {
        #expect(RepoPathStore.key == "repoPath")  // unchanged: the Lab has always stored it under this key
        #expect(RepoPathStore.sharedSuite == "io.github.magicbundle.alife")
        #expect(RepoPathStore.labDomain == "io.github.magicbundle.alifecontrol")
        #expect(RepoPathStore.suiteOverrideEnvironment == "ALIFE_PREFS_SUITE")
    }
}

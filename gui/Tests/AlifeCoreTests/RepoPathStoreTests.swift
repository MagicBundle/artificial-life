import Foundation
import Testing

@testable import AlifeCore

/// The alife folder: kept in the app's own preferences (the domain the Lab and Storm Worlds shared), taken once from their domains,
/// which are only read. Every domain here is in memory: the real preferences are never read or written, and no property list is left behind.
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

    @Test func aSecondAppTakesTheFolderOfTheLabsDomain() {
        let shared = MemoryDomain(), lab = MemoryDomain()
        lab.set("/work/artificial-life", forKey: key)
        // A second app: its own preferences are empty, the Lab's are a legacy source.
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

    @Test func aFolderChosenInAnOldAppIsTheNewOnesAtItsNextStartAndTheOtherWayRound() {
        // The old apps write the shared domain, which is the new app's own (the same bundle identifier).
        let domain = MemoryDomain(), labOwn = MemoryDomain()
        let oldApp = RepoPathStore(shared: domain, own: labOwn)
        let app = RepoPathStore(shared: domain, own: domain)
        oldApp.save("/chosen/in/the/old/app")
        #expect(app.load() == "/chosen/in/the/old/app")
        app.save("/chosen/in/alife")
        #expect(oldApp.load() == "/chosen/in/alife")
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

    @Test func aBundleWithTheKeyUsesItsOwnDomainAndReadsTheOldAppsFromTheirs() {
        let standard = MemoryDomain(), suites = Suites()
        let lab = MemoryDomain(), storm = MemoryDomain()
        lab.set("/labs", forKey: key)
        suites.domains[RepoPathStore.labDomain] = lab
        suites.domains[RepoPathStore.stormDomain] = storm
        let store = RepoPathStore.forApp(info: "alife", environment: [:], standard: standard, suite: suites.open)
        #expect(suites.asked == [RepoPathStore.previewDomain, RepoPathStore.stormDomain, RepoPathStore.labDomain], "never its own identifier as a suite")
        #expect(store.load() == "/labs")
        #expect(standard.string(forKey: key) == "/labs", "copied into the app's own domain")
        store.save("/chosen")
        #expect(standard.string(forKey: key) == "/chosen")
        #expect(lab.string(forKey: key) == "/labs" && storm.all.isEmpty, "the old apps' domains are only read")
    }

    @Test func theAppsOwnFolderBeatsStormWorldsAndStormWorldsBeatsTheLab() {
        let suites = Suites()
        suites.domains[RepoPathStore.labDomain] = { let d = MemoryDomain(); d.set("/lab", forKey: key); return d }()
        suites.domains[RepoPathStore.stormDomain] = { let d = MemoryDomain(); d.set("/storm", forKey: key); return d }()
        let fresh = MemoryDomain()
        #expect(RepoPathStore.forApp(info: "alife", environment: [:], standard: fresh, suite: suites.open).load() == "/storm")
        let own = MemoryDomain()
        own.set("/own", forKey: key)
        #expect(RepoPathStore.forApp(info: "alife", environment: [:], standard: own, suite: suites.open).load() == "/own")
    }

    @Test func theOverrideSuiteReplacesTheSharedOneAndTheLabsDomain() {
        let standard = MemoryDomain(), suites = Suites()
        standard.set("/own", forKey: key)
        let store = RepoPathStore.forApp(
            info: "alife", environment: [RepoPathStore.suiteOverrideEnvironment: "test-suite"], standard: standard,
            suite: suites.open)
        #expect(suites.asked == ["test-suite"])
        #expect(store.load() == "/own")  // migrated from the app's own preferences into the override suite
        #expect(suites.domains["test-suite"]?.string(forKey: key) == "/own")
        store.save("/saved")
        #expect(suites.domains["test-suite"]?.string(forKey: key) == "/saved")
    }

    @Test func theOverrideNeverWritesTheAppsOwnPreferences() {
        let standard = MemoryDomain(), suites = Suites()
        standard.set("/owners", forKey: key)
        let store = RepoPathStore.forApp(
            info: "alife", environment: [RepoPathStore.suiteOverrideEnvironment: "test-suite"], standard: standard,
            suite: suites.open)
        _ = store.load()
        store.save("/owners")  // what the Repository field does on its own, unchanged
        store.save("/typed")
        #expect(standard.string(forKey: key) == "/owners", "the owner's domain is only read")
        #expect(suites.domains["test-suite"]?.string(forKey: key) == "/typed")
        #expect(store.load() == "/typed")
    }

    @Test func theOverrideWithNothingAnywhereStartsEmptyAndStaysInTheSuite() {
        let standard = MemoryDomain(), suites = Suites()
        let store = RepoPathStore.forApp(
            info: nil, environment: [RepoPathStore.suiteOverrideEnvironment: "test-suite"], standard: standard,
            suite: suites.open)
        #expect(store.load() == nil)
        store.save("/x")
        #expect(standard.string(forKey: key) == nil)
    }

    @Test func anOverrideThatNamesTheAppsOwnOrAnOlderDomainIsRefused() {
        for name in [AppPlan.bundleID, RepoPathStore.previewDomain, RepoPathStore.stormDomain, RepoPathStore.labDomain] {
            let env = [RepoPathStore.suiteOverrideEnvironment: name]
            #expect(RepoPathStore.overrideProblem(environment: env)?.contains(name) == true, Comment(rawValue: name))
        }
        #expect(RepoPathStore.overrideProblem(environment: [RepoPathStore.suiteOverrideEnvironment: "alife-private-trial"]) == nil)
        #expect(RepoPathStore.overrideProblem(environment: [:]) == nil)
        // and the store never opens such a name as the override even if asked directly
        let suites = Suites(), standard = MemoryDomain()
        _ = RepoPathStore.forApp(info: "alife", environment: [RepoPathStore.suiteOverrideEnvironment: RepoPathStore.stormDomain],
                                 standard: standard, suite: suites.open)
        #expect(suites.asked == [RepoPathStore.previewDomain, RepoPathStore.stormDomain, RepoPathStore.labDomain], "only the read-only sources are opened")
    }

    @Test func theNamesAreTheOnesTheBundlesUse() {
        #expect(RepoPathStore.previewDomain == "io.github.magicbundle.alife.preview")
        #expect(RepoPathStore.key == "repoPath")  // unchanged: the Lab has always stored it under this key
        #expect(RepoPathStore.sharedSuite == "io.github.magicbundle.alife")
        #expect(RepoPathStore.stormDomain == "io.github.magicbundle.stormworlds")
        #expect(RepoPathStore.labDomain == "io.github.magicbundle.alifecontrol")
        #expect(RepoPathStore.suiteOverrideEnvironment == "ALIFE_PREFS_SUITE")
    }
}

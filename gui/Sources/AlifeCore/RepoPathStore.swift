import Foundation

/// A preferences domain, as much of it as the repository folder needs. `UserDefaults` is one; the tests use one in memory,
/// so that they leave no property list in the owner's Library.
package protocol PreferenceDomain: Sendable {
    func string(forKey key: String) -> String?
    func store(_ value: String, forKey key: String)
}

extension UserDefaults: PreferenceDomain {
    package func store(_ value: String, forKey key: String) { set(value, forKey: key) }
}

/// Where the repository folder is remembered, shared by the Lab app and Storm Worlds so that it is set once.
///
/// Both bundles keep it under `RepoPathStore.key` of one shared defaults suite. A value that is not there yet is taken
/// from the app's own preferences and from the Lab's (`labDomain`, where the Lab has always kept it) and copied into the
/// suite on first read; what it was copied from is left alone. A save writes the suite and mirrors the value into the
/// app's own preferences, so that an older Lab build, which reads only those, keeps agreeing.
///
/// Only a bundle made by `make-app.sh` (it has `LaunchMode.infoKey`) uses the suite. `swift run` and the tests keep their own
/// preferences domain, separate from the installed apps, as they always did.
package struct RepoPathStore: Sendable {
    package static let key = "repoPath"
    /// The shared suite (a plain preferences domain: `~/Library/Preferences/<suite>.plist`; no app group needed).
    package static let sharedSuite = "io.github.magicbundle.alife"
    /// The Lab's own bundle ID, the preferences domain the stored folder is migrated from.
    package static let labDomain = "io.github.magicbundle.alifecontrol"
    /// Names a suite to use instead of `sharedSuite` (and no migration from `labDomain`): for trying a bundle without touching
    /// the owner's preferences.
    package static let suiteOverrideEnvironment = "ALIFE_PREFS_SUITE"

    private let shared: any PreferenceDomain
    private let own: any PreferenceDomain
    private let legacy: [any PreferenceDomain]

    /// `legacy` are the other places a value may already be, in order.
    package init(shared: any PreferenceDomain, own: any PreferenceDomain, legacy: [any PreferenceDomain] = []) {
        self.shared = shared
        self.own = own
        self.legacy = legacy
    }

    /// The stored folder, migrating it into the suite when the suite has none. Nil when nothing was ever stored.
    package func load() -> String? {
        if let value = Self.value(in: shared) { return value }
        for defaults in [own] + legacy {
            if let value = Self.value(in: defaults) {
                shared.store(value, forKey: Self.key)
                return value
            }
        }
        return nil
    }

    package func save(_ path: String) {
        shared.store(path, forKey: Self.key)
        own.store(path, forKey: Self.key)
    }

    private static func value(in defaults: any PreferenceDomain) -> String? {
        guard let value = defaults.string(forKey: key), !value.isEmpty else { return nil }
        return value
    }

    /// The store for a process. `info` is its bundle's `AlifeLaunchMode`: without it (unbundled, or a bundle from before the
    /// key) the app's own preferences are used alone, with no sharing. `suite` opens a preferences domain by name.
    package static func forApp(
        info: Any?, environment: [String: String] = ProcessInfo.processInfo.environment,
        standard: any PreferenceDomain = UserDefaults.standard,
        suite: (String) -> (any PreferenceDomain)? = { UserDefaults(suiteName: $0) }
    ) -> RepoPathStore {
        if let name = environment[suiteOverrideEnvironment], !name.isEmpty, let override = suite(name) {
            return RepoPathStore(shared: override, own: standard)
        }
        guard info != nil, let shared = suite(sharedSuite) else {
            return RepoPathStore(shared: standard, own: standard)
        }
        // The Lab's domain cannot be opened as a suite from the Lab itself (that is `standard`, already in the list).
        let lab = suite(labDomain)
        return RepoPathStore(shared: shared, own: standard, legacy: lab.map { [$0] } ?? [])
    }
}

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

/// Where the alife folder is remembered. `alife` keeps it in its own preferences, which are the domain the Lab and Storm Worlds shared
/// (`sharedSuite` is the app's bundle identifier), so a folder chosen in either is already there. A value that is not there yet is
/// taken, read only, from Storm Worlds' and the Lab's own domains (`legacy`, opened by name) and copied into the app's domain on first
/// read; what it was copied from is left alone. A bundle cannot open its own identifier as a suite, so the app's domain is `standard`.
///
/// Only a bundle made by `make-app.sh` (it has `AppPlan.infoKey`) reads the old domains. `swift run` and the tests keep their own
/// preferences domain, separate from the installed app, as they always did.
package struct RepoPathStore: Sendable {
    package static let key = "repoPath"
    /// The app's own preferences domain, the one the old apps shared: `~/Library/Preferences/<suite>.plist`. Never opened as a suite
    /// by the app itself (that is `standard`).
    package static let sharedSuite = AppPlan.bundleID
    /// The Lab's own bundle ID, a domain the stored folder is migrated from.
    package static let labDomain = "io.github.magicbundle.alifecontrol"
    /// Storm Worlds' bundle ID, the other domain it is migrated from.
    package static let stormDomain = "io.github.magicbundle.stormworlds"
    /// Names a suite to use instead of the app's preferences, for trying a bundle without touching the owner's preferences: it is read
    /// and written in place of them (which are only read, as a source for the first value, and never written), and the old apps'
    /// domains are not used.
    package static let suiteOverrideEnvironment = "ALIFE_PREFS_SUITE"

    private let shared: any PreferenceDomain
    private let own: any PreferenceDomain
    private let legacy: [any PreferenceDomain]

    /// `legacy` are the other places a value may already be, in order, read only.
    package init(shared: any PreferenceDomain, own: any PreferenceDomain, legacy: [any PreferenceDomain] = []) {
        self.shared = shared
        self.own = own
        self.legacy = legacy
    }

    /// The stored folder, migrating it into the shared domain when that has none. Nil when nothing was ever stored.
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

    /// The store for a process. `info` is its bundle's `AppPlan.infoKey`: without it (unbundled, or a bundle from before the key) the
    /// app's own preferences are used alone. `suite` opens a preferences domain by name.
    package static func forApp(
        info: Any?, environment: [String: String] = ProcessInfo.processInfo.environment,
        standard: any PreferenceDomain = UserDefaults.standard,
        suite: (String) -> (any PreferenceDomain)? = { UserDefaults(suiteName: $0) }
    ) -> RepoPathStore {
        if let name = environment[suiteOverrideEnvironment], !name.isEmpty, let override = suite(name) {
            // `own` is the override too, so that a save never reaches `standard` (an installed app's `standard` is the owner's
            // domain); `standard` stays a read-only source for the first value.
            return RepoPathStore(shared: override, own: override, legacy: [standard])
        }
        guard info != nil else { return RepoPathStore(shared: standard, own: standard) }
        // The app's domain is `standard` (the shared one): only the two old apps' domains are opened, by name, and only read.
        return RepoPathStore(shared: standard, own: standard, legacy: [stormDomain, labDomain].compactMap { suite($0) })
    }
}

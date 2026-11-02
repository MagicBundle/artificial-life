import Foundation

/// A preferences domain that also holds values that are not strings (the `world.*` keys are booleans, numbers and a JSON string).
/// `UserDefaults` is one; the tests use one in memory (docs/APP.md §6).
package protocol TypedPreferenceDomain: PreferenceDomain {
    func object(forKey key: String) -> Any?
    func set(_ value: Any?, forKey key: String)
}

extension UserDefaults: TypedPreferenceDomain {}

/// The keys of the preferences of the `alife` app, in its own domain: the alife folder (`RepoPathStore.key`), the 14 display and dock keys of the World window, and the page the window opened on.
package enum AppPreferenceKeys {
    /// The 14 keys of the World window, as the Lab app and Storm Worlds keep them (docs/APP.md §1.1): the dock (side, two
    /// extents, collapsed), the pauses (a JSON string), the display (brightness, scroll, minimap, globe, spin, north up, height
    /// colour), the confirmation shortcut for a power of cost one and the one-line hint.
    package static let world: [String] = [
        "world.dockSide", "world.dockExtent", "world.dockExtentBottom", "world.dockCollapsed", "world.autopause",
        "world.exposure", "world.scrollZooms", "world.minimap", "world.globe", "world.spin", "world.keepNorthUp",
        "world.heightColour", "world.skipConfirmCostOne", "world.hintShown",
    ]
    /// Set once the migration has run (`PreferenceMigration`).
    package static let migrated = "migrated.v1"
}

/// Brings the settings of the two old apps into the preferences of `alife`, once (docs/APP.md §6). It never deletes and never writes
/// anywhere but the domain it is given; it is idempotent (`migrated.v1`), so a value the app has already set is not replaced and
/// a second run changes nothing.
///
/// The app's own domain is the one the old apps shared (its bundle identifier), so a `repoPath` they stored there is already in
/// `own`. The old apps' own domains are read by name (a bundle cannot open its own identifier as a suite, and `forApp` does not try).
///
/// - `repoPath`: the app's own, else Storm Worlds', else the Lab's, only when the app has none.
/// - the 14 `world.*` keys: Storm Worlds' value, else the Lab's, only where the app has none.
///
/// Window frames and saved state are not migrated (their names derive from scene ids).
package struct PreferenceMigration {
    package let own: any TypedPreferenceDomain
    package let storm: (any TypedPreferenceDomain)?
    package let lab: (any TypedPreferenceDomain)?

    package init(own: any TypedPreferenceDomain, storm: (any TypedPreferenceDomain)?, lab: (any TypedPreferenceDomain)?) {
        self.own = own
        self.storm = storm
        self.lab = lab
    }

    /// Runs the migration unless it has run. True when it did something (it ran and set the mark).
    @discardableResult
    package func run() -> Bool {
        guard own.object(forKey: AppPreferenceKeys.migrated) == nil else { return false }
        if Self.folder(in: own) == nil {
            let found = [storm, lab].compactMap { $0 }.lazy.compactMap { Self.folder(in: $0) }.first
            if let found { own.store(found, forKey: RepoPathStore.key) }
        }
        for key in AppPreferenceKeys.world where own.object(forKey: key) == nil {
            if let value = [storm, lab].compactMap({ $0 }).lazy.compactMap({ $0.object(forKey: key) }).first {
                own.set(value, forKey: key)
            }
        }
        own.set(true, forKey: AppPreferenceKeys.migrated)
        return true
    }

    private static func folder(in domain: any PreferenceDomain) -> String? {
        guard let value = domain.string(forKey: RepoPathStore.key), !value.isEmpty else { return nil }
        return value
    }

    /// The migration for this process: into `standard` (the app's own domain), from the two old apps' domains, which are only read.
    /// **With `ALIFE_PREFS_SUITE` set (a trial that must not touch the owner's preferences) there is no migration at all** (nil): it
    /// would mark the real domain as migrated with the trial's values, and the first real launch would never import the old apps' settings.
    package static func forApp(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        standard: any TypedPreferenceDomain = UserDefaults.standard,
        suite: (String) -> (any TypedPreferenceDomain)? = { UserDefaults(suiteName: $0) }
    ) -> PreferenceMigration? {
        if let name = environment[RepoPathStore.suiteOverrideEnvironment], !name.isEmpty { return nil }
        return PreferenceMigration(own: standard, storm: suite(RepoPathStore.stormDomain), lab: suite(RepoPathStore.labDomain))
    }
}

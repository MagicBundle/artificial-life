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
    /// Other keys of the preview that the app carries (`PreferenceMigration`): what closing the window with a run going does, whether Watch
    /// picks up where it left off, the autopilot's address and model, and the set-up page's toggles and remembered edits.
    package static let previewCarried: [String] = [
        "shell.closeWithRun", "shell.pickUp", "shell.autopilotURL", "shell.autopilotModel", "setup.advanced", "setup.showCommand",
        "setup.remembered.v1",
    ]
    /// Set once the migration has run (`PreferenceMigration`).
    package static let migrated = "migrated.v1"
}

/// Brings the settings of the preview and the two old apps into the preferences of `alife`, once (docs/APP.md §6). It never deletes and
/// never writes anywhere but the domain it is given; it is idempotent (`migrated.v1`), so a value the app has already set is not replaced
/// and a second run changes nothing.
///
/// The app's own domain is the one the old apps shared (its bundle identifier), so a `repoPath` they stored there is already in
/// `own`. The other domains are read by name (a bundle cannot open its own identifier as a suite, and `forApp` does not try).
///
/// - `repoPath`: the preview's, when it names an existing folder (it is the owner's latest deliberate choice: the preview never wrote the
///   shared domain), else the app's own; with none, Storm Worlds', else the Lab's.
/// - the 14 `world.*` keys and `AppPreferenceKeys.previewCarried` (the close answer, pick-up, the autopilot's address and model, the set-up
///   page's two toggles and its remembered edits): the preview's value, else Storm Worlds', else the Lab's, only where the app has none.
///   `shell.page` is not carried (the app opens on Worlds).
///
/// Window frames and saved state are not migrated (their names derive from scene ids).
package struct PreferenceMigration {
    package let own: any TypedPreferenceDomain
    package let preview: (any TypedPreferenceDomain)?
    package let storm: (any TypedPreferenceDomain)?
    package let lab: (any TypedPreferenceDomain)?
    /// Whether a folder exists (the tests replace it).
    package let folderExists: @Sendable (String) -> Bool

    package init(own: any TypedPreferenceDomain, preview: (any TypedPreferenceDomain)? = nil, storm: (any TypedPreferenceDomain)?,
                 lab: (any TypedPreferenceDomain)?,
                 folderExists: @escaping @Sendable (String) -> Bool = { var d: ObjCBool = false; return FileManager.default.fileExists(atPath: $0, isDirectory: &d) && d.boolValue }) {
        self.own = own
        self.preview = preview
        self.storm = storm
        self.lab = lab
        self.folderExists = folderExists
    }

    /// Runs the migration unless it has run. True when it did something (it ran and set the mark).
    @discardableResult
    package func run() -> Bool {
        guard own.object(forKey: AppPreferenceKeys.migrated) == nil else { return false }
        if let chosen = preview.flatMap({ Self.folder(in: $0) }), folderExists(chosen) {
            own.store(chosen, forKey: RepoPathStore.key)
        } else if Self.folder(in: own) == nil {
            let found = [storm, lab].compactMap { $0 }.lazy.compactMap { Self.folder(in: $0) }.first
            if let found { own.store(found, forKey: RepoPathStore.key) }
        }
        let sources = [preview, storm, lab].compactMap { $0 }
        for key in AppPreferenceKeys.world + AppPreferenceKeys.previewCarried where own.object(forKey: key) == nil {
            if let value = sources.lazy.compactMap({ $0.object(forKey: key) }).first { own.set(value, forKey: key) }
        }
        own.set(true, forKey: AppPreferenceKeys.migrated)
        return true
    }

    private static func folder(in domain: any PreferenceDomain) -> String? {
        guard let value = domain.string(forKey: RepoPathStore.key), !value.isEmpty else { return nil }
        return value
    }

    /// The migration for this process: into `standard` (the app's own domain), from the preview's and the two old apps' domains, which are
    /// only read. **With `ALIFE_PREFS_SUITE` set (a trial that must not touch the owner's preferences) there is no migration at all** (nil): it
    /// would mark the real domain as migrated with the trial's values, and the first real launch would never import the old settings.
    package static func forApp(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        standard: any TypedPreferenceDomain = UserDefaults.standard,
        suite: (String) -> (any TypedPreferenceDomain)? = { UserDefaults(suiteName: $0) }
    ) -> PreferenceMigration? {
        if let name = environment[RepoPathStore.suiteOverrideEnvironment], !name.isEmpty { return nil }
        return PreferenceMigration(own: standard, preview: suite(RepoPathStore.previewDomain), storm: suite(RepoPathStore.stormDomain),
                                   lab: suite(RepoPathStore.labDomain))
    }
}

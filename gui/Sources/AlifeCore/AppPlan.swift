import Foundation

/// What `alife` is, in one place (docs/APP.md §6, PR-H). The executable used to be three apps (the research Lab, Storm Worlds and
/// the `alife` preview) chosen by a `LaunchMode` read from the bundle; since the old modes were retired it is one app, and what
/// was a per-mode plan is a constant of it: it quits with its last window unless a run is kept going, it asks before it stops a run,
/// and its quit waits for the run's save point.
package enum AppPlan {
    /// The name in the menu bar and the About box (the bundle's `CFBundleName` says the same).
    package static let name = "alife"
    /// The bundle identifier, which is also the preferences domain the old apps shared (`RepoPathStore.sharedSuite`).
    package static let bundleID = "io.github.magicbundle.alife"
    /// The `Info.plist` key that `make-app.sh` writes ("alife"). Its presence marks a bundle made by the script: only such a process
    /// reads the old apps' preferences (`RepoPathStore.forApp`, `PreferenceMigration.forApp`); `swift run` and the tests keep their own.
    package static let infoKey = "AlifeLaunchMode"
    /// An argument of the old apps (`--launch-mode lab|world`), still accepted so that a script or a shortcut that passes it keeps
    /// working. It is read by nothing.
    package static let ignoredArgument = "--launch-mode"
}

import Foundation

// The Dock icon while a run goes with the window closed (docs/APP.md D3 = C): a bar with the run's progress, then a badge and one bounce
// when it ends. The numbers are pure; `DockTileController` in the app draws them.

package enum DockProgress {
    /// How far a run has got, 0...1; nil when it cannot be told (no target, or no step read yet). A step past the target is full, never more.
    package static func fraction(step: Int?, target: Int?, start: Int = 0) -> Double? {
        guard let step, let target, target > start else { return nil }
        return min(1, max(0, Double(step - start) / Double(target - start)))
    }

    package static func percentText(_ fraction: Double) -> String { "\(Int((fraction * 100).rounded(.down)))%" }

    /// The badge a run that ended leaves on the icon until the window is shown again: "Done" for a run that ended at a save point (it
    /// finished, or was stopped), "Failed" for one that did not.
    package static func endBadge(status: Int32, signalled: Bool) -> String {
        !signalled && (status == 0 || status == 130) ? "Done" : "Failed"
    }

    /// Whether a run's end is news for the person: they did not ask for it themselves. A Stop (or the quit's) is their own; the timer and
    /// every other end are not.
    package static func endIsNews(stopRequested: Bool, byTimer: Bool) -> Bool { !stopRequested || byTimer }
}

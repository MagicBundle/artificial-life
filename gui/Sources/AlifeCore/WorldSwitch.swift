import Foundation

// R5 of docs/APP.md: only one world is played at a time. Asking to play another world, or to make a planet, while a world is open puts one
// sheet in front of the person: "Stop A to play B?", with the time A needs to be saved and the reason. B starts after A has been saved
// (R2: a world is saved before anything stops it). The words and the estimate are pure; `WorldSwitcher` in the app drives the sheet.

/// What the person asked for that needs the open world to stop first.
package struct SwitchRequest: Equatable, Sendable {
    package enum Goal: Equatable, Sendable {
        /// Play another world of the library.
        case play(String)
        /// Make a planet with this name (it replaces the open world).
        case makePlanet(String)

        package var target: String {
            switch self {
            case .play(let w), .makePlanet(let w): return w
            }
        }
    }

    /// The world that is open now and is to be stopped.
    package var open: String
    package var goal: Goal
    /// How long saving `open` is expected to take, in seconds (`SwitchWording.estimateSeconds`).
    package var estimateSeconds: Int

    package init(open: String, goal: Goal, estimateSeconds: Int = SwitchWording.minimumSeconds) {
        self.open = open
        self.goal = goal
        self.estimateSeconds = estimateSeconds
    }

    private func quoted(_ name: String) -> String { "\u{2018}\(name)\u{2019}" }

    package var title: String {
        switch goal {
        case .play(let w): return "Stop \(quoted(open)) to play \(quoted(w))?"
        case .makePlanet(let w): return "Stop \(quoted(open)) to make the planet \(quoted(w))?"
        }
    }

    /// Why a world has to stop at all.
    package var reason: String {
        switch goal {
        case .play: return "Only one world can be played at a time."
        case .makePlanet: return "Only one world can be played at a time, and a new planet is played as soon as it is ready."
        }
    }

    /// What happens to the open world, with the estimate.
    package var detail: String {
        "\(quoted(open)) is saved first, which takes \(SwitchWording.estimateText(estimateSeconds)). Nothing is lost: you can play it "
            + "again from Worlds whenever you like."
    }

    /// The safe button: nothing changes, and it is the default.
    package var keepLabel: String { "Keep Playing" }

    package var confirmLabel: String {
        switch goal {
        case .play: return "Stop and Play"
        case .makePlanet: return "Stop and Make Planet"
        }
    }

    /// The sheet while the open world is saved.
    package var savingTitle: String { "Saving \(quoted(open))\u{2026}" }

    package func savingDetail(elapsed seconds: Int) -> String {
        let then: String
        switch goal {
        case .play(let w): then = "\(quoted(w)) opens as soon as it is saved."
        case .makePlanet(let w): then = "The planet \(quoted(w)) starts as soon as it is saved."
        }
        return "\(then) Waited \(max(0, seconds)) s; expected \(SwitchWording.estimateText(estimateSeconds))."
    }
}

package enum SwitchWording {
    package static let minimumSeconds = 5
    /// The longest a played world's stop can take: its ladder is 30 s, then 10 s (`EngineSession.Ladder.play`).
    package static let maximumSeconds = 40

    /// About how long a played world needs to stop and be saved, from the size of its newest save point. An estimate, said as one: the process
    /// stops (a couple of seconds), then the state is written. Measured on the M5 Pro (DESIGN §33.4.4, §34.11): a 686 MB save is written in
    /// 0.5 s on the internal disk and 5.8 s on an external drive; the compressed form (233 MB) takes about 11 s to compress. So 0.01 s per
    /// MB for a plain save and 0.05 s per MB for a compressed one, plus 3 s, rounded up to a multiple of 5 s, between 5 s and 40 s. With
    /// no save point to measure, the minimum.
    package static func estimateSeconds(newestSaveBytes bytes: Int64?, compressed: Bool) -> Int {
        guard let bytes, bytes > 0 else { return minimumSeconds }
        let megabytes = Double(bytes) / 1_000_000
        let raw = 3 + megabytes * (compressed ? 0.05 : 0.01)
        let rounded = Int((raw / 5).rounded(.up)) * 5
        return min(maximumSeconds, max(minimumSeconds, rounded))
    }

    package static func estimateText(_ seconds: Int) -> String { "about \(max(1, seconds)) seconds" }

    /// The size of the newest save point of `world` under `runs`, and whether it is the compressed form; nil with none.
    package static func newestSave(of world: String, in runs: URL, fileManager fm: FileManager = .default) -> (bytes: Int64, compressed: Bool)? {
        let directory = runs.appendingPathComponent(world).appendingPathComponent("checkpoints")
        guard let names = try? fm.contentsOfDirectory(atPath: directory.path) else { return nil }
        let saves = names.compactMap { name in StorageScan.saveStep(of: name).map { (step: $0, name: name) } }
        guard let newest = saves.max(by: { $0.step < $1.step }) else { return nil }
        let url = directory.appendingPathComponent(newest.name)
        guard let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize else { return nil }
        return (Int64(size), newest.name.hasSuffix(".zst"))
    }
}

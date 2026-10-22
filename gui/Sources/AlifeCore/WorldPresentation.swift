import Foundation

// Small pure rules of the World window's presentation (DESIGN §34.7): which pauses the player armed, when to ask for the
// player's attention, and what the window is called.

/// The pauses the player chose, kept by the window across worlds and sent back after each header (the world keeps no
/// preference file). An absent value is the world's own default.
package struct AutopausePrefs: Equatable, Sendable, Codable {
    /// The pause at a chapter's end; nil: the world's default.
    package var chapter: Bool?
    /// The pause of each alert class that was chosen either way.
    package var alert: [String: Bool] = [:]

    package init(chapter: Bool? = nil, alert: [String: Bool] = [:]) {
        self.chapter = chapter
        self.alert = alert
    }

    /// Whether the chapter pause is on, as the player has it (the world's default when not chosen).
    package func chapterOn(default world: PlayHeader.Autopause?) -> Bool { chapter ?? world?.chapter ?? true }

    package func alertOn(_ name: String, default world: PlayHeader.Autopause?) -> Bool {
        alert[name] ?? world?.alert[name] ?? false
    }

    /// The lines that bring a world to the player's choice: one for each choice that differs from the world's header. A class
    /// the world does not know is left out (it would only be noted on its stderr).
    package func commands(against world: PlayHeader.Autopause?, classes: Set<String>) -> [PlayCommand] {
        var out: [PlayCommand] = []
        if let chapter, chapter != (world?.chapter ?? true) { out.append(.autopause(.chapter(chapter))) }
        for (name, on) in alert.sorted(by: { $0.key < $1.key }) where classes.contains(name) && on != (world?.alert[name] ?? false) {
            out.append(.autopause(.alert(name, on)))
        }
        return out
    }

    // MARK: Storage (a JSON text in the user defaults)

    package var json: String {
        let data = (try? JSONEncoder().encode(self)) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }

    package init(json: String?) {
        if let json, let data = json.data(using: .utf8), let prefs = try? JSONDecoder().decode(AutopausePrefs.self, from: data) {
            self = prefs
        } else {
            self.init()
        }
    }
}

/// When to ask for the player's attention: a pause that arrives while the window is not the one in front is noticed once (the
/// dock icon bounces once), and the dock shows how many alerts came while the player was away. No notification is sent (no
/// permission prompt in G3b).
package struct AttentionTracker: Equatable, Sendable {
    /// Alerts of the stack raised while the window was not key, since it last was.
    package private(set) var unread = 0
    private var lastPausedStep: Int?

    package init() {}

    /// The dock badge: nil when nothing is unread.
    package var badge: String? { unread > 0 ? String(unread) : nil }

    /// `count` live alerts of the stack surface were raised.
    package mutating func raised(_ count: Int, windowIsKey: Bool) {
        guard !windowIsKey, count > 0 else { return }
        unread += count
    }

    /// The world stopped at `step` for a chapter or an alert. True once per stop, and only when the window is not key.
    package mutating func paused(atStep step: Int, windowIsKey: Bool) -> Bool {
        guard lastPausedStep != step else { return false }
        lastPausedStep = step
        return !windowIsKey
    }

    /// The window became key (or the app active): the player is looking.
    package mutating func windowBecameKey() { unread = 0 }

    package mutating func reset() {
        unread = 0
        lastPausedStep = nil
    }
}

/// What the dock and the window call the world. The run's directory name is a handle ("r", "p0"), not a title: a planet is
/// "Planet p0" with the run it was forked from under it.
package enum WorldTitle {
    package static func heading(world: String?, header: PlayHeader?) -> String {
        let name = (world ?? "").trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty else { return "No world" }
        return header?.planet.source != nil ? "Planet \(name)" : "World \(name)"
    }

    /// The line under the heading: where a planet came from.
    package static func subtitle(header: PlayHeader?) -> String? {
        header?.planet.source.map { "forked from \($0)" }
    }

    /// The window's title.
    package static func window(world: String?, header: PlayHeader?) -> String {
        world == nil ? "World" : heading(world: world, header: header)
    }
}

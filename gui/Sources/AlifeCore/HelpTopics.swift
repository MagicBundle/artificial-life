import Foundation

// The pages of Help (docs/APP.md §2.4): short, in the words of the app (§2.5), one topic per question a person has. The text is data so that the
// name audit reads it like every other page, and so that Help and the pages cannot say different things about the two kinds of world.

package struct HelpTopic: Equatable, Identifiable, Sendable {
    package var id: String
    package var title: String
    /// An SF Symbol.
    package var symbol: String
    package var paragraphs: [String]

    package init(id: String, title: String, symbol: String, paragraphs: [String]) {
        self.id = id
        self.title = title
        self.symbol = symbol
        self.paragraphs = paragraphs
    }
}

package enum HelpTopics {
    /// The topic that shows the key tables (drawn from `ControlTables`, the form in view first).
    package static let controlsID = "controls"

    package static let all: [HelpTopic] = [
        HelpTopic(id: "start", title: "Getting started", symbol: "sparkles", paragraphs: [
            "alife grows small worlds of simple animals that eat, move, mate and evolve. You make a world, watch it, and, if it is a planet, play it.",
            "The alife folder holds the engine (.venv/bin/alife) and your worlds (runs/). Choose it, and set up the engine, in Settings (Command-comma) or on the Worlds page.",
            "The quickest start is Worlds > Create a small planet: about two minutes of growing, then the planet opens in Watch.",
        ]),
        HelpTopic(id: "kinds", title: "Played and research worlds", symbol: "globe", paragraphs: [
            "Every world is one of two kinds, shown as a badge. A Played world is for playing: it runs live, and you can use powers, read its chronicle and its reports. A Research world grows by itself, and you watch it from outside: it has numbers, settings, an autopilot and an observer picture.",
            "The kind is a fact about the world. It is never a mode you switch. A research world becomes a played one by Make planet, which makes a new world from one of its save points. Nothing is changed in the world it came from.",
            "The shape is separate: Flat, 3D box or Planet. Only a planet can be made from a world that is not flat, because the weather runs on a planet.",
        ]),
        HelpTopic(id: "watch", title: "Watch", symbol: "eye", paragraphs: [
            "Watch shows one world. A played world is shown live, with the control center beside it. A research world is shown as an observer: the picture of a world that grows by itself, with the Run panel beside it (Numbers, Autopilot, Log).",
            "In an observer view you can pause, run and step, change the colours, show the food and chemical layers, turn the camera and save a picture. You can follow the world while it grows, or re-simulate it from its latest save point. You cannot use powers, read a chronicle or a report, or stop at a chapter: those belong to a played world. The toolbar says which one you are looking at: Live or Observer.",
            "Leaving Watch pauses a world you are playing, and it continues when you come back (Settings > While you are away). Leaving Watch ends the observer view of a research world; the world itself goes on growing.",
        ]),
        HelpTopic(id: "details", title: "Details", symbol: "info.circle", paragraphs: [
            "Details answers three questions about a world: what it is, how it is doing and what you can do with it. It shows the numbers and the autopilot's journal of any world, growing or not, the settings it was made with, where it came from and which worlds were branched from it.",
            "Open it by choosing a world in the sidebar, or with Details on a world's card. The engine's names (the folder, the keys) are in smaller type under the plain words.",
        ]),
        HelpTopic(id: "actions", title: "What the buttons do", symbol: "plus.circle", paragraphs: [
            "Create makes a new research world from a starting point. Branch copies a world from its newest save point and changes some settings. Keep growing continues a research world from where it stopped. Make planet makes a played world from a save point of a research world.",
            "Play opens a played world live. Check world asks the engine to verify a world. Close World saves the open world and closes it. Stop ends a growing run at a save point; Force stop ends it at once.",
        ]),
        HelpTopic(id: "autopilot", title: "Autopilot", symbol: "wand.and.stars", paragraphs: [
            "The autopilot steps in while a world grows: it rescues a population in trouble and stirs it with seasons, pulses and shocks, and it takes back what hurts. A style sets how it behaves: Gardener (gentle), Trickster (varied), Architect (works towards a goal you write) or Random control (random moves, for comparison).",
            "It can ask a language model that runs on your Mac. Its address and name are in Settings > Autopilot; each new world starts from them, and each world can change them under Advanced > Autopilot. Test connection checks that the model answers.",
            "Its journal, every move it made and why, is in the Run panel while a world grows and in Details afterwards.",
        ]),
        HelpTopic(id: controlsID, title: "World Controls", symbol: "keyboard", paragraphs: []),
    ]

    package static func topic(_ id: String) -> HelpTopic? { all.first { $0.id == id } }

    /// The heading above a form's table.
    package static func controlsHeading(_ form: WatchForm) -> String {
        form == .played ? "When you play a world" : "When you watch a research world"
    }

    /// Said above the two tables.
    package static let controlsIntro = "Watch has two sets of keys, one for each kind of world, because the same letter does different things. The World menu shows the set for the world in view; the keys work only on the Watch page."
}

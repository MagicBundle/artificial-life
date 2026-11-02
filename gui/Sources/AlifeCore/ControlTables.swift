import Foundation

// The two key tables of Watch, as the World menu and Help > World Controls show them (docs/APP.md §2.4, §3.4). They conflict on `f`, `s`, `o`
// and `r`, so they are never merged: each Watch form has its own table, the menu follows the form in view, and no key is listed in a table
// that does not act on it. Bare letters are never menu key equivalents (they would fire while a text field has the focus): a menu item names
// its key in its title, and a test reads every named key back from `WorldKeys` or `ViewerKeys`.

/// What a menu item or a key does, in either form.
package enum ControlAction: Equatable, Sendable {
    case world(WorldAction)
    case observer(ViewerAction)
}

package struct ControlEntry: Equatable, Sendable, Identifiable {
    /// The menu item's words and the help row's.
    package var title: String
    /// The characters the table maps (as `charactersIgnoringModifiers` gives them).
    package var keys: [String]
    /// The key as the player reads it: "Space", "H or R".
    package var hint: String
    /// What the menu item does; nil for a row that only the help lists (several actions in one row).
    package var action: ControlAction?
    /// The menu lists it (the rest are in the dock's View menu or in the help only).
    package var inMenu: Bool
    /// One more line for the help.
    package var detail: String?

    package var id: String { title }

    /// The menu item's title: the key is text in it, not an equivalent.
    package var menuTitle: String { "\(title)  (\(hint))" }
}

package enum ControlTables {
    private static func world(_ title: String, _ keys: [String], _ hint: String, _ action: WorldAction?, menu: Bool = false,
                              detail: String? = nil) -> ControlEntry {
        ControlEntry(title: title, keys: keys, hint: hint, action: action.map { .world($0) }, inMenu: menu, detail: detail)
    }

    private static func observer(_ title: String, _ keys: [String], _ hint: String, _ action: ViewerAction?, menu: Bool = false,
                                 detail: String? = nil) -> ControlEntry {
        ControlEntry(title: title, keys: keys, hint: hint, action: action.map { .observer($0) }, inMenu: menu, detail: detail)
    }

    /// The played world's keys (`WorldKeys`).
    package static let played: [ControlEntry] = [
        world("Pause or Run", [" "], "Space", .togglePause, menu: true),
        world("Step", ["."], ".", .step, menu: true, detail: "Runs one step, then pauses."),
        world("Slower", ["["], "[", .slower, menu: true, detail: "One rung down the rate ladder."),
        world("Faster", ["]"], "]", .faster, menu: true, detail: "One rung up the rate ladder."),
        world("Continue", ["g"], "G", .continueRun, menu: true, detail: "Runs again after a chapter or an alert, at the rate the world holds."),
        world("Overview", ["h", "r"], "H or R", .home, menu: true, detail: "Flies the camera to the whole world, north up."),
        world("Follow the selection", ["f"], "F", .followSelection, menu: true),
        world("North up", ["n"], "N", .northUp, menu: true),
        world("Powers", ["p"], "P", .showPowers, menu: true),
        world("Next alert", ["a"], "A", .flyToAlert, menu: true),
        world("Acknowledge the alert", ["k"], "K", .acknowledgeAlert, menu: true),
        world("Next colour", ["c"], "C", .cycleColor, menu: true),
        world("Agents, bonds, territory", ["1", "2", "3"], "1, 2, 3", nil, detail: "Show or hide each; territory goes off, borders, tint."),
        world("Built material, farmed soil", ["5", "6"], "5, 6", nil, detail: "In a world that has them."),
        world("Minimap", ["i"], "I", .toggleMinimap),
        world("Top, oblique, low, side view", ["t", "o", "l", "s"], "T, O, L, S", nil, detail: "In a 3D world; the camera flies there."),
        world("Back one level", ["\u{1B}"], "Esc", .escape, detail: "Cancels aiming a power, then drops its preview, then closes the chapter card, then clears the selection."),
        world("Confirm the previewed power", ["\r", "\u{3}"], "Return", .confirmPower),
    ]

    /// The observer's keys (`ViewerKeys`).
    package static let research: [ControlEntry] = [
        observer("Pause or Run", [" "], "Space", .togglePause, menu: true),
        observer("Step", ["."], ".", .step, menu: true, detail: "While following: fetches the newest save point once."),
        observer("Slower", ["-", "_"], "-", .slower, menu: true, detail: "Fewer steps per frame (re-simulating only)."),
        observer("Faster", ["+", "="], "+", .faster, menu: true, detail: "More steps per frame (re-simulating only)."),
        observer("Next colour", ["c"], "C", .cycleColor, menu: true),
        observer("Field slice", ["f", "4"], "F", .toggleFields, menu: true, detail: "The food and chemical colours of one layer."),
        observer("Field layer down", ["["], "[", .fieldLayerDown),
        observer("Field layer up", ["]"], "]", .fieldLayerUp),
        observer("Snapshot", ["s", "p"], "S", .snapshot, menu: true, detail: "Saves a picture under the world's media folder."),
        observer("Agents", ["1"], "1", .toggleAgents),
        observer("Bonds", ["2"], "2", .toggleBonds),
        observer("Frame", ["3"], "3", .toggleFrame, detail: "The box edges, the floor and the surface."),
        observer("Orbit or Fly", ["o"], "O", .toggleCameraMode, menu: true, detail: "Orbit turns about the target; Fly turns about the eye."),
        observer("Reset view", ["r"], "R", .resetCamera, menu: true),
    ]

    package static func entries(_ form: WatchForm) -> [ControlEntry] { form == .played ? played : research }

    /// The items of the World menu for a form.
    package static func menu(_ form: WatchForm) -> [ControlEntry] { entries(form).filter(\.inMenu) }

    /// Every character a form's table maps.
    package static func keys(_ form: WatchForm) -> Set<String> { Set(entries(form).flatMap(\.keys)) }

    /// What the mouse and the arrow keys do, one line per gesture, for the help.
    package static func pointer(_ form: WatchForm) -> [String] {
        switch form {
        case .played:
            return [
                "Drag moves the map. Two-finger scroll pans; pinch or a mouse wheel zooms (View > Scroll zooms swaps them for a Magic Mouse).",
                "Zoom out for the planet, in for the map. Option-drag tilts and turns. Click a cell, an organism or a storm to see it.",
            ]
        case .research:
            return [
                "Left-drag turns the view (Orbit) or looks around (Fly). Right-drag, middle-drag or Option-drag pans. The wheel or a pinch zooms.",
                "Arrow keys orbit and dolly, or fly; Shift is faster, Page Up and Page Down go up and down.",
            ]
        }
    }

    /// The hint on the map of a played world ("once on the first World window and again from View > Controls"). `overview` names the camera
    /// button as the `alife` window does (the two old apps say Home).
    package static func playedHint(overview: Bool, planet: Bool) -> String {
        let camera = overview ? "H overview" : "H home"
        if planet {
            return "Drag turns the planet. Two-finger scroll pans; pinch or a mouse wheel zooms (View > Scroll zooms swaps them); zoom out for the whole planet, in for the horizon. \(camera), N north up, I minimap, C colour, T O L S: top, oblique, low and side views; Option-drag tilts and turns. Click a cell, an organism or a storm to see it."
        }
        return "Drag moves the map. Two-finger scroll pans; pinch or a mouse wheel zooms (Magic Mouse: View > Scroll zooms); zoom out for the planet, zoom in for the map. \(camera)\(overview ? "" : " (the planet)"), N north up, I minimap, C colour. T O L S: top, oblique, low and side views; Option-drag tilts and turns."
    }
}

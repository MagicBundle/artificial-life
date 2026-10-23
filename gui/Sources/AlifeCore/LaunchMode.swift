import Foundation

/// The windows of the app by the identifier its scenes give them.
package enum AppWindow: String, CaseIterable, Sendable {
    /// The research Lab: the run controls, the log, the director panel. In the Lab app it is the main window.
    case lab
    /// The game: the World window with the launcher sheet.
    case world
    case viewer3d
}

/// Which app the executable is. One binary is bundled twice (`gui/scripts/make-app.sh`): as the research Lab
/// (`AlifeControl.app`, the default and what `swift run` gives) and as Storm Worlds, which opens straight into the game.
/// The bundle says which: `Info.plist` key `AlifeLaunchMode`, "world" or "lab". A missing key, an empty or an unknown
/// value is the Lab, so a bundle from before the key and an unbundled `swift run` behave as they always did.
package enum LaunchMode: String, CaseIterable, Sendable {
    case lab
    case world

    /// The `Info.plist` key.
    package static let infoKey = "AlifeLaunchMode"
    /// A command-line override for unbundled runs (`swift run AlifeControl --launch-mode world`); it wins over the bundle.
    package static let argumentName = "--launch-mode"

    /// The mode for an `Info.plist` value and the command-line arguments (after the executable).
    package static func resolve(info: Any?, arguments: [String]) -> LaunchMode {
        if let override = argumentValue(arguments), let mode = parse(override) { return mode }
        if let text = info as? String, let mode = parse(text) { return mode }
        return .lab
    }

    /// The mode of the running process: its bundle and its arguments.
    package static func current(bundle: Bundle = .main, arguments: [String] = CommandLine.arguments) -> LaunchMode {
        resolve(info: bundle.object(forInfoDictionaryKey: infoKey), arguments: arguments)
    }

    private static func parse(_ text: String) -> LaunchMode? {
        LaunchMode(rawValue: text.trimmingCharacters(in: .whitespaces).lowercased())
    }

    /// The value of the last `--launch-mode X` or `--launch-mode=X` (the last wins, as for `--world`).
    private static func argumentValue(_ arguments: [String]) -> String? {
        var value: String?
        var i = 0
        while i < arguments.count {
            let argument = arguments[i]
            if argument == argumentName, i + 1 < arguments.count {
                value = arguments[i + 1]
                i += 1
            } else if argument.hasPrefix(argumentName + "=") {
                value = String(argument.dropFirst(argumentName.count + 1))
            }
            i += 1
        }
        return value
    }

    package var plan: LaunchPlan { LaunchPlan(mode: self) }
}

/// A Window-menu item that opens a window; `key` is its Command key equivalent.
package struct WindowMenuItem: Equatable, Sendable, Identifiable {
    package let title: String
    package let window: AppWindow
    package let key: Character
    package var id: AppWindow { window }
}

/// What each mode does with its windows: the choice the two app structs are built from and the tests pin.
package struct LaunchPlan: Equatable, Sendable {
    package let mode: LaunchMode

    package init(mode: LaunchMode) { self.mode = mode }

    /// The name in the menu bar and the About box (the bundle's `CFBundleName` says the same).
    package var appName: String { mode == .world ? "Storm Worlds" : "Alife Control" }

    /// The windows open at launch. Storm Worlds opens only the World window (on its launcher); the Lab is one menu item away.
    /// `--world RUN` adds no window, it starts RUN in the one that opens (the Lab, whose first scene is its main window,
    /// opens the World window for it too).
    package func windowsAtLaunch(worldArgument: String? = nil) -> [AppWindow] {
        switch mode {
        case .lab: return worldArgument == nil ? [.lab] : [.lab, .world]
        case .world: return [.world]
        }
    }

    /// The first scene is the one SwiftUI opens at launch.
    package var firstScene: AppWindow { mode == .world ? .world : .lab }

    /// Whether the Lab is the main window: closing the last of them closes the others and ends the app. In Storm Worlds the
    /// Lab is a tool window: closing it closes the 3D viewer it opened and nothing else.
    package var labIsMainWindow: Bool { mode == .lab }

    /// The windows that go when the last Lab window closes.
    package var windowsClosedWithLab: [AppWindow] { labIsMainWindow ? [.viewer3d, .world] : [.viewer3d] }

    /// The app quits when its last window closes. For Storm Worlds this is the convention of a single-window game: closing
    /// the World window ends the game (a played world saves its checkpoint first), unless the Lab is still open. The
    /// Window menu brings either window back while the app runs.
    package var quitsAfterLastWindow: Bool { true }

    /// The Window-menu items that open a window, with their Command key equivalents. Only Storm Worlds has them: the Lab app
    /// opens its other windows from its own buttons, as it always did.
    package var windowMenu: [WindowMenuItem] {
        mode == .world ? [WindowMenuItem(title: "World", window: .world, key: "1"),
                          WindowMenuItem(title: "Lab", window: .lab, key: "2")] : []
    }
}

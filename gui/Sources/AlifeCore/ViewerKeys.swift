import Foundation

// The research Watch's key table (the observer of a research world), apart from the pieces that draw: the menu and the help read it
// (docs/APP.md §3.4). The played world's table is `WorldKeys`; the two are never merged (`f`, `s`, `o`, `r` mean different things).

/// What a key does. The keys follow the 2D viewer (DESIGN §20.2) where there is an equivalent.
package enum ViewerAction: Equatable, Sendable {
    case togglePause, step, faster, slower
    case cycleColor, toggleFields, fieldLayerDown, fieldLayerUp
    case snapshot
    case toggleAgents, toggleBonds, toggleFrame
    case toggleCameraMode, resetCamera
}

package enum ViewerKeys {
    /// The action of a key, from `NSEvent.charactersIgnoringModifiers`; nil for a key that does nothing.
    package static func action(for characters: String) -> ViewerAction? {
        switch characters.lowercased() {
        case " ": return .togglePause
        case ".": return .step
        case "+", "=": return .faster
        case "-", "_": return .slower
        case "c": return .cycleColor
        case "f", "4": return .toggleFields
        case "[": return .fieldLayerDown
        case "]": return .fieldLayerUp
        case "s", "p": return .snapshot
        case "1": return .toggleAgents
        case "2": return .toggleBonds
        case "3": return .toggleFrame
        case "o": return .toggleCameraMode
        case "r": return .resetCamera
        default: return nil
        }
    }
}

extension ViewerKeys {
    /// Every character that does something, for the help text and for the tests (the table above is a `switch`: this list is pinned to it).
    package static let characters: [String] = [" ", ".", "+", "=", "-", "_", "c", "f", "4", "[", "]", "s", "p", "1", "2", "3", "o", "r"]
}

import AlifeCore
import SwiftUI

/// Play, from wherever the button is (a card on the Worlds page, a world's Details page): the folder is looked at off the main thread, the reason is read
/// again at the moment of the click, another world that is open is stopped first by the sheet (R5), and the world that goes live is shown.
@MainActor
struct WorldOpener {
    let shell: ShellModel
    let library: WorldLibrary
    let launcher: WorldLauncherModel
    var switcher: WorldSwitcher?

    func play(_ name: String, facts: WorldsGuide.Facts) async {
        // The folder is looked at off the main thread with a deadline; an answer that does not come does not stop the play (the engine says).
        guard await library.checkThere(name) != .gone else { return library.noteGone(name) }
        guard let world = library.state.catalog.worlds.first(where: { $0.name == name }) else { return }
        // (The reason is read again at the moment of the click: another process may have taken the world since the card was drawn.)
        guard WorldsGuide.reason(forPlaying: world, in: facts) == nil else { return }
        let start: @MainActor () -> Void = {
            shell.playerStartsWorld()
            launcher.resume(world)
        }
        // Another world is open: the sheet stops it first (it is saved), then this one starts (R5).
        if let switcher, switcher.ask(.play(name), then: start) { return }
        start()
    }
}

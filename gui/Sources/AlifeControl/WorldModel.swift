import AlifeCore
import AppKit
import Foundation

/// What the World window does that is not the world's: the display options, the free rectangle the map is drawn into, the
/// launcher sheet and the last thing worth telling the user. The world's own commands go to the `PlayController`. (The
/// camera, the selection and the follow target are added with the renderer and the interaction: DESIGN §33.4.1.)
@MainActor
final class WorldModel: ObservableObject {
    let play: PlayController
    @Published var display = WorldDisplay()
    /// The launcher sheet.
    @Published var launcherOpen = false
    /// The last thing worth telling the user.
    @Published private(set) var note: String?
    /// Why nothing can be drawn (no Metal device, shaders that do not compile, a draw that failed); nil while it works.
    @Published private(set) var drawingFailure: String?
    /// The part of the window the map is drawn into (the window less the dock), in the window's coordinates: the camera's
    /// aspect and every pixel rule are its.
    private(set) var freeRect = CGRect.zero

    init(play: PlayController) {
        self.play = play
    }

    func failDrawing(_ message: String?) { drawingFailure = message }

    func tell(_ message: String?) { note = message }

    func setFreeRect(_ rect: CGRect) {
        guard rect != freeRect else { return }
        freeRect = rect
    }

    /// The view's width over its height (1 before there is one).
    var aspect: Float { DockLayout.aspect(of: freeRect) }
}

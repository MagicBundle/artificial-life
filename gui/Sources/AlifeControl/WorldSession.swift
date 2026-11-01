import AlifeCore
import Combine
import Foundation

/// The played world's model, owned by the app and not by a page (docs/APP.md §2.4 item 1). `PlayController` has single delivery
/// closures (`onDelivery`, `onEvents`, `onAlerts`, `onReports`, `onPlans`, `onReset`) that `WorldModel.init` assigns: a model that
/// does not exist when a header or a backlog arrives loses the toasts, the alerts and the reports. Here the model is made once, at
/// launch, so a world started by a click on Play or by `--world RUN` is delivered whether or not the Watch page has ever been
/// shown. (The old World window made its model in the view and needed `WorldWindowRoot`'s ordering for the same reason.)
@MainActor
final class WorldSession: ObservableObject {
    static let shared = WorldSession()

    let play: PlayController
    let model: WorldModel

    /// `defaults` is where the model keeps the display options (the app's own preferences; the tests give theirs).
    init(play: PlayController = .shared, defaults: UserDefaults = .standard) {
        self.play = play
        model = WorldModel(play: play, defaults: defaults)
        model.adoptGlobeDefault()  // (Overview is the globe until the player turns it off)
        model.setWatchActive(false)  // the window opens on Worlds
    }
}

import AlifeCore
import Foundation

/// The logic behind the launcher sheet: what is under `runs/`, which worlds can be resumed, and the planet being set up
/// (DESIGN §33.4.7). The sheet's view is `WorldLauncherView`; `launch` is how a request reaches the `PlayController`.
@MainActor
final class WorldLauncherModel: ObservableObject {
    @Published private(set) var catalog = PlayCatalog()
    /// The planet the form describes.
    @Published var planet = PlanetSpec(source: "", name: "")
    /// The directory names the app has asked to stop and whose process has not exited: they are saving.
    @Published private(set) var saving: Set<String> = []

    private let runsDirectory: () -> URL?
    private let launch: @MainActor (PlayRequest) -> Void

    /// `runsDirectory` is `<repository>/runs` (nil: no repository); `launch` starts a request.
    init(runsDirectory: @escaping () -> URL?, launch: @escaping @MainActor (PlayRequest) -> Void) {
        self.runsDirectory = runsDirectory
        self.launch = launch
    }

    /// Scans `runs/` again, and tells the model which worlds are still saving.
    func refresh(saving names: [String] = []) {
        saving = Set(names)
        guard let runs = runsDirectory() else {
            catalog = PlayCatalog()
            return
        }
        let scanned = PlayCatalogScanner.scan(runsDirectory: runs)
        if scanned != catalog { catalog = scanned }
        if planet.source.isEmpty || !catalog.sources.contains(where: { $0.name == planet.source }) {
            selectSource(catalog.sources.first?.name ?? "")
        } else if let size = selectedSource?.worldSize, size != planet.sourceWorldSize {
            // The selected source's config.json changed under a running launcher: follow its size (the overlay
            // is chosen from it), and its default spin-up if the user had not changed that.
            let oldDefault = PlanetSpec.defaultSpinup(forWorldSize: planet.sourceWorldSize)
            planet.sourceWorldSize = size
            if planet.storm, planet.weatherSpinup == oldDefault {
                planet.weatherSpinup = PlanetSpec.defaultSpinup(forWorldSize: size)
            }
        }
        if planet.name.trimmingCharacters(in: .whitespaces).isEmpty { suggestName() }
    }

    // MARK: Worlds

    /// Why `world` cannot be resumed now; nil when it can.
    func obstacle(to world: PlayWorld) -> String? {
        if saving.contains(world.name) { return "saving…" }
        if world.locked { return "being played" }
        if world.newestStep == nil { return "no checkpoint" }
        return nil
    }

    func resume(_ world: PlayWorld, rate: PlayRate? = .multiplier(1)) {
        guard obstacle(to: world) == nil else { return }
        launch(.resume(PlayResume(run: world.name, rate: rate)))
    }

    // MARK: Planets

    var selectedSource: PlaySource? { catalog.sources.first { $0.name == planet.source } }

    /// The line under the planet's source when that source is a played world (it has a `play.json`): continuing a world is what
    /// Resume is for, and a planet is a new branch from one of its saves (DESIGN 34.11).
    static let continueHint = "To continue this world, use Resume. Make planet starts a new branch from one of its saves."

    /// Whether the planet's source is a played world, so that the hint is shown.
    var sourceIsWorld: Bool { catalog.worlds.contains { $0.name == planet.source } }

    /// Picks the run a planet is made from: its newest checkpoint, and the weather overlay only if the source has none.
    func selectSource(_ name: String) {
        planet.source = name
        let source = catalog.sources.first { $0.name == name }
        planet.sourceStep = source?.checkpoints.last
        planet.sourceWorldSize = source?.worldSize
        applySourceDefaults(source)
    }

    func selectCheckpoint(_ step: Int?) { planet.sourceStep = step }

    private func applySourceDefaults(_ source: PlaySource?) {
        guard let source else { return }
        // A weather-on source is forked as it is: the engine refuses the spin-up for it (§33.2.1).
        planet.storm = !source.weatherOn
        planet.weatherSpinup = source.weatherOn ? 0 : PlanetSpec.defaultSpinup(forWorldSize: source.worldSize)
    }

    /// Switching the weather overlay keeps the spin-up consistent with it: none without the overlay, at least 1 with it.
    func setStorm(_ on: Bool) {
        planet.storm = on
        planet.weatherSpinup = on
            ? (planet.weatherSpinup > 0 ? planet.weatherSpinup : PlanetSpec.defaultSpinup(forWorldSize: planet.sourceWorldSize))
            : 0
    }

    func suggestName() {
        let taken = catalog.existingNames
        planet.name = taken.contains("planet_a") ? RunName.nextFree(after: "planet_a", taken: taken) : "planet_a"
    }

    var planetErrors: [String] { planet.validate(existing: catalog.existingNames) }

    /// Whether the exact houses can be rebuilt from the chosen source (it has event chunks).
    var housesNote: String? {
        guard let source = selectedSource else { return nil }
        if planet.houses == .exact, !source.hasEvents {
            return "'\(source.name)' has no event chunks: the exact houses cannot be rebuilt."
        }
        if planet.houses == .auto, !source.hasEvents {
            return "No event chunks: the houses will be the clades (dynasties approximate)."
        }
        return nil
    }

    func startPlanet() {
        guard planetErrors.isEmpty else { return }
        launch(.planet(planet))
    }
}

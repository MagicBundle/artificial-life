import AlifeCore
import SwiftUI

/// Everything the View menu shows: the one value its view observes (`DistinctState`), so that the menu is evaluated again when
/// one of these changes and not with every clock, storm record or toast of the world (a menu that is open is rebuilt, and flashes).
struct ViewMenuInputs: Equatable {
    struct Colour: Equatable {
        var mode: String
        var title: String
    }

    struct Pause: Equatable {
        var name: String
        var label: String
        var on: Bool
    }

    var live: Bool
    var colours: [Colour]
    /// The colour mode of the frame in view ("" before there is one).
    var current: String
    var labelsOn: Bool
    /// The level in view is the planet's (the 1 px points then have a toggle of their own). A step of the level, not a continuous
    /// camera value: the menu is not rebuilt while the player zooms (DESIGN §36.9).
    var planetLevel: Bool
    /// The layers and the look: the agents, bonds, territory (off, borders, tint), the map, the frame, the brightness and the
    /// organism size are all fields of `WorldDisplay`.
    var display: WorldDisplay
    var chapterPause: Bool
    var pauses: [Pause]
    var skipConfirm: Bool
    var debugFilters: Bool

    @MainActor
    init(play: PlayController, model: WorldModel) {
        let store = play.store
        live = play.phase == .live
        // The archetype and intent modes paint labels: they are tagged like every other presentation of one (DESIGN §33.7).
        let labels = store.play?.labels
        let validity = LabelValidity(labels: labels, environment: store.environment?.environment)
        colours = (store.header?.colorModes ?? []).map {
            Colour(mode: $0, title: validity.colorModeTitle($0, arch: labels?.arch ?? [], intent: labels?.intent ?? []))
        }
        current = store.frame.flatMap { store.header?.colorMode(index: $0.colorIndex) } ?? ""
        labelsOn = play.labelsOn
        planetLevel = model.level == .planet
        display = model.display
        chapterPause = store.clock?.chapterPauseArmed ?? true
        pauses = model.pausableClasses.map { Pause(name: $0.name, label: $0.label, on: model.alertPauseOn($0.name)) }
        skipConfirm = model.defaults.bool(forKey: WorldModel.skipConfirmKey)
        debugFilters = model.debugFilters
    }
}

/// The View menu: the colour mode, labels, and the layers. It reads `state` and nothing else that changes: `play` and `model` are
/// only written to (they are not observed), so a world that plays does not rebuild the menu while it is open.
struct ViewMenu: View {
    @ObservedObject var state: DistinctState<ViewMenuInputs>
    let play: PlayController
    let model: WorldModel

    var body: some View {
        let _ = ViewProbe.note("ViewMenu", for: play)
        let inputs = state.value
        Menu("View") {
            Picker("Colour", selection: Binding(get: { inputs.current }, set: { play.setColor($0) })) {
                ForEach(inputs.colours, id: \.mode) { Text($0.title).tag($0.mode) }
            }
            Divider()
            // The world's `labels on|off`: whether it sends the intent labels (INTN) at all. The record itself paints nothing; the
            // archetype and intent colour modes do (from the host's colours), and the picker above tags them.
            Toggle("Intent labels (stream)", isOn: Binding(get: { inputs.labelsOn }, set: { play.send(.labels($0)) }))
            Divider()
            Toggle("Agents", isOn: layer(\.agents))
            // The 1 px translucent points exist at the planet level only.
            if inputs.planetLevel { Toggle("Agent points", isOn: layer(\.agentPoints)) }
            Toggle("Bonds", isOn: layer(\.bonds))
            Picker("Territory", selection: Binding(get: { inputs.display.territory }, set: { model.display.territory = $0 })) {
                Text("Off").tag(TerritoryMode.off)
                Text("Borders").tag(TerritoryMode.borders)
                Text("Borders and tint").tag(TerritoryMode.tint)
            }
            Picker("Organism size", selection: Binding(get: { inputs.display.sizeMode }, set: { model.display.sizeMode = $0 })) {
                Text("Capped (symbols)").tag(OrganismSizeMode.capped)
                Text("Physical size").tag(OrganismSizeMode.physical)
            }
            Toggle("Weather map", isOn: layer(\.map))
            Picker("Brightness", selection: Binding(get: { inputs.display.exposure }, set: { model.display.exposure = $0 })) {
                ForEach(WorldDisplay.exposurePresets, id: \.self) { Text("\(Int(($0 * 100).rounded())) %").tag($0) }
            }
            Toggle("Frame", isOn: layer(\.frame))
            Divider()
            // The pauses the world takes by itself. The player's choice is kept and sent to every world after its header.
            Menu("Pauses") {
                Toggle("At the end of a chapter", isOn: Binding(get: { inputs.chapterPause }, set: { model.setChapterPause($0) }))
                if !inputs.pauses.isEmpty { Divider() }
                ForEach(inputs.pauses, id: \.name) { entry in
                    Toggle(entry.label, isOn: Binding(get: { entry.on }, set: { model.setAlertPause(entry.name, $0) }))
                }
            }
            Toggle("Skip the second click for a cost-1 power", isOn: Binding(
                get: { inputs.skipConfirm },
                set: {
                    model.defaults.set($0, forKey: WorldModel.skipConfirmKey)
                    state.refresh()  // (a default is not observable)
                }))
            Toggle("Raw kin events (developer)", isOn: Binding(get: { inputs.debugFilters }, set: { model.debugFilters = $0 }))
            Divider()
            Button("Reset view") { model.resetCamera() }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .disabled(!inputs.live)
    }

    private func layer(_ keyPath: WritableKeyPath<WorldDisplay, Bool>) -> Binding<Bool> {
        Binding(get: { state.value.display[keyPath: keyPath] }, set: { model.display[keyPath: keyPath] = $0 })
    }
}

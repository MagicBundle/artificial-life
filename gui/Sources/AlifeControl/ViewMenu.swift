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
    /// The layers and the look: the agents, bonds, territory (off, borders, tint), the map, the frame, the brightness, the
    /// organism size and the trackpad's scroll law are all fields of `WorldDisplay`. No camera-driven value is a field of the inputs
    /// (not even the level of the view): the menu is not rebuilt while the player zooms, drags or flies (DESIGN §36.9).
    var display: WorldDisplay
    /// A weather map is in view (`WorldModel.mapShown`): the brightness scales the map's colours, so it has nothing to do without one.
    var mapShown: Bool
    /// The world has a height to tilt over (3D): the standard views are enabled. Constant for a world, so the camera does not change it.
    var tiltable: Bool
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
        var modes = (store.header?.colorModes ?? []).map {
            Colour(mode: $0, title: validity.colorModeTitle($0, arch: labels?.arch ?? [], intent: labels?.intent ?? []))
        }
        // The height colour is computed here from the positions, not a mode of the world; it is listed last in a 3D world.
        if model.heightColourAvailable { modes.append(Colour(mode: HeightColour.modeName, title: HeightColour.title)) }
        colours = modes
        current = model.currentColourMode
        labelsOn = play.labelsOn
        display = model.display
        mapShown = model.mapShown
        tiltable = model.hasTiltableView
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
            Picker("Colour", selection: Binding(get: { inputs.current }, set: { model.setColourMode($0) })) {
                ForEach(inputs.colours, id: \.mode) { Text($0.title).tag($0.mode) }
            }
            Divider()
            // The world's `labels on|off`: whether it sends the intent labels (INTN) at all. The record itself paints nothing; the
            // archetype and intent colour modes do (from the host's colours), and the picker above tags them.
            Toggle("Intent labels (stream)", isOn: Binding(get: { inputs.labelsOn }, set: { play.send(.labels($0)) }))
            Divider()
            Toggle("Agents", isOn: layer(\.agents))
            // The 1 px translucent points are drawn at the planet level (zoomed out) only.
            Toggle("Agent points (zoomed out)", isOn: layer(\.agentPoints))
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
            // The brightness is the weather map's: without a map in view (a world with no MAP1, or the layer off) it would do nothing.
            Picker("Brightness", selection: Binding(get: { inputs.display.exposure }, set: { model.display.exposure = $0 })) {
                ForEach(WorldDisplay.exposurePresets, id: \.self) { Text("\(Int(($0 * 100).rounded())) %").tag($0) }
            }
            .disabled(!inputs.mapShown)
            .help(inputs.mapShown ? "How bright the weather map is" : "The brightness is the weather map's: there is none in view")
            Toggle("Frame", isOn: layer(\.frame))
            // The square map of the whole world with the view's footprint (the `i` key); only a world with a weather map has one.
            Toggle("Minimap (I)", isOn: layer(\.minimap))
                .disabled(!inputs.mapShown)
            // The planet: zooming out leads on from the flat map to a globe, Home is the globe (a world with a weather map and a height).
            Toggle("Globe overview", isOn: layer(\.globe))
                .disabled(!inputs.mapShown || !inputs.tiltable)
                .help("Zoom out past the whole map to a planet, and back; Home is the globe. Off: the flat map at every distance")
            // The trackpad's two-finger scroll pans (the Maps convention); on, it zooms as in v1. The mouse wheel and a pinch always zoom.
            // A Magic Mouse reports precise deltas like a trackpad, so its scroll pans too: this is how it zooms.
            Toggle("Scroll zooms", isOn: layer(\.scrollZooms))
                .help("The trackpad's two-finger scroll (and a Magic Mouse's) zooms instead of panning; a wheel mouse and a pinch always zoom")
            Divider()
            // The standard views of a 3D world (the keys T O L S, the strip over the map): animated, and static items, so that the menu
            // does not depend on where the camera is.
            Menu("View angle") {
                ForEach(ViewPreset.allCases, id: \.self) { preset in
                    Button(Self.title(of: preset)) { model.viewPreset(preset) }
                        .help(preset.help)
                }
                Divider()
                Button(Self.toggle3DTitle) { model.toggle3D() }
            }
            .disabled(!inputs.tiltable)
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
            Button("Home view (H)") { model.home() }
            Button("Controls…") { model.showHint() }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .disabled(!inputs.live)
    }

    /// The titles of the View angle submenu: a view each, with its key, and the 3D button's. (Static: the menu does not depend on where
    /// the camera is, so it is not rebuilt while it moves.)
    nonisolated static func title(of preset: ViewPreset) -> String { "\(preset.title) (\(preset.key.uppercased()))" }
    nonisolated static let toggle3DTitle = "Tilt on or off (3D button)"
    nonisolated static var viewAngleTitles: [String] { ViewPreset.allCases.map(title(of:)) + [toggle3DTitle] }

    private func layer(_ keyPath: WritableKeyPath<WorldDisplay, Bool>) -> Binding<Bool> {
        Binding(get: { state.value.display[keyPath: keyPath] }, set: { model.display[keyPath: keyPath] = $0 })
    }
}

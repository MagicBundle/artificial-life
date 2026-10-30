import AlifeCore
import SwiftUI

/// The sheet that lists the worlds under `runs/`, resumes one, and makes a planet from a checkpoint (DESIGN §33.4.5).
struct WorldLauncherView: View {
    @ObservedObject var model: WorldLauncherModel
    @ObservedObject var play: PlayController
    var close: () -> Void

    @State private var showUnfinished = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if play.phase == .starting {
                startingBox.padding(.horizontal, 16).padding(.top, 12)
            }
            worldsBox.padding(.horizontal, 16).padding(.top, 12)
            // What does not fit scrolls: the planet's form and the unfinished builds. The Resume list above and the
            // action below stay where they are.
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    LauncherPlanetBox(model: model)
                    if !model.catalog.unfinished.isEmpty { unfinishedBox }
                }
                .padding(16)
            }
            Divider()
            footer
        }
        .frame(minWidth: LauncherLayout.minWidth, idealWidth: LauncherLayout.minWidth,
               minHeight: LauncherLayout.minHeight, idealHeight: LauncherLayout.idealHeight,
               maxHeight: LauncherLayout.maxHeight)
        .onAppear { model.refresh(saving: play.retiredWorlds) }
        .onChange(of: play.retiredWorlds) { _, names in model.refresh(saving: names) }
        .onChange(of: play.phase) { _, phase in
            if phase == .live { close() }
        }
    }

    // MARK: Pieces

    private var header: some View {
        HStack {
            Text("Worlds").font(.title2)
            Spacer()
            Button("Done", action: close).keyboardShortcut(.cancelAction)
        }
        .padding(16)
    }

    private var startingBox: some View {
        GroupBox("Starting \(play.world ?? "")") {
            VStack(alignment: .leading, spacing: 6) {
                ScrollView {
                    Text(play.progress.suffix(40).joined(separator: "\n"))
                        .font(.caption.monospaced())
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(height: 80)
                HStack {
                    Text("A planet's fork and spin-up take a while; cancelling leaves an unfinished directory (nothing is deleted).")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Cancel") { play.stop() }
                }
            }
        }
    }

    private var worldsBox: some View {
        GroupBox("Resume") {
            Group {
                if model.catalog.worlds.isEmpty {
                    Text("No world yet: make a planet below.").foregroundStyle(.secondary)
                } else if LauncherLayout.resumeListScrolls(count: model.catalog.worlds.count) {
                    ScrollView { worldRows }
                        .frame(height: LauncherLayout.resumeListHeight(count: model.catalog.worlds.count))
                } else {
                    worldRows
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var worldRows: some View {
        VStack(alignment: .leading, spacing: LauncherLayout.resumeRowSpacing) {
            ForEach(model.catalog.worlds) { world in
                HStack {
                    VStack(alignment: .leading, spacing: 1) {
                        Text(world.name).font(.body)
                        Text(describe(world)).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if let why = model.obstacle(to: world) {
                        Text(why).font(.caption).foregroundStyle(.orange)
                    }
                    Button("Resume") { model.resume(world) }
                        .disabled(model.obstacle(to: world) != nil || play.isActive)
                }
                .frame(minHeight: LauncherLayout.resumeRowHeight)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Worlds that `alife play` began to build and did not finish; collapsed, because a long history of cancelled builds must
    /// not push the form around.
    private var unfinishedBox: some View {
        DisclosureGroup(isExpanded: $showUnfinished) {
            VStack(alignment: .leading, spacing: 2) {
                Text("A planet build that was cancelled or failed; not resumable, and nothing is deleted. Research runs are not listed here.")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(model.catalog.unfinished, id: \.self) { Text($0).font(.caption.monospaced()) }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 4)
        } label: {
            Text("Unfinished builds (\(model.catalog.unfinished.count))")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    /// The planet's action, outside the scroll area so that it is always in reach; Return triggers it while the form is
    /// valid (the default action of the sheet; the World window's own keys are not active under a sheet, §34.7).
    private var footer: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                ForEach(model.planetErrors, id: \.self) { Text($0).font(.caption).foregroundStyle(.red) }
            }
            Spacer()
            Button("Make planet and play") { model.startPlanet() }
                .keyboardShortcut(.defaultAction)
                .disabled(!model.planetErrors.isEmpty || play.isActive)
        }
        .padding(16)
    }

    private func describe(_ world: PlayWorld) -> String {
        var parts: [String] = []
        if world.geometry == "sphere" {
            parts.append("sphere" + (world.planetRadius.map { ", radius " + World3DLevers.text($0) } ?? ""))
        }
        if let step = world.newestStep { parts.append("step \(step)") }
        if let planet = world.file.planet {
            parts.append("planet of \(planet.source ?? "a preset")")
            parts.append("houses \(planet.houses)")
        }
        if let status = world.status { parts.append(status) }
        return parts.joined(separator: " · ")
    }
}

/// The *New planet* form of the launcher. A view of its own that reads the launcher model and not the play controller: the
/// controller publishes the stderr of the build while a planet starts, and the form's pop-ups and its name field must not be
/// evaluated again with each line (a pop-up that is open would be rebuilt under the pointer).
struct LauncherPlanetBox: View {
    @ObservedObject var model: WorldLauncherModel

    /// The source's name, with "sphere" and its radius for a planet (DESIGN §46.9: the launcher shows a world's geometry).
    static func title(of source: PlaySource) -> String {
        source.geometry == "sphere" ? "\(source.name)  (sphere\(source.planetRadius.map { ", radius " + World3DLevers.text($0) } ?? ""))" : source.name
    }

    /// What a sphere source means for the planet: its weather overlay is its own, and it stays a sphere.
    static func sphereNote(_ source: PlaySource) -> String {
        "A sphere world: the new planet is a sphere of the same shape (a fork cannot change its geometry), and the weather overlay is stormsph."
    }

    var body: some View {
        let _ = ViewProbe.note("LauncherPlanetBox", for: model)
        GroupBox("New planet") {
            VStack(alignment: .leading, spacing: 8) {
                Picker("Source", selection: Binding(
                    get: { model.planet.source }, set: { model.selectSource($0) })
                ) {
                    ForEach(model.catalog.sources) { Text(Self.title(of: $0)).tag($0.name) }
                }
                if let source = model.selectedSource, source.geometry == "sphere" {
                    Text(Self.sphereNote(source)).font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if model.sourceIsWorld {
                    Text(WorldLauncherModel.continueHint).font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let source = model.selectedSource {
                    Picker("Checkpoint", selection: Binding(
                        get: { model.planet.sourceStep ?? source.checkpoints.last ?? 0 },
                        set: { model.selectCheckpoint($0) })
                    ) {
                        ForEach(source.checkpoints, id: \.self) { Text("step \($0)").tag($0) }
                    }
                }
                Toggle("Turn the weather on (storm overlay)", isOn: Binding(
                    get: { model.planet.storm }, set: { model.setStorm($0) }))
                    .disabled(model.selectedSource?.weatherOn == true || model.weatherObstacle != nil)
                if let obstacle = model.weatherObstacle {
                    Text(obstacle).font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if model.planet.storm {
                    Stepper("Weather spin-up: \(model.planet.weatherSpinup) steps",
                            value: $model.planet.weatherSpinup, in: 1...200_000, step: 1000)
                }
                Toggle("Compile the director into the world (director.enabled)", isOn: $model.planet.directorEnabled)
                Picker("Houses", selection: $model.planet.houses) {
                    ForEach(PlayHouses.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                if let note = model.housesNote {
                    Text(note).font(.caption).foregroundStyle(.secondary)
                }
                TextField("Name", text: $model.planet.name).textFieldStyle(.roundedBorder)
            }
        }
    }
}

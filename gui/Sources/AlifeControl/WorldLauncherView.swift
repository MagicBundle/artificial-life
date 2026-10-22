import AlifeCore
import SwiftUI

/// The sheet that lists the worlds under `runs/`, resumes one, and makes a planet from a checkpoint (DESIGN §33.4.7).
struct WorldLauncherView: View {
    @ObservedObject var model: WorldLauncherModel
    @ObservedObject var play: PlayController
    var close: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Worlds").font(.title2)
                Spacer()
                Button("Done", action: close).keyboardShortcut(.cancelAction)
            }
            if play.phase == .starting {
                startingBox
            }
            worldsBox
            planetBox
        }
        .padding(16)
        .frame(minWidth: 640, minHeight: 560)
        .onAppear { model.refresh(saving: play.retiredWorlds) }
        .onChange(of: play.retiredWorlds) { _, names in model.refresh(saving: names) }
        .onChange(of: play.phase) { _, phase in
            if phase == .live { close() }
        }
    }

    // MARK: Pieces

    private var startingBox: some View {
        GroupBox("Starting \(play.world ?? "")") {
            VStack(alignment: .leading, spacing: 6) {
                ScrollView {
                    Text(play.progress.suffix(40).joined(separator: "\n"))
                        .font(.caption.monospaced())
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(height: 100)
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
            VStack(alignment: .leading, spacing: 6) {
                if model.catalog.worlds.isEmpty {
                    Text("No world yet: make a planet below.").foregroundStyle(.secondary)
                }
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
                }
                if !model.catalog.unfinished.isEmpty {
                    Divider()
                    Text("Unfinished (a planet build that was cancelled or failed; not resumable):")
                        .font(.caption).foregroundStyle(.secondary)
                    ForEach(model.catalog.unfinished, id: \.self) { Text($0).font(.caption.monospaced()) }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var planetBox: some View {
        GroupBox("New planet") {
            VStack(alignment: .leading, spacing: 8) {
                Picker("Source", selection: Binding(
                    get: { model.planet.source }, set: { model.selectSource($0) })
                ) {
                    ForEach(model.catalog.sources) { Text($0.name).tag($0.name) }
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
                    .disabled(model.selectedSource?.weatherOn == true)
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
                ForEach(model.planetErrors, id: \.self) { Text($0).font(.caption).foregroundStyle(.red) }
                HStack {
                    Spacer()
                    Button("Make planet and play") { model.startPlanet() }
                        .disabled(!model.planetErrors.isEmpty || play.isActive)
                }
            }
        }
    }

    private func describe(_ world: PlayWorld) -> String {
        var parts: [String] = []
        if let step = world.newestStep { parts.append("step \(step)") }
        if let planet = world.file.planet {
            parts.append("planet of \(planet.source ?? "a preset")")
            parts.append("houses \(planet.houses)")
        }
        if let status = world.status { parts.append(status) }
        return parts.joined(separator: " · ")
    }
}

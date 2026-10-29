import AlifeCore
import SwiftUI

/// The lever groups of the five feature layers (arms, structures, bonds and roles, comm, niche; `FeatureLevers`): one box per feature
/// with its switch, an overlay toggle for a new run, and one field per key of its table. Everything is read from the tables, so a new
/// key in `<Feature>Levers.swift` shows up here without a change.
struct FeatureLeverSections: View {
    @ObservedObject var controller: RunController

    var body: some View {
        ForEach(FeatureLevers.groups, id: \.feature) { group in
            FeatureLeverBox(controller: controller, group: group)
        }
    }
}

private struct FeatureLeverBox: View {
    @ObservedObject var controller: RunController
    let group: FeatureLeverGroup

    private var settings: RunSettings { controller.settings }
    private var isFork: Bool { settings.mode == .fork }
    private var features: FeatureLevers { settings.levers.features }

    private func binding(_ key: String) -> Binding<String> {
        Binding(get: { controller.settings.levers.features[key] },
                set: { controller.settings.levers.features[key] = $0 })
    }

    private func flag(_ key: String) -> Binding<Bool> {
        Binding(get: { controller.settings.levers.features.values.isOn(key) },
                set: { controller.settings.levers.features[key] = $0 ? "true" : "false" })
    }

    private var overlayBinding: Binding<Bool> {
        Binding(get: { controller.settings.featureOverlays.contains(group.feature) },
                set: { controller.setFeatureOverlay(group.feature, on: $0) })
    }

    var body: some View {
        GroupBox(group.title) {
            VStack(alignment: .leading, spacing: 8) {
                Text(group.summary).font(.caption).foregroundStyle(.secondary)
                if settings.mode == .new, let overlay = group.overlayName(dims: settings.worldDims) {
                    Toggle("Start from the overlay (--overlay \(overlay))", isOn: overlayBinding)
                        .help("Passes the overlay and pre-fills the levers below with its values; only levers you change from them are passed.")
                }
                if isFork {
                    Text("Pre-filled from the source run. The switches and the layout keys are fixed in a fork (`alife migrate` changes a layout).")
                        .font(.caption).foregroundStyle(.secondary)
                }
                ForEach(group.specs) { spec in
                    row(spec)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .disabled(settings.mode == .resume)
        }
        .disabled(controller.job != nil)
    }

    @ViewBuilder private func row(_ spec: FeatureLeverSpec) -> some View {
        let live = features.isLive(spec)
        let locked = isFork && spec.layout
        VStack(alignment: .leading, spacing: 2) {
            switch spec.kind {
            case .flag:
                Toggle(spec.title, isOn: flag(spec.key)).disabled(!live || locked)
            case .options(let options):
                HStack {
                    Text(spec.title).frame(width: 140, alignment: .leading)
                    Picker("", selection: binding(spec.key)) {
                        ForEach(options, id: \.value) { Text($0.label).tag($0.value) }
                    }
                    .labelsHidden()
                    .disabled(!live)
                }
            case .decimal, .integer:
                HStack {
                    Text(spec.title).frame(width: 140, alignment: .leading)
                    TextField(spec.key, text: binding(spec.key)).textFieldStyle(.roundedBorder).disabled(!live || locked)
                }
            }
            Text(spec.help).font(.caption).foregroundStyle(.secondary)
        }
        .opacity(live ? 1 : 0.5)
    }
}

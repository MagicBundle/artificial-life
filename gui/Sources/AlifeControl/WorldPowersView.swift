import AlifeCore
import SwiftUI

// The Powers tab (DESIGN §34.7): the units, the grid, the aim-preview-confirm exchange, the effects that run and the cards of
// what happened. The state machine is `PowersModel`; this lays it out.

struct PowersTabView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel

    var body: some View {
        let _ = play.revision
        let store = play.store
        let offers = model.offers
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                if offers.isEmpty {
                    Text(store.play?.records.contains("PLAN") == true
                         ? "This world takes no powers: it was not made with a director."
                         : "This world takes no powers.")
                        .font(.callout).foregroundStyle(.secondary)
                } else {
                    InfluenceMeterView(influence: store.influence)
                    grid(offers, step: model.currentStep)
                    ForEach(PowerCatalog.reserved.indices, id: \.self) { i in
                        HStack {
                            Text(PowerCatalog.reserved[i].label).font(.callout)
                            Spacer()
                            Text(PowerCatalog.reserved[i].note).font(.caption)
                        }
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 8)
                    }
                    if let offer = model.powers.offer {
                        Divider()
                        PowerEditorView(offer: offer, model: model, baseRate: store.play?.baseRate ?? 14.4)
                    }
                    effects(store)
                    cards(store)
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func grid(_ offers: [PowerOffer], step: Int) -> some View {
        LazyVGrid(columns: [GridItem(.flexible(), spacing: 8), GridItem(.flexible(), spacing: 8)], spacing: 8) {
            ForEach(offers) { offer in
                PowerCell(offer: offer, selected: model.powers.selected == offer.name, step: step) {
                    model.choosePower(offer)
                }
            }
        }
    }

    @ViewBuilder
    private func effects(_ store: WorldStore) -> some View {
        let running = ActiveEffect.running(store)
        if !running.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                Text("Running").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                ForEach(running) { effect in
                    HStack {
                        Text("\(effect.label) until step \(Numbers.grouped(effect.end))").font(.callout)
                        Spacer()
                        Button("End early") { model.cancelPower(effect.name) }
                            .controlSize(.small)
                            .help("Stops it now. The unit is not given back, and the power's cooldown runs from the planned end.")
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func cards(_ store: WorldStore) -> some View {
        let cards = PowerCardModel.cards(store.plans, atStep: model.currentStep)
        if !cards.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("What happened").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                ForEach(cards) { card in PowerCardView(card: card) }
            }
        }
    }
}

/// The units of influence as pips: the free ones filled, each spent one with the step it comes back.
struct InfluenceMeterView: View {
    let influence: PlayEnvironment.Influence?

    var body: some View {
        if let influence {
            let pips = InfluenceMeter.pips(influence)
            HStack(spacing: 6) {
                ForEach(pips.indices, id: \.self) { i in
                    Circle()
                        .fill(pips[i].filled ? Color.accentColor : Color.clear)
                        .overlay(Circle().stroke(Color.accentColor.opacity(0.7), lineWidth: 1.5))
                        .frame(width: 12, height: 12)
                        .help(pips[i].filled ? "A unit you can spend"
                              : pips[i].freeAt.map { "Back at step \(Numbers.grouped($0))" } ?? "Spent")
                }
                Spacer()
                Text("\(influence.have) of \(influence.cap) units").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
        }
    }
}

/// One power of the grid: its name and cost, why it cannot be used when it cannot, a ring for the cooldown.
struct PowerCell: View {
    let offer: PowerOffer
    let selected: Bool
    let step: Int
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(offer.label).font(.callout.weight(.semibold))
                    Spacer()
                    ZStack {
                        Circle().stroke(Color.primary.opacity(0.15), lineWidth: 2)
                        Circle().trim(from: 0, to: offer.cooldownFraction(atStep: step))
                            .stroke(Color.orange, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                            .rotationEffect(.degrees(-90))
                    }
                    .frame(width: 14, height: 14)
                    .help(offer.readyAt.map { "Ready at step \(Numbers.grouped($0))" } ?? "Ready")
                }
                Text(offer.busy ?? offer.blurb).font(.caption2).foregroundStyle(offer.usable ? Color.secondary : Color.orange)
                    .lineLimit(3).fixedSize(horizontal: false, vertical: true)
                Text("cost \(offer.cost)").font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            }
            .padding(8)
            .frame(maxWidth: .infinity, minHeight: 72, alignment: .topLeading)
            .background(selected ? Color.accentColor.opacity(0.18) : Color.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(selected ? Color.accentColor : Color.primary.opacity(0.12)))
        }
        .buttonStyle(.plain)
        .disabled(!offer.usable)
        .opacity(offer.usable ? 1 : 0.65)
    }
}

/// The sliders of the chosen power, what the map is waiting for, the preview, and Confirm.
struct PowerEditorView: View {
    let offer: PowerOffer
    @ObservedObject var model: WorldModel
    let baseRate: Double

    var body: some View {
        let powers = model.powers
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(offer.label).font(.headline)
                Spacer()
                Button { _ = model.powers.escape() } label: { Image(systemName: "xmark.circle") }
                    .buttonStyle(.borderless).help("Cancel (esc)")
            }
            ForEach(offer.parameters, id: \.key) { parameter in
                ParameterSlider(
                    label: parameter.label, value: powers.values[parameter.key] ?? parameter.initial, low: parameter.lowest,
                    high: parameter.highest, step: parameter.increment, text: { Self.format(parameter.key, $0) }) {
                        model.setPowerValue(parameter.key, $0)
                    }
            }
            if let steps = offer.steps {
                ParameterSlider(
                    label: "Duration", value: powers.values[.steps] ?? Double(steps.lowerBound), low: Double(steps.lowerBound),
                    high: Double(steps.upperBound), step: 250, text: { Self.duration($0, baseRate: baseRate) }) {
                        model.setPowerValue(.steps, $0)
                    }
            }
            status(powers)
        }
    }

    @ViewBuilder
    private func status(_ powers: PowersModel) -> some View {
        switch powers.phase {
        case .idle:
            EmptyView()
        case .aiming:
            Label(offer.scope == .patch ? "Click the map near the food patch to aim at it." : "Click the map to choose the column.",
                  systemImage: "scope")
                .font(.callout).foregroundStyle(.secondary)
        case .previewing:
            HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Asking the world…").font(.callout).foregroundStyle(.secondary) }
        case .previewed:
            if let plan = powers.preview {
                let card = PowerPreviewCard(plan)
                VStack(alignment: .leading, spacing: 4) {
                    Text(card.text).font(.callout).fixedSize(horizontal: false, vertical: true)
                    ForEach(card.cautions, id: \.self) { caution in
                        Label(caution, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    if !card.clamped.isEmpty {
                        Text("Adjusted to the allowed range: \(card.clamped.joined(separator: ", ")).")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if let precedent = card.precedent {
                        Text(precedent).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                    HStack {
                        Button("Confirm") { model.confirmPower() }
                            .help("Send it (return)")
                        Button("Cancel") { _ = model.powers.escape() }.help("Drop the preview (esc)")
                    }
                    .padding(.top, 2)
                }
            }
        case .refused(_, let reason):
            Label(reason, systemImage: "xmark.octagon").font(.callout).foregroundStyle(.red)
                .fixedSize(horizontal: false, vertical: true)
        case .submitted:
            HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Sent. Waiting for the world…").font(.callout).foregroundStyle(.secondary) }
        }
    }

    static func format(_ key: PowerCommand.Key, _ value: Double) -> String {
        switch key {
        case .mult, .gain: return "x" + PlayCommand.decimal(value)
        case .kill: return "\(Int((value * 100).rounded()))%"
        case .radius: return PlayCommand.decimal(value) + " units"
        default: return PlayCommand.decimal(value)
        }
    }

    /// "6,000 steps (6.9 min at 1x)".
    static func duration(_ steps: Double, baseRate: Double) -> String {
        let minutes = baseRate > 0 ? steps / (60 * baseRate) : 0
        return "\(Numbers.grouped(Int(steps))) steps (" + String(format: "%.1f", minutes) + " min at 1x)"
    }
}

/// A slider that sends its value when it is let go (a drag is not a stream of previews).
struct ParameterSlider: View {
    let label: String
    let value: Double
    let low: Double
    let high: Double
    let step: Double
    let text: (Double) -> String
    let commit: (Double) -> Void
    @State private var draft: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(label).font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text(text(draft ?? value)).font(.caption.monospacedDigit())
            }
            Slider(value: Binding(get: { draft ?? value }, set: { draft = $0 }), in: low...max(high, low + step), step: step) { editing in
                if !editing, let draft {
                    commit(draft)
                    self.draft = nil
                }
            }
            .controlSize(.small)
        }
    }
}

/// What happened to one request: what was asked, what the world did, the rows since.
struct PowerCardView: View {
    let card: PowerCardModel

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(card.text).font(.callout).fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
                Text(card.status.rawValue).font(.caption2).padding(.horizontal, 5).padding(.vertical, 1)
                    .background(color.opacity(0.2), in: Capsule())
            }
            if let refusal = card.refusal { Text(refusal).font(.caption).foregroundStyle(.red) }
            if let applied = card.stepApplied {
                Text("applied at step \(Numbers.grouped(applied))" + (card.ends.map { ", ends at \(Numbers.grouped($0))" } ?? ""))
                    .font(.caption2).foregroundStyle(.secondary)
            }
            if let disc = card.disc { Text(disc).font(.caption).foregroundStyle(.secondary) }
            ForEach(card.rows.indices, id: \.self) { i in
                Text(card.rows[i]).font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
                    .lineLimit(1).minimumScaleFactor(0.7)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 6))
    }

    private var color: Color {
        switch card.status {
        case .refused: return .red
        case .queued: return .blue
        case .applied: return .green
        case .cancelled, .ended: return .gray
        }
    }
}

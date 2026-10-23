import AlifeCore
import SwiftUI

// The chapter card over the map and the Report tab (DESIGN §34.7). The sentences are the server's; a click on one flies to
// the entity it names.

/// A sentence of a report: a click flies to the entity it names, when it names one the report can place.
struct ReportLineView: View {
    let line: ReportCard.Line
    var font: Font = .callout
    var weight: Font.Weight = .regular
    @ObservedObject var model: WorldModel

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            Text(line.text).font(font.weight(weight)).fixedSize(horizontal: false, vertical: true)
            if line.action != nil {
                Image(systemName: "location").font(.caption2).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .contentShape(Rectangle())
        .onTapGesture { model.activate(line.action) }
        .help(line.action == nil ? "" : "Fly to it")
    }
}

/// The units, the powers, the effects that end and the alerts of the chapter, in a few lines.
struct SinceStripView: View {
    let strip: SinceStrip
    @ObservedObject var model: WorldModel

    var body: some View {
        if !strip.isEmpty {
            VStack(alignment: .leading, spacing: 3) {
                if let have = strip.haveUnits, let cap = strip.capUnits {
                    row("Units", "\(have) of \(cap)" + (strip.nextUnit.map { ", the next back at step \(Numbers.grouped($0))" } ?? ""))
                }
                if !strip.ready.isEmpty { row("Ready", strip.ready.joined(separator: ", ")) }
                if !strip.ending.isEmpty {
                    row("Ending", strip.ending.map { "\($0.label) at step \(Numbers.grouped($0.end))" }.joined(separator: "; "))
                }
                if !strip.alertCounts.isEmpty {
                    row("Alerts", strip.alertCounts.map { "\($0.label) \($0.count)" }.joined(separator: ", "))
                }
                ForEach(strip.top) { alert in
                    HStack(spacing: 4) {
                        Circle().fill(alert.tone.color).frame(width: 6, height: 6)
                        Text(alert.title + ": " + alert.text).font(.caption).lineLimit(2)
                        Spacer(minLength: 0)
                        if alert.action != nil {
                            Button { model.activate(alert.action) } label: { Image(systemName: "location.fill") }
                                .buttonStyle(.borderless).controlSize(.small).help("Fly to it")
                        }
                    }
                }
            }
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(label).font(.caption).foregroundStyle(.secondary).frame(width: 52, alignment: .leading)
            Text(value).font(.caption).fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }
}

/// The words of one report: headline, lead, body, the interventions last.
struct ReportContentView: View {
    let card: ReportCard
    @ObservedObject var model: WorldModel
    var showBody = true

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let withheld = card.withheld {
                Text("No report for this chapter: \(withheld).").font(.callout).foregroundStyle(.secondary)
            } else {
                if let headline = card.headline { ReportLineView(line: headline, font: .headline, model: model) }
                if let lead = card.lead { ReportLineView(line: lead, model: model) }
                if showBody {
                    ForEach(card.body) { ReportLineView(line: $0, font: .callout, model: model) }
                    if card.hiddenBody > 0 {
                        Text("\(card.hiddenBody) more in the Report tab").font(.caption).foregroundStyle(.secondary)
                    }
                    if !card.interventions.isEmpty {
                        Text("Interventions").font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.top, 2)
                        ForEach(card.interventions) { ReportLineView(line: $0, font: .callout, model: model) }
                    }
                }
                if card.ordering { Text("ordering…").font(.caption2).foregroundStyle(.secondary) }
            }
        }
    }
}

/// The card over the map when a chapter ends (reason 3 of the clock): the report, the strip, and the way on. With the pause off
/// it is a notice that goes by itself. `Esc` folds it to a bar; nothing here resumes the world but Continue.
struct ChapterCardView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel

    var body: some View {
        let _ = play.revision
        if let card = model.chapterCard {
            Group {
                if card.form == .notice {
                    notice(card)
                } else if model.cardFolded {
                    folded(card)
                } else {
                    full(card)
                }
            }
            .padding(.top, 12)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
    }

    /// Below this height of the map the card keeps headline and lead only (the body is in the Report tab).
    static let compactBelow: CGFloat = 520

    private func full(_ card: ChapterCard) -> some View {
        let compact = model.freeRect.height < Self.compactBelow
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(card.title).font(.title3.weight(.semibold))
                if card.saving { Label("saving…", systemImage: "externaldrive").font(.caption).foregroundStyle(.orange) }
                Spacer()
                Button { model.cardCollapsed = true } label: { Image(systemName: "chevron.up") }
                    .buttonStyle(.borderless).help("Fold to a bar and see the map (esc)")
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    if let report = card.report {
                        ReportContentView(card: report, model: model, showBody: !compact && !report.quiet)
                    } else if let previous = card.previousHeadline {
                        Text("Building this chapter's report…").font(.callout).foregroundStyle(.secondary)
                        Text(previous).font(.callout).foregroundStyle(.tertiary)
                    } else {
                        Text("Building this chapter's report…").font(.callout).foregroundStyle(.secondary)
                    }
                    if card.report?.quiet != true {
                        Divider()
                        SinceStripView(strip: card.since, model: model)
                    }
                }
            }
            .frame(maxHeight: max(model.freeRect.height - 190, 120))
            if card.form == .ready { buttons }
        }
        .padding(14)
        .frame(width: 520)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.primary.opacity(0.15)))
    }

    private var buttons: some View {
        HStack(spacing: 8) {
            Button("Continue") {
                play.resume()
                model.dismissNotice()
            }
            .help("Run at the rate the world holds (g, space)")
            Button("Continue at 2x") { play.setRate(.multiplier(2)) }
            Button("Powers") { model.tab = .powers }
                .help("Open the Powers tab (p)")
            Button("Show on map") { model.cardCollapsed = true }
                .help("Fold the card to a bar (esc)")
            Spacer()
        }
        .controlSize(.regular)
    }

    private func folded(_ card: ChapterCard) -> some View {
        HStack(spacing: 10) {
            Text(card.title).font(.callout.weight(.semibold))
            if let headline = card.report?.headline?.text { Text(headline).font(.callout).lineLimit(1).foregroundStyle(.secondary) }
            Spacer(minLength: 0)
            if card.form == .ready, !model.powers.isAiming {
                Button("Continue") { play.resume() }.controlSize(.small)
            }
            Button { model.cardCollapsed = false } label: { Image(systemName: "chevron.down") }
                .buttonStyle(.borderless)
                .disabled(model.powers.isAiming)
                .help(model.powers.isAiming ? "Finish or cancel the power first (esc)" : "Show the whole card")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .frame(width: 520)
        .background(.regularMaterial, in: Capsule())
        .overlay(Capsule().stroke(Color.primary.opacity(0.15)))
    }

    private func notice(_ card: ChapterCard) -> some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(card.title).font(.callout.weight(.semibold))
                if let headline = card.report?.headline?.text { Text(headline).font(.caption).lineLimit(2) }
            }
            Spacer(minLength: 0)
            Button("Report") {
                model.tab = .report
                model.dismissNotice()
            }
            .controlSize(.small)
            Button { model.dismissNotice() } label: { Image(systemName: "xmark") }
                .buttonStyle(.borderless)
                .help("Close")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(width: 440)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.primary.opacity(0.15)))
    }
}

/// The Report tab: the newest chapter in full, the earlier ones as headlines that open.
struct ReportTabView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel

    var body: some View {
        let _ = play.revision
        let tab = ReportTabModel(shelf: play.store.reports)
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                if let newest = tab.newest {
                    section(newest, expanded: true)
                    if !tab.earlier.isEmpty {
                        Text("Earlier chapters").font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.top, 4)
                        ForEach(tab.earlier, id: \.nAbs) { card in
                            DisclosureGroup {
                                section(card, expanded: false)
                            } label: {
                                Text("\(Self.title(card))  " + (card.headline?.text ?? card.withheld.map { "no report: \($0)" } ?? ""))
                                    .font(.callout).lineLimit(2)
                            }
                        }
                    }
                } else {
                    Text(play.store.play?.records.contains("RPRT") == true
                         ? "The first report arrives at the end of the chapter." : "This world sends no reports.")
                        .font(.callout).foregroundStyle(.secondary)
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private static func title(_ card: ReportCard) -> String {
        "Chapter \(card.chapter)"
    }

    private func section(_ card: ReportCard, expanded: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            if expanded {
                Text("\(Self.title(card)) · steps \(Numbers.grouped(card.a)) to \(Numbers.grouped(card.b))")
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            }
            ReportContentView(card: card, model: model)
            if expanded, card.withheld == nil {
                Divider()
                SinceStripView(
                    strip: SinceStrip.make(
                        store: play.store, range: (card.a + 1)..<(card.b + 1), reported: card.since, horizon: max(card.b - card.a, 1),
                        watch: model.watch),
                    model: model)
            }
            if !card.ticker.isEmpty {
                Text("Ticker").font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.top, 2)
                ForEach(card.ticker.indices, id: \.self) { i in
                    Text("\(Numbers.grouped(card.ticker[i].step))  \(card.ticker[i].text)")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

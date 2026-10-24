import AlifeCore
import SwiftUI

// The chapter bar, the alert stack and the ticker (DESIGN §34.7). Each view lays out what a pure type of AlifeCore decided
// (`ChapterBar`, `AlertStack`, `Ticker`) and sends clicks to the `WorldModel`; none writes a sentence of its own.

extension AlertTone {
    var color: Color {
        switch self {
        case .quiet: return .gray
        case .notice: return .blue
        case .important: return .orange
        case .urgent: return .red
        }
    }
}

/// "Chapter 3", how far it is, when the report is due, and the lock that says whether the world stops for it.
struct ChapterBarView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel

    var body: some View {
        let _ = play.revision
        let bar = model.chapterBar
        let ticks = bar.ticks(events: play.store.chronicle.events, alerts: play.store.alerts.alerts)
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Text(bar.label).font(.callout.weight(.semibold))
                Spacer()
                if let next = bar.nextReport {
                    Text(next).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Button {
                    model.toggleChapterPause()
                } label: {
                    Image(systemName: bar.pauseArmed ? "lock.fill" : "lock.open")
                        .foregroundStyle(bar.pauseArmed ? Color.primary : Color.secondary)
                }
                .buttonStyle(.borderless)
                .disabled(play.phase != .live)
                .help(bar.pauseArmed ? "The world stops at the end of the chapter (click to let it run on)"
                      : "The world runs on at the end of the chapter (click to stop there)")
                if bar.alertPauseArmed {
                    Image(systemName: "bell.fill").font(.caption).foregroundStyle(.secondary)
                        .help("An alert can stop the world (View, Pauses)")
                }
            }
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.12))
                    Capsule().fill(Color.accentColor.opacity(0.8))
                        .frame(width: max(geometry.size.width * CGFloat(bar.progress), 0))
                    ForEach(ticks.indices, id: \.self) { i in
                        Rectangle().fill(ticks[i].alert ? Color.orange : Color.primary.opacity(0.55))
                            .frame(width: 2, height: ticks[i].alert ? 8 : 6)
                            .offset(x: max(geometry.size.width * CGFloat(ticks[i].fraction) - 1, 0))
                    }
                }
            }
            .frame(height: 8)
            .accessibilityLabel(Text("\(bar.label), \(Int((bar.progress * 100).rounded())) percent"))
        }
    }
}

/// The alerts of the stack surface that nobody has acknowledged: at most `maxRows`, the rest under a count.
struct AlertStackView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel
    var maxRows = AlertStack.roomyRows

    var body: some View {
        let _ = play.revision
        let layout = model.alertStack(maxRows: maxRows)
        if !layout.rows.isEmpty {
            VStack(spacing: 4) {
                ForEach(layout.rows) { row in
                    AlertRowView(row: row, model: model)
                }
                if layout.hidden > 0 {
                    Text("+\(layout.hidden) more").font(.caption).foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                }
            }
        }
    }
}

struct AlertRowView: View {
    let row: AlertRow
    @ObservedObject var model: WorldModel

    static let height: CGFloat = 44

    var body: some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 1.5).fill(row.tone.color).frame(width: 3)
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text(row.title).font(.caption.weight(.semibold)).lineLimit(1)
                    if row.pauses { Image(systemName: "pause.circle.fill").font(.caption2).foregroundStyle(.secondary) }
                    if !row.calibrated {
                        Text("provisional").font(.caption2).foregroundStyle(.secondary)
                            .help("No measurement backs this alert's threshold yet")
                    }
                }
                Text(row.text).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
            }
            Spacer(minLength: 0)
            if row.action != nil {
                Button { model.activate(row.action) } label: { Image(systemName: "location.fill") }
                    .help("Fly to it (a)")
            }
            if let kin = row.kin {
                Button { model.toggleWatch(kin: kin) } label: { Image(systemName: row.watched ? "star.fill" : "star") }
                    .help(row.watched ? "Stop watching this group" : "Watch this group (lineage alerts read the watch set)")
            }
            Button { model.acknowledge(alert: row.id) } label: { Image(systemName: "checkmark.circle") }
                .help("Acknowledge (k)")
        }
        .buttonStyle(.borderless)
        .controlSize(.small)
        .padding(.horizontal, 8)
        .frame(height: Self.height)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.primary.opacity(0.12)))
    }
}

/// The alerts over the map of a bottom dock, top left; in a side dock they are in the dock itself.
struct AlertStackOverlay: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel

    var body: some View {
        AlertStackView(play: play, model: model, maxRows: AlertStack.roomyRows)
            .frame(width: 320)
            .padding(EdgeInsets(top: 10 + 30, leading: 10, bottom: 10, trailing: 10))  // (under the readout line of the view's place)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

/// The one line at the bottom of the map: the newest thing the world told. A click flies to it.
struct TickerView: View {
    @ObservedObject var model: WorldModel

    var body: some View {
        if let line = model.tickerLine {
            Text(line.text)
                .font(.callout)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.regularMaterial, in: Capsule())
                .overlay(Capsule().stroke(Color.primary.opacity(0.12)))
                .frame(maxWidth: 560)
                .padding(.bottom, 12)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
                .onTapGesture { model.activate(line.action) }
                .transition(.opacity)
                .animation(.easeInOut(duration: 0.2), value: line.id)
        }
    }
}

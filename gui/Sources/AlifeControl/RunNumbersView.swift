import AlifeCore
import SwiftUI

/// The numbers of a world (docs/APP.md §3.8 row 18), every one the Lab's Status showed, with plain labels and the engine's name in the tooltip.
/// The Run panel of the research Watch and Details show them; they come from the world's own file (`WorldReader`), so a world that is not growing
/// has them too. `clock` is the run's elapsed time and its stop timer, only for the world this app grows now.
struct RunNumbersView: View {
    struct Clock: Equatable {
        var elapsed: TimeInterval
        var remaining: TimeInterval?
    }

    var metrics: MetricsSnapshot?
    var clock: Clock?
    /// What to say when there is no row yet.
    var emptyText = "No numbers yet. The first appear a few seconds after the engine starts."
    /// The box has a title of its own ("Numbers"); the panel's tab already says it.
    var titled = true

    var body: some View {
        if titled { GroupBox("Numbers") { content } } else { content }
    }

    private var content: some View {
            VStack(alignment: .leading, spacing: 6) {
                if let clock {
                    HStack {
                        Text("Elapsed \(formatDuration(clock.elapsed))")
                        if let remaining = clock.remaining { Text("| stops in \(formatDuration(remaining))") }
                    }
                    .font(.callout.monospacedDigit())
                }
                if let m = metrics {
                    Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 4) {
                        GridRow {
                            stat("Step", String(m.step), "step")
                            stat("Alive", number(m.nAlive, "%.0f"), "n_alive")
                            stat("Speed", number(m.stepsPerSecond, "%.1f") + " steps/s", "steps_per_s")
                        }
                        GridRow {
                            stat("Family lines", number(m.clades, "%.0f"), "clades")
                            stat("Deepest generation", number(m.generationMax, "%.0f"), "generation_max")
                            VStack(alignment: .leading, spacing: 0) {
                                Text("Energy check").font(.caption).foregroundStyle(.secondary)
                                Text(m.ledgerError.map { $0 == 0 ? "OK" : String(format: "off by %.0f", $0) } ?? "-")
                                    .font(.title3.monospacedDigit())
                                    .foregroundStyle((m.ledgerError ?? 0) == 0 ? Color.primary : Color.red)
                            }
                            .help("ledger_error: the energy books balance when this is 0")
                        }
                        if m.heightMean != nil {
                            GridRow {
                                stat("Height (mean)", number(m.heightMean, "%.2f"), "height_mean")
                                stat("On the floor", number(m.groundedFrac.map { $0 * 100 }, "%.0f%%"), "grounded_frac")
                                stat("At the surface", number(m.surfaceFrac.map { $0 * 100 }, "%.0f%%"), "surface_frac")
                            }
                        }
                    }
                    ForEach(FeatureMetrics.groups(m.features), id: \.title) { group in
                        DisclosureGroup(group.title) {
                            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 2) {
                                ForEach(group.rows, id: \.key) { row in
                                    GridRow {
                                        Text(row.label).font(.caption).foregroundStyle(.secondary).help(row.key)
                                        Text(FeatureMetrics.text(row.value)).font(.caption.monospacedDigit())
                                    }
                                }
                            }
                            .padding(.vertical, 2)
                        }
                        .font(.callout)
                    }
                } else {
                    Text(emptyText).foregroundStyle(.secondary)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func stat(_ title: String, _ value: String, _ key: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.monospacedDigit())
        }
        .help(key)
    }

    private func number(_ value: Double?, _ format: String) -> String { value.map { String(format: format, $0) } ?? "-" }

    private func formatDuration(_ t: TimeInterval) -> String {
        let s = Int(t.rounded())
        return String(format: "%02d:%02d:%02d", s / 3600, (s / 60) % 60, s % 60)
    }
}

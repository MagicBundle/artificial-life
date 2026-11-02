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
    /// The step is said once, in the sentence above the numbers (a growing world's bar, a saved world's "Saved at step"); the numbers' own is a
    /// second, slightly different one (they are read every two seconds). Off in the `alife` window's panel and Details.
    var showsStep = true
    /// A world that does not run has no speed: the last one measured is history, not a fact about now.
    var showsSpeed = true
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
                        ForEach(Array(cells(m).chunked(by: 3).enumerated()), id: \.offset) { _, row in
                            GridRow {
                                ForEach(row, id: \.title) { cell in
                                    VStack(alignment: .leading, spacing: 0) {
                                        Text(cell.title).font(.caption).foregroundStyle(.secondary)
                                        Text(cell.value).font(.title3.monospacedDigit()).foregroundStyle(cell.alarming ? Color.red : Color.primary)
                                    }
                                    .help(cell.key)
                                }
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

    private struct Cell {
        var title: String
        var value: String
        var key: String
        var alarming = false
    }

    private func cells(_ m: MetricsSnapshot) -> [Cell] {
        var out: [Cell] = []
        if showsStep { out.append(Cell(title: "Step", value: WorldsGuide.grouped(m.step), key: "step")) }
        out.append(Cell(title: "Alive", value: number(m.nAlive, "%.0f"), key: "n_alive"))
        if showsSpeed { out.append(Cell(title: "Speed", value: number(m.stepsPerSecond, "%.1f") + " steps/s", key: "steps_per_s")) }
        out.append(Cell(title: "Family lines", value: number(m.clades, "%.0f"), key: "clades"))
        out.append(Cell(title: "Deepest generation", value: number(m.generationMax, "%.0f"), key: "generation_max"))
        out.append(Cell(title: "Energy check", value: DetailsGuide.energyCheck(m.ledgerError), key: "ledger_error: the energy books balance when this is 0",
                        alarming: (m.ledgerError ?? 0) != 0))
        if m.heightMean != nil {
            out.append(Cell(title: "Height (mean)", value: DetailsGuide.height(m.heightMean), key: "height_mean"))
            out.append(Cell(title: "On the floor", value: number(m.groundedFrac.map { $0 * 100 }, "%.0f%%"), key: "grounded_frac"))
            out.append(Cell(title: "At the surface", value: number(m.surfaceFrac.map { $0 * 100 }, "%.0f%%"), key: "surface_frac"))
        }
        return out
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

private extension Array {
    func chunked(by size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map { Array(self[$0..<Swift.min($0 + size, count)]) }
    }
}

import AlifeCore
import SwiftUI

// The pipeline as the person sees it (docs/APP.md §3.2): what is happening, how far along it is, and what comes next. One card on the
// set-up page, and the same thing as a strip across the top of every other page, so that the wait never has to be looked for.

extension PipelinePlan.Tone {
    var tint: Color {
        switch self {
        case .working: return .blue
        case .attention: return .orange
        case .failure: return .red
        case .success: return .green
        }
    }

    var symbol: String {
        switch self {
        case .working: return "hourglass"
        case .attention: return "pause.circle"
        case .failure: return "exclamationmark.triangle.fill"
        case .success: return "checkmark.circle.fill"
        }
    }
}

/// The buttons of a banner. On the set-up page's card the Stop button also answers Command-period (§3.8 row 17): the page that shows the run in full.
/// The strip on the other pages has no key (a key that ends a run from a page that does not say which run it is would be a trap), and on Watch,
/// where the Run panel holds the one Stop, it has no Stop at all (`hidesStop`).
private struct BannerButtons: View {
    let buttons: [PipelinePlan.Button]
    var stopKey = true
    var hidesStop = false
    let press: (PipelinePlan.Button) -> Void

    var body: some View {
        HStack(spacing: 8) {
            ForEach(buttons.filter { !(hidesStop && $0 == .stop) }, id: \.self) { button in
                let prominent = button == .makePlanet || button == .makePlanetFromHere || button == .watch
                    || (button == .makeAgainNewName || button == .tryAgainNewName)
                if prominent {
                    Button(button.label) { press(button) }.buttonStyle(.borderedProminent)
                } else if button == .stop, stopKey {
                    Button(button.label) { press(button) }.keyboardShortcut(".", modifiers: .command)
                } else {
                    Button(button.label) { press(button) }
                }
            }
        }
        .controlSize(.regular)
    }
}

/// The bar of a banner: the fraction, or an indeterminate bar when none is known, and the time so far.
private struct BannerBar: View {
    let banner: PipelinePlan.Banner
    let elapsed: Double
    var remaining: TimeInterval?

    var body: some View {
        if banner.tone == .working {
            HStack(spacing: 10) {
                if let fraction = banner.fraction {
                    ProgressView(value: fraction)
                } else {
                    ProgressView().progressViewStyle(.linear)
                }
                Text(TimeWords.clock(elapsed)).font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 44, alignment: .trailing)
                    .accessibilityLabel("Time so far \(TimeWords.clock(elapsed))")
                if let remaining {
                    Text("stops in \(TimeWords.countdown(remaining))").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        .accessibilityLabel("The stop timer has \(TimeWords.countdown(remaining)) left")
                }
            }
        }
    }
}

/// The steps of the pipeline, with the one that is under way: what is happening and what comes next, as a list.
struct PipelineSteps: View {
    let plan: PipelinePlan

    private enum Mark { case done, current, waiting, stopped }

    private var steps: [(title: String, mark: Mark)] {
        let grow = plan.growName ?? plan.name
        switch plan.mode {
        case .growThenPlay:
            let first: Mark, second: Mark, third: Mark
            switch plan.state {
            case .growing: (first, second, third) = (.current, .waiting, .waiting)
            case .parked: (first, second, third) = (.stopped, .waiting, .waiting)
            case .making: (first, second, third) = (.done, .current, .waiting)
            case .live, .dismissed: (first, second, third) = (.done, .done, .done)
            case .failed(.growing), .failed(.nothingSaved): (first, second, third) = (.stopped, .waiting, .waiting)
            case .failed(.making), .cancelled: (first, second, third) = (.done, .stopped, .waiting)
            case .finished: (first, second, third) = (.done, .waiting, .waiting)
            case .diedOut: (first, second, third) = (.stopped, .waiting, .waiting)
            }
            return [("Grow the world", first), ("Make the planet \(plan.name)", second), ("Open it in Watch", third)]
        case .makePlanet:
            let first: Mark
            switch plan.state {
            case .live, .dismissed: first = .done
            case .failed, .cancelled, .diedOut: first = .stopped
            default: first = .current
            }
            return [("Make the planet \(plan.name)", first), ("Open it in Watch", plan.isOver && first == .done ? .done : .waiting)]
        case .growOnly, .keepGrowing:
            let first: Mark
            switch plan.state {
            case .finished, .dismissed: first = .done
            case .failed, .diedOut: first = .stopped
            default: first = .current
            }
            return [("Grow \(grow)", first), ("It is saved when it finishes", first == .done ? .done : .waiting)]
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(steps.enumerated()), id: \.offset) { index, step in
                HStack(spacing: 8) {
                    Image(systemName: symbol(step.mark)).foregroundStyle(color(step.mark)).frame(width: 18)
                    Text("\(index + 1). \(step.title)").foregroundStyle(step.mark == .waiting ? .secondary : .primary)
                        .fontWeight(step.mark == .current ? .semibold : .regular)
                    Spacer(minLength: 0)
                }
                .font(.callout)
                .accessibilityElement(children: .combine)
            }
        }
    }

    private func symbol(_ mark: Mark) -> String {
        switch mark {
        case .done: return "checkmark.circle.fill"
        case .current: return "circle.inset.filled"
        case .waiting: return "circle"
        case .stopped: return "pause.circle"
        }
    }

    private func color(_ mark: Mark) -> Color {
        switch mark {
        case .done: return .green
        case .current: return .blue
        case .waiting: return .secondary
        case .stopped: return .orange
        }
    }
}

/// The big card of the set-up page: the title, the sentence, the bar, the steps, what comes next and the buttons.
struct PipelineCard: View {
    @ObservedObject var coordinator: PipelineCoordinator

    var body: some View {
        if let plan = coordinator.plan, let banner = coordinator.banner {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Image(systemName: banner.tone.symbol).font(.title2).foregroundStyle(banner.tone.tint).accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(banner.title).font(.title2.weight(.semibold))
                        Text(banner.detail).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
                    }
                    Spacer(minLength: 0)
                }
                BannerBar(banner: banner, elapsed: coordinator.elapsed, remaining: coordinator.timerRemaining)
                PipelineSteps(plan: plan)
                if let next = banner.next {
                    Text(next).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                BannerButtons(buttons: banner.buttons) { coordinator.press($0) }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(banner.tone.tint.opacity(0.10), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(banner.tone.tint.opacity(0.35)))
            .accessibilityElement(children: .contain)
            .accessibilityLabel("\(banner.title). \(banner.detail)")
        }
    }
}

/// The strip across the top of the other pages: the same banner, one line and a bar, so that the wait is never out of sight.
struct PipelineStrip: View {
    @ObservedObject var coordinator: PipelineCoordinator
    /// The Run panel of Watch holds the Stop of the run that Watch shows: the strip does not repeat it there.
    var hidesStop = false
    var openSetup: () -> Void

    var body: some View {
        if let banner = coordinator.banner {
            HStack(alignment: .center, spacing: 12) {
                Image(systemName: banner.tone.symbol).foregroundStyle(banner.tone.tint).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(banner.title).font(.callout.weight(.semibold)).lineLimit(1)
                    Text(banner.detail + (banner.next.map { (banner.detail.hasSuffix(".") ? " " : ". ") + $0 } ?? "")).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if banner.tone == .working {
                    BannerBar(banner: banner, elapsed: coordinator.elapsed, remaining: coordinator.timerRemaining).frame(minWidth: 140, maxWidth: 320)
                }
                Spacer(minLength: 8)
                BannerButtons(buttons: banner.buttons, stopKey: false, hidesStop: hidesStop) { coordinator.press($0) }.controlSize(.small)
                Button("Show progress", action: openSetup).buttonStyle(.link).controlSize(.small)
                    .help("Show this on the New world page, with the steps and the commands")
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background {
                // (an opaque page colour under the tint: the strip sits under the title bar, over a dark Watch or a light page alike)
                ZStack { Color(nsColor: .windowBackgroundColor); banner.tone.tint.opacity(0.12) }
            }
            .overlay(alignment: .bottom) { Divider() }
            .accessibilityElement(children: .contain)
            .accessibilityLabel("\(banner.title). \(banner.detail)")
        }
    }
}

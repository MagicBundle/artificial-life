import Foundation

// The chapter card and the Report tab (DESIGN §34.7): what the world says at the end of a chapter, and the reports of the
// chapters before it. The sentences are the server's (code-written, checked, DESIGN §34.4); the client lays them out, links
// each to the entity it names, and adds the strip of what has changed for the player since. Pure, so the card is tested
// without a window.

/// The report of one chapter as the window shows it.
package struct ReportCard: Equatable, Sendable {
    package struct Line: Equatable, Identifiable, Sendable {
        package var id: String
        package var text: String
        /// Where a click on the sentence goes (the first entity it names that the report can place).
        package var action: ChronicleAction?
    }

    package var chapter: Int
    package var nAbs: Int
    package var a: Int
    package var b: Int
    package var headline: Line?
    package var lead: Line?
    package var body: [Line]
    /// Body sentences left out of `body` by the card's cap.
    package var hiddenBody: Int
    package var interventions: [Line]
    package var ticker: [ReportBody.Tick]
    /// Why the chapter has no report (a row missing, an unobserved run): nothing is guessed in its place.
    package var withheld: String?
    /// A select report (G3c) is on its way: the template one shows, with a small "ordering" mark.
    package var ordering: Bool
    /// The chapter had nothing to report: "A quiet chapter".
    package var quiet: Bool
    package var since: ReportBody.Since
    package var factsDigest: String

    /// The body sentences a card shows (the Report tab shows them all).
    package static let cardBodyLines = 6

    /// The lines with ids that are unique in the list: a list of lines is drawn by id, and two sentences of one fact share the
    /// fact's id in an older record (the world makes them unique now). The second is `id#2`, the third `id#3`.
    package static func unique(_ lines: [Line]) -> [Line] {
        var seen: [String: Int] = [:]
        return lines.map { line in
            seen[line.id, default: 0] += 1
            let n = seen[line.id] ?? 1
            return n == 1 ? line : Line(id: "\(line.id)#\(n)", text: line.text, action: line.action)
        }
    }

    package static func make(_ record: PlayReport, bodyLimit: Int? = nil) -> ReportCard {
        let report = record.report
        func line(_ sentence: ReportBody.Sentence?) -> Line? {
            sentence.map { Line(id: $0.id, text: $0.text, action: ChronicleAction.of(refs: $0.refs, entities: report.entities)) }
        }
        let all = unique(report.body.map {
            Line(id: $0.id, text: $0.text, action: ChronicleAction.of(refs: $0.refs, entities: report.entities))
        })
        let shown = bodyLimit.map { Array(all.prefix($0)) } ?? all
        return ReportCard(
            chapter: report.chapter, nAbs: report.nAbs, a: report.a, b: report.b, headline: line(report.headline),
            lead: line(report.lead), body: shown, hiddenBody: all.count - shown.count,
            interventions: unique(report.interventions.compactMap { line($0) }), ticker: report.ticker,
            withheld: report.isWithheld ? report.withheld.joined(separator: "; ") : nil, ordering: !record.isFinal,
            quiet: report.headline?.id == "HQ", since: report.since, factsDigest: report.factsDigest)
    }
}

/// What has changed for the player since the last look: units, powers, effects that end, the alerts of the chapter.
package struct SinceStrip: Equatable, Sendable {
    package struct Ending: Equatable, Sendable {
        package var label: String
        package var end: Int
    }

    package struct Count: Equatable, Sendable {
        package var label: String
        package var count: Int
    }

    package var haveUnits: Int?
    package var capUnits: Int?
    /// The step the next spent unit comes back.
    package var nextUnit: Int?
    /// The powers usable now, by label.
    package var ready: [String]
    package var ending: [Ending]
    package var alertCounts: [Count]
    /// The most important alerts of the chapter, with their fly-to.
    package var top: [AlertRow]

    package var isEmpty: Bool {
        haveUnits == nil && ready.isEmpty && ending.isEmpty && alertCounts.isEmpty && top.isEmpty
    }

    /// The strip for the chapter `(range)` from the live store: the units and powers as `ENVS` says now, the effects ending
    /// within `horizon` steps, the alerts raised in the chapter (by class from the report when it has them, else counted on the
    /// board).
    package static func make(store: WorldStore, range: Range<Int>, reported: ReportBody.Since?, horizon: Int, topCount: Int = 3,
                             watch: WatchSet = WatchSet()) -> SinceStrip {
        let step = Int(store.clock?.head.step ?? 0)
        let powers = store.play?.powers ?? [:]
        let ready = PowerCatalog.order.compactMap { name -> String? in
            guard let spec = powers[name.rawValue], store.powerStates[name.rawValue].map({ $0.busy == nil }) ?? false else { return nil }
            return spec.label
        }
        let labelOfAction = Dictionary(powers.values.map { ($0.action, $0.label) }, uniquingKeysWith: { a, _ in a })
        let ending = (store.environment?.environment.effects ?? []).compactMap { effect -> Ending? in
            guard let end = effect.end, end > step, end - step <= horizon else { return nil }
            return Ending(label: labelOfAction[effect.action] ?? effect.action, end: end)
        }.sorted { $0.end < $1.end }
        let classes = store.play?.alerts?.classes ?? [:]
        let counts: [Count]
        if let reported, !reported.alerts.isEmpty {
            counts = reported.alerts.map { Count(label: classes[$0.key]?.label ?? $0.key, count: $0.value) }
        } else {
            var tally: [String: Int] = [:]
            for alert in store.alerts.raised(in: range) { tally[alert.alertClass, default: 0] += 1 }
            counts = tally.map { Count(label: classes[$0.key]?.label ?? $0.key, count: $0.value) }
        }
        let top = store.alerts.raised(in: range).prefix(topCount).map { AlertStack.row(for: $0, watch: watch) }
        let influence = store.influence
        return SinceStrip(
            haveUnits: influence?.have, capUnits: influence?.cap,
            nextUnit: influence.flatMap { $0.have < $0.cap ? $0.freeAt.min() : nil }, ready: ready, ending: ending,
            alertCounts: counts.sorted { $0.count != $1.count ? $0.count > $1.count : $0.label < $1.label },
            top: Array(top))
    }
}

/// The card over the map at a chapter's end (reason 3 of `CLCK`).
package struct ChapterCard: Equatable, Sendable {
    package enum Form: Equatable, Sendable {
        /// The checkpoint is being written: the card is up, its buttons are not.
        case saving
        /// The world is stopped at the chapter's end: the whole card with its buttons.
        case ready
        /// The pause is off: the same words as a notice that the player closes (the world runs on).
        case notice
    }

    package var nAbs: Int
    package var number: Int
    /// "Chapter 3 complete".
    package var title: String
    package var form: Form
    package var report: ReportCard?
    /// The previous chapter's headline, greyed, while this chapter's report is still being built.
    package var previousHeadline: String?
    package var since: SinceStrip

    package var saving: Bool { form == .saving }

    /// The card for the newest `CLCK`, or nil when the clock is not at a chapter's end. `noticeVisible` is whether a chapter
    /// whose pause is off is still on screen (the window times it).
    package static func make(store: WorldStore, originChapter: Int, noticeVisible: Bool = true,
                             watch: WatchSet = WatchSet()) -> ChapterCard? {
        guard let clock = store.clock, clock.reason == .chapter else { return nil }
        let nAbs = Int(clock.chapter)
        let bar = ChapterBar(clock: clock, originChapter: originChapter, baseRate: store.play?.baseRate ?? 14.4)
        let form: Form
        if clock.saving {
            // (a world whose pause is off keeps running through the save: its card is the notice, not a card with no buttons)
            form = clock.chapterPauseArmed || clock.rate == 0 ? .saving : .notice
        } else if clock.rate == 0 || clock.chapterPauseArmed {
            form = .ready
        } else if noticeVisible {
            form = .notice
        } else {
            return nil
        }
        let record = store.reports.report(nAbs: nAbs)
        let length = bar.chapterLength > 0 ? bar.chapterLength : Int(clock.chapterLen)
        let end = nAbs * max(length, 1)
        let range = (max(end - length, 0) + 1)..<(end + 1)  // the steps (a, b]
        let card = record.map { ReportCard.make($0, bodyLimit: ReportCard.cardBodyLines) }
        let previous = store.reports.report(nAbs: nAbs - 1)?.report.headline?.text
        return ChapterCard(
            nAbs: nAbs, number: max(nAbs - max(originChapter, 0), 1), title: "Chapter \(max(nAbs - max(originChapter, 0), 1)) complete",
            form: form, report: card, previousHeadline: card == nil ? previous : nil,
            since: SinceStrip.make(store: store, range: range, reported: record?.report.since, horizon: max(length, 1), watch: watch))
    }
}

/// The Report tab: the chapters held, newest first; the newest in full, the earlier ones as headlines that open.
package struct ReportTabModel: Equatable, Sendable {
    package var newest: ReportCard?
    package var earlier: [ReportCard]

    package init(shelf: ReportShelf) {
        let cards = shelf.newestFirst.map { ReportCard.make($0) }
        newest = cards.first
        earlier = Array(cards.dropFirst())
    }
}

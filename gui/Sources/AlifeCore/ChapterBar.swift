import Foundation

// The chapter bar of the clock strip (DESIGN §34.7): which chapter the world is in, how far, when the next report is due, and
// whether the world will stop for it. A pure reduction of the newest `CLCK`, so the bar is tested without a window.

/// Where a world's chapter numbering starts. A planet is a fork made at some step: the first chapter after the fork is
/// "Chapter 1", whatever the absolute number of the chapter that step falls in (DESIGN §34.2.1).
package enum ChapterOrigin {
    /// The absolute chapter the numbering starts after: `made_at_step / chapter` of the world's `play.json`, 0 for a preset or
    /// an adopted run (no planet) and for a file that cannot be read.
    package static func chapter(playFile url: URL, chapterLength: Int) -> Int {
        guard chapterLength > 0, let planet = PlayWorldFile.read(at: url)?.planet else { return 0 }
        return max(planet.madeAtStep, 0) / chapterLength
    }
}

package struct ChapterBar: Equatable, Sendable {
    /// A mark on the bar: a told event (a toast-sized one) or an alert of the stack, at a fraction of the chapter.
    package struct Tick: Equatable, Sendable {
        package var fraction: Double
        package var alert: Bool
    }

    /// The number shown: the absolute chapter less the origin, at least 1.
    package var number: Int
    package var label: String
    /// `in_chapter / chapter_len`, 0...1; 0 at the very first moment of a fork (the clock still names the chapter the fork
    /// was made in, at its end).
    package var progress: Double
    package var stepsLeft: Int
    /// The step the chapter on screen starts after (the chapter is `(chapterStart, chapterStart + chapterLength]`).
    package var chapterStart: Int
    package var chapterLength: Int
    /// "4:10": the time to the chapter's end at the pace of the last two seconds; nil while paused or before a pace is known.
    package var remaining: String?
    /// "next report in 4:10".
    package var nextReport: String?
    /// The world stops by itself at the chapter's end (`CLCK.flags` bit 0).
    package var pauseArmed: Bool
    /// An alert class is armed to stop it (bit 1).
    package var alertPauseArmed: Bool
    /// The clock names a chapter that is not after the origin yet: the first moment of a fork.
    package var atOrigin: Bool

    package init(clock: PlayClock?, originChapter: Int, baseRate: Double, requested: PlayRate? = nil) {
        guard let clock else {
            number = 1
            label = "Chapter 1"
            progress = 0
            stepsLeft = 0
            chapterStart = 0
            chapterLength = 0
            remaining = nil
            nextReport = nil
            pauseArmed = false
            alertPauseArmed = false
            atOrigin = false
            return
        }
        let length = max(Int(clock.chapterLen), 1)
        let relative = Int(clock.chapter) - max(originChapter, 0)
        atOrigin = relative < 1
        number = max(relative, 1)
        label = "Chapter \(number)"
        chapterLength = length
        let done = atOrigin ? 0 : min(Int(clock.inChapter), length)
        progress = Double(done) / Double(length)
        stepsLeft = length - done
        chapterStart = (max(originChapter, 0) + number - 1) * length
        pauseArmed = clock.chapterPauseArmed
        alertPauseArmed = clock.alertPauseArmed
        let paused = requested.map { $0 == .multiplier(0) } ?? (clock.rate == 0)
        let pace: Double? = {
            if paused { return nil }
            if clock.achieved > 0.5 { return Double(clock.achieved) }
            if let requested, case .multiplier(let m) = requested, m > 0, baseRate > 0 { return m * baseRate }
            if clock.rate > 0, baseRate > 0 { return Double(clock.rate) * baseRate }
            return nil
        }()
        if let pace, stepsLeft > 0 {
            let text = Self.clockText(seconds: Double(stepsLeft) / pace)
            remaining = text
            nextReport = "next report in " + text
        } else {
            remaining = nil
            nextReport = nil
        }
    }

    /// "4:10" for 250 s; "0:07" for 7 s; minutes keep counting past an hour ("75:00").
    package static func clockText(seconds: Double) -> String {
        let total = max(Int(seconds.rounded()), 0)
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    /// The marks for the chapter on screen: toast-sized events (importance 2 or more, by step) and the alerts of the stack
    /// (except the chapter's own, which is the bar's end), as fractions of the chapter `(start, start + length]`.
    package func ticks(events: [WorldEvent], alerts: [PlayAlert]) -> [Tick] {
        guard chapterLength > 0 else { return [] }
        func fraction(_ step: Int) -> Double? {
            guard step > chapterStart, step <= chapterStart + chapterLength else { return nil }
            return Double(step - chapterStart) / Double(chapterLength)
        }
        var out: [Tick] = []
        for alert in alerts where alert.surface == .alert && alert.alertClass != "chapter" && alert.state != .cleared {
            if let f = fraction(alert.step) { out.append(Tick(fraction: f, alert: true)) }
        }
        for event in events where event.imp >= 2 {
            if let f = fraction(event.step) { out.append(Tick(fraction: f, alert: false)) }
        }
        return Array(out.sorted { $0.fraction != $1.fraction ? $0.fraction < $1.fraction : $0.alert && !$1.alert }.prefix(48))
    }
}

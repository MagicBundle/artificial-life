import AlifeCore
import AppKit
import Foundation
import simd

/// The decide half of the World window (DESIGN §34.7): the chapter bar and card, the alert stack and its acknowledgements, the
/// watch set, the pauses the player armed, the ticker, the attention a pause asks for, and the exchange about a power. The
/// decisions are the pure types of `AlifeCore` (`ChapterBar`, `ChapterCard`, `AlertStack`, `WatchSet`, `AutopausePrefs`,
/// `Ticker`, `AttentionTracker`, `PowersModel`); this applies them to the world's control channel and the window.
extension WorldModel {
    // MARK: Reading the store

    /// The step the world is at, as the newest clock says (0 before the first).
    var currentStep: Int { Int(play.store.clock?.head.step ?? 0) }

    var chapterBar: ChapterBar {
        ChapterBar(clock: play.store.clock, originChapter: play.originChapter, baseRate: play.store.play?.baseRate ?? 14.4,
                   requested: play.requestedRate)
    }

    /// The card for a chapter's end, if the clock is at one.
    var chapterCard: ChapterCard? {
        let noticeShown = play.store.clock.map { chapterNotice == Int($0.chapter) } ?? false
        return ChapterCard.make(store: play.store, originChapter: play.originChapter, noticeVisible: noticeShown, watch: watch)
    }

    /// The card is a one-line bar: folded by the player, or the map is a power's target.
    var cardFolded: Bool { cardCollapsed || powers.isAiming }

    /// The stack for the dock or the map, `maxRows` high.
    func alertStack(maxRows: Int) -> AlertStackLayout {
        AlertStack.layout(play.store.alerts, step: currentStep, watch: watch, maxRows: maxRows,
                          excluding: chapterCard == nil ? [] : ["chapter"])
    }

    /// A power is being set up in the Powers tab: the side dock gives the tab the room (the inspector card waits).
    var editingPower: Bool { tab == .powers && powers.offer != nil }

    var offers: [PowerOffer] { PowerOffer.offers(header: play.store.play, states: play.store.powerStates) }

    // MARK: Alerts

    /// What a delivery did to the alerts. The backlog loaded at the start is history: no toast, no ticker line, no badge.
    func handle(alerts changes: [AlertChange], backlog: Bool) {
        guard !backlog else { return }
        // The chapter's own alert is the card (or its notice): a toast beside it would say the same thing twice.
        let toasts = AlertBoard.toasts(from: changes, backlog: false).filter { $0.alertClass != "chapter" }
        if !toasts.isEmpty {
            toastQueue.push(alerts: toasts, at: now())
            self.toasts = toastQueue.toasts
        }
        let raised = changes.filter { $0.kind == .raised }.map(\.alert)
        ticker.show(alerts: raised, at: now())
        tickerLine = ticker.line
        let stacked = raised.filter { $0.surface == .alert && $0.alertClass != "chapter" }.count
        attention.raised(stacked, windowIsKey: windowIsKey())
        setBadge(attention.badge)
        scheduleTimers()
    }

    /// The newest line of a report that just arrived is a line of the ticker if it is news: dated within `freshSteps` of the
    /// chapter's end (a report's lines go back to the chapter's start, and a line of 750 steps ago is not news; the Report tab
    /// has it). The backlog of a re-opened world is not news either.
    func handle(reports: [PlayReport]) {
        guard let report = reports.last(where: { !$0.isBacklog && !$0.report.isWithheld && $0.isFinal }),
              let line = report.report.ticker.last(where: { $0.step > report.report.b - Self.freshSteps }) else { return }
        ticker.show(id: 0, text: line.text, at: now())
        tickerLine = ticker.line
        scheduleTimers()
    }

    func handle(plans: [PlayPlan]) {
        powers.see(play.store.plans)
        for plan in plans {
            if let command = powers.receive(plan) { play.send(command) }
        }
    }

    /// The check button of a row, or `k`: the alert leaves the stack and the world is told. `Esc` never does this.
    func acknowledge(alert id: UInt64) {
        play.acknowledge(alert: id)
    }

    func acknowledgeNewest() {
        guard let alert = play.store.alerts.newestUnacknowledged(atStep: currentStep) else { return }
        acknowledge(alert: alert.id)
    }

    /// `a`: fly to the newest alert of the stack.
    func flyToNewestAlert() {
        guard let alert = play.store.alerts.newestUnacknowledged(atStep: currentStep), let action = ChronicleAction.of(alert) else {
            return
        }
        activate(action)
    }

    /// The star of a row or a lineage: watch the kin group, or stop; at most three.
    func toggleWatch(kin id: Int) {
        let was = watch
        var next = watch
        if let command = next.toggle(id) {
            watch = next
            play.send(command)
        } else if !was.contains(id) {
            tell("Three groups are watched: stop watching one first")
        }
    }

    // MARK: Pauses

    /// The lock icon: the world stops (or not) at a chapter's end. The truth is the clock's flag.
    func toggleChapterPause() {
        setChapterPause(!(play.store.clock?.chapterPauseArmed ?? true))
    }

    func setChapterPause(_ on: Bool) {
        autopause.chapter = on
        saveAutopause()
        play.send(.autopause(.chapter(on)))
    }

    func setAlertPause(_ name: String, _ on: Bool) {
        autopause.alert[name] = on
        saveAutopause()
        play.send(.autopause(.alert(name, on)))
    }

    private func saveAutopause() {
        defaults.set(autopause.json, forKey: Self.autopauseKey)
    }

    /// After a header: bring the new world to the pauses the player chose.
    func resendAutopause() {
        let header = play.store.play
        let classes = Set(header?.alerts?.classes.keys.map { $0 } ?? [])
        for command in autopause.commands(against: header?.autopause, classes: classes) { play.send(command) }
    }

    /// The alert classes the player can arm: the stack's, not the log's.
    var pausableClasses: [(name: String, label: String)] {
        (play.store.play?.alerts?.classes ?? [:]).filter { $0.value.surface == .alert && $0.key != "chapter" }
            .map { ($0.key, $0.value.label) }.sorted { $0.label < $1.label }
    }

    func alertPauseOn(_ name: String) -> Bool {
        autopause.alertOn(name, default: play.store.play?.autopause)
    }

    // MARK: The clock

    /// A `CLCK` arrived: a chapter notice for a world that does not stop, and the attention a stop asks for.
    func handleClock() {
        guard let clock = play.store.clock else { return }
        let chapter = Int(clock.chapter)
        if clock.reason == .chapter {
            if chapter != lastCardChapter {
                lastCardChapter = chapter
                cardCollapsed = false
                _ = powers.escape()  // a preview or an aim from the chapter before: its steps and cautions are old
            }
            if clock.rate != 0, !clock.saving, !clock.chapterPauseArmed, noticeShownFor < chapter {
                noticeShownFor = chapter
                chapterNotice = chapter
                noticeExpires = now() + Self.noticeSeconds
                scheduleTimers()
            }
        }
        if (clock.reason == .chapter || clock.reason == .alert), clock.rate == 0, !clock.saving,
           attention.paused(atStep: currentStep, windowIsKey: windowIsKey()) {
            requestAttention()
        }
    }

    static let noticeSeconds = 15.0
    /// A line of a report is news for the ticker when it is dated this close to the chapter's end (one social snapshot).
    static let freshSteps = 250

    /// Closes the notice of a chapter (its own button, or the time).
    func dismissNotice() {
        chapterNotice = nil
        noticeExpires = nil
    }

    // MARK: Timers (toasts, the ticker, the notice)

    func expireTimers() {
        expireToasts()
        if ticker.expire(at: now()) { tickerLine = nil }
        if let end = noticeExpires, end <= now() { dismissNotice() }
    }

    func scheduleTimers() {
        let candidates = [toastQueue.nextExpiry, ticker.nextExpiry, noticeExpires].compactMap { $0 }
        guard let next = candidates.min() else { return }
        let wait = max(next - now(), 0) + 0.05
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(wait))
            self?.expireTimers()
            self?.scheduleTimers()
        }
    }

    // MARK: Attention (the dock icon)

    /// The window is the one the unread alerts and the dock bounce are about. Called once the view has a window.
    func observeWindow(_ window: NSWindow?) {
        for token in windowObservers { NotificationCenter.default.removeObserver(token) }
        windowObservers = []
        guard let window else { return }
        windowIsKey = { [weak window] in (window?.isKeyWindow ?? false) && NSApp.isActive }
        requestAttention = { NSApp.requestUserAttention(.informationalRequest) }
        setBadge = { NSApp.dockTile.badgeLabel = $0 }
        let center = NotificationCenter.default
        for name in [NSWindow.didBecomeKeyNotification, NSApplication.didBecomeActiveNotification] {
            windowObservers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] note in
                // (the notification's object is the window, or the app)
                let isThis = (note.object as? NSWindow).map { $0 === window } ?? true
                MainActor.assumeIsolated { if isThis { self?.windowBecameKey() } }
            })
        }
    }

    func windowBecameKey() {
        attention.windowBecameKey()
        setBadge(nil)
    }

    // MARK: Powers

    /// A click on a power of the grid: a global one is previewed at once, the others wait for a click on the map.
    func choosePower(_ offer: PowerOffer) {
        powers.see(play.store.plans)
        powers.skipsConfirmForCostOne = defaults.bool(forKey: Self.skipConfirmKey)
        if let command = powers.choose(offer) { play.send(command) }
    }

    static let skipConfirmKey = "world.skipConfirmCostOne"

    func setPowerValue(_ key: PowerCommand.Key, _ value: Double) {
        if let command = powers.set(key, to: value) { play.send(command) }
    }

    func confirmPower() {
        if let command = powers.confirm() { play.send(command) }
    }

    /// `Return` sends a power only while the Powers tab shows the preview it would send: after a change of tab the preview is
    /// not on screen, and a key must not apply what the player cannot see (third review, stage 8).
    var returnConfirmsPower: Bool { tab == .powers && powers.canConfirm }

    func cancelPower(_ name: PowerCommand.Name) {
        powers.see(play.store.plans)
        play.send(powers.cancelCommand(name))
    }

    /// The floor point under `point` of the free rectangle.
    func floorPoint(at point: CGPoint) -> SIMD2<Float>? {
        projector(size: freeRect.size).world(at: point, height: min(max(camera.target.z, 0), max(height, 0)))
    }

    /// A click while a power is being aimed names its target; true when it was taken.
    func aimClick(at point: CGPoint) -> Bool {
        guard powers.isAiming else { return false }
        guard let place = floorPoint(at: point) else {
            tell("That point is not on the floor: aim at the map")
            return true
        }
        let patches = play.store.patches
        if let command = powers.aim(at: place, patches: patches, size: worldSize) {
            play.send(command)
        } else if powers.offer?.scope == .patch, patches.isEmpty {
            tell("This world has no food patch to aim at")
        }
        return true
    }

    /// The pointer over the map while a power is aimed: where the target would be.
    func aimHover(at point: CGPoint?) {
        guard powers.isAiming, let point else {
            overlay.setAim(point: nil, world: nil)
            return
        }
        overlay.setAim(point: point, world: floorPoint(at: point))
    }

    // MARK: Esc

    /// `Esc`, one level at a time: aiming or the preview of a power, then the chapter card (folded to the map), then the
    /// selection and the follow. It never resumes the world and never acknowledges an alert (DESIGN §34.7).
    func escape() {
        if powers.escape() { return }
        if chapterCard != nil, !cardCollapsed {
            cardCollapsed = true
            return
        }
        clearSelection()
    }
}

@testable import AlifeCore
import AppKit
import CoreGraphics
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// The decide half's views: they build and draw from the records, and the side dock fits the window's minimum size with the
/// inspector open, three alert rows and the chapter card (DESIGN §34.7: measured through `NSHostingView`, which draws AppKit
/// controls as the app does; `ImageRenderer` cannot).
@MainActor
@Suite struct WorldDecideViewTests {
    // MARK: Rig

    private func rig(_ scratch: Scratch, strikeReady: Bool = true) async throws -> (play: PlayController, model: WorldModel) {
        let commands = scratch.url.appendingPathComponent("commands.txt")
        let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done")
        let play = PlayController()
        let suite = try #require(UserDefaults(suiteName: "alife-views-\(UUID().uuidString)"))
        let model = WorldModel(play: play, defaults: suite)
        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                   arguments: [Repo.playFixtureURL("3d_g3b").path, commands.path])
        #expect(await eventually { play.store.header != nil && play.store.plans.records.count == 7 && play.store.clock?.reason == .alert })
        if strikeReady, var record = play.store.environment {
            record.environment.powers?["strike"] = .init(ready: nil, busy: nil)
            play.store.apply(.envs(record))
        }
        model.setFreeRect(CGRect(x: 0, y: 0, width: 740, height: 700))
        model.goHome()
        return (play, model)
    }

    /// Pixels with any alpha in an image of the view, drawn by AppKit in a window that is never shown (`ImageRenderer` draws
    /// neither a `ScrollView` nor an AppKit-backed control).
    private func inked<V: View>(_ view: V, width: CGFloat, height: CGFloat) throws -> Int {
        let host = NSHostingView(rootView: view.frame(width: width, height: height, alignment: .topLeading))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: height), styleMask: [.borderless], backing: .buffered, defer: true)
        window.contentView = host
        host.frame = NSRect(x: 0, y: 0, width: width, height: height)
        host.layoutSubtreeIfNeeded()
        let rep = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: rep)
        guard let data = rep.bitmapData else { return 0 }
        let bytes = rep.bytesPerRow * rep.pixelsHigh
        let channels = rep.samplesPerPixel
        guard channels == 4 else { return 0 }
        return stride(from: 3, to: bytes, by: 4).reduce(0) { $0 + (data[$1] > 0 ? 1 : 0) }
    }

    /// The height a view asks for at `width`, laid out by AppKit.
    private func height<V: View>(_ view: V, width: CGFloat) -> CGFloat {
        let host = NSHostingView(rootView: view.frame(width: width))
        host.layoutSubtreeIfNeeded()
        return host.fittingSize.height
    }

    private func chapterClock(reason: PlayClock.Reason = .chapter, rate: Float = 0, saving: Bool = false, flags: UInt16 = 1) -> PlayClock {
        PlayClock(head: RecordHead(step: 12_000, seq: 0, version: 1, flags: 0), rate: rate, achieved: rate == 0 ? 0 : 14.4, stepMs: 15,
                  frameMs: 3, lateSteps: 0, saveStallSteps: 0, reason: reason, saving: saving, flags: flags, fe: 1, chapter: 2,
                  inChapter: 3000, chapterLen: 6000)
    }

    // MARK: The views draw

    @Test func theChapterBarTheStackAndTheRowsDraw() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown() }
        #expect(try inked(ChapterBarView(play: play, model: model), width: 340, height: 50) > 200)
        #expect(try inked(AlertStackView(play: play, model: model, maxRows: 3), width: 340, height: 200) > 2000)
        #expect(try inked(AlertStackOverlay(play: play, model: model), width: 400, height: 300) > 2000)
        let row = AlertStack.row(for: try #require(play.store.alerts.alert(id: 12_000_601)))
        #expect(try inked(AlertRowView(row: row, model: model), width: 340, height: 60) > 500)
        // Nothing on the stack, nothing drawn.
        for id in [12_000_600, 12_000_601, 11_500_603] as [UInt64] { play.acknowledge(alert: id) }
        #expect(try inked(AlertStackView(play: play, model: model), width: 340, height: 200) == 0)
    }

    @Test func theTickerDrawsOnlyWhileThereIsALine() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown() }
        #expect(try inked(TickerView(model: model), width: 600, height: 80) == 0)
        model.ticker.show(id: 0, text: "Storm Merrow reached named strength.", at: 0)
        model.tickerLine = model.ticker.line
        #expect(try inked(TickerView(model: model), width: 600, height: 80) > 500)
    }

    @Test func theChapterCardDrawsInItsThreeFormsAndNotWhenThereIsNone() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown() }
        // (the fixture's clock is an alert stop: no card)
        #expect(try inked(ChapterCardView(play: play, model: model), width: 740, height: 700) == 0)
        play.store.apply(.clck(chapterClock()))
        let full = try inked(ChapterCardView(play: play, model: model), width: 740, height: 700)
        #expect(full > 8000, "the card: \(full)")
        model.cardCollapsed = true
        let folded = try inked(ChapterCardView(play: play, model: model), width: 740, height: 700)
        #expect(folded > 500 && folded < full / 2, "a one-line bar: \(folded) against \(full)")
        model.cardCollapsed = false
        play.store.apply(.clck(chapterClock(rate: 1, saving: true)))
        #expect(try inked(ChapterCardView(play: play, model: model), width: 740, height: 700) > 8000)
        play.store.apply(.clck(chapterClock(rate: 1, flags: 2)))
        model.handleClock()
        let notice = try inked(ChapterCardView(play: play, model: model), width: 740, height: 700)
        #expect(notice > 500 && notice < full / 2, "the notice: \(notice)")
    }

    @Test func aSmallMapKeepsHeadlineAndLeadOnlyAndAQuietChapterKeepsLittle() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown() }
        play.store.apply(.clck(chapterClock()))
        let tall = try inked(ChapterCardView(play: play, model: model), width: 740, height: 700)
        model.setFreeRect(CGRect(x: 0, y: 0, width: 740, height: ChapterCardView.compactBelow - 20))
        let short = try inked(ChapterCardView(play: play, model: model), width: 740, height: 700)
        #expect(short < tall, "the body gives way: \(short) against \(tall)")
    }

    @Test func theReportTabDrawsTheNewestAndTheEarlierChapters() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown() }
        #expect(try inked(ReportTabView(play: play, model: model), width: 360, height: 600) > 4000)
        let empty = PlayController()
        #expect(try inked(ReportTabView(play: empty, model: WorldModel(play: empty, defaults: UserDefaults(suiteName: "alife-empty-\(UUID().uuidString)")!)), width: 360, height: 300) > 100,
                "a world with no report says so")
    }

    @Test func thePowersTabDrawsTheGridThePipsAndTheExchange() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown() }
        let grid = try inked(PowersTabView(play: play, model: model), width: 360, height: 700)
        #expect(grid > 3000, "the grid, the pips, the running Food and the cards: \(grid)")
        model.choosePower(try #require(model.offers.first { $0.name == .food }))
        let chosen = try inked(PowersTabView(play: play, model: model), width: 360, height: 900)
        #expect(chosen > grid, "the sliders and the waiting mark are more ink")
        #expect(try inked(InfluenceMeterView(influence: play.store.influence), width: 340, height: 30) > 100)
        #expect(try inked(InfluenceMeterView(influence: nil), width: 340, height: 30) == 0)
        let offer = try #require(model.offers.first)
        #expect(try inked(PowerCell(offer: offer, selected: true, step: 12_000) {}, width: 170, height: 90) > 500)
        #expect(PowerEditorView.format(.mult, 1.3) == "x1.3" && PowerEditorView.format(.kill, 0.45) == "45%")
        #expect(PowerEditorView.format(.radius, 15.625) == "15.625 units")
        #expect(PowerEditorView.duration(6000, baseRate: 14.4) == "6,000 steps (6.9 min at 1x)")
    }

    @Test func aLineageRowShowsAllItsDeathsAndGroupedCounts() throws {
        // The Mac day's finding: "died: 3091 starved, 3099 killed, 3..." was cut to one line and "6.216 born" read as a decimal.
        let store = try PlayFixtureStore.store("3d")
        var lineage = try #require(CastPolicy.lineages(store.identities, territory: store.territory).first)
        lineage.deaths = [0, 0, 0, 0, 0]
        lineage.births = 6216
        let plain = height(LineageRow(lineage: lineage, names: [], selected: false), width: 320)
        lineage.deaths = [3091, 3099, 4005, 120, 77]
        let long = height(LineageRow(lineage: lineage, names: [], selected: false), width: 320)
        #expect(long > plain + 10, "the death line takes the lines it needs: \(plain) against \(long)")
        #expect(CastPolicy.deathSummary(lineage.deaths) == "3,091 starved, 3,099 killed, 4,005 aged, 120 catastrophe, 77 storm")
        #expect(Numbers.grouped(6216) == "6,216")
    }

    @Test func aWorldWithNoPowersSaysSo() throws {
        let play = PlayController()
        let model = WorldModel(play: play, defaults: UserDefaults(suiteName: "alife-nopowers-\(UUID().uuidString)")!)
        #expect(model.offers.isEmpty)
        #expect(try inked(PowersTabView(play: play, model: model), width: 360, height: 300) > 100)
    }

    // MARK: The side dock fits the minimum window

    @Test func theSideDockFitsTheMinimumWindow() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown() }
        play.store.apply(.clck(chapterClock()))  // the card is over the map, not in the dock
        let width = DockLayout.defaultExtent
        // The fixed parts, measured, are inside the budget the layout plans with.
        let strip = height(ClockStripView(play: play, model: model).padding(.horizontal, 10).padding(.top, 10).padding(.bottom, 4), width: width)
        let storage = height(StorageStripView(play: play, model: model).padding(.horizontal, 10).padding(.bottom, 6), width: width)
        let row = height(AlertRowView(row: AlertStack.row(for: try #require(play.store.alerts.alert(id: 12_000_601))), model: model), width: width - 20)
        let picker = height(Picker("", selection: .constant(DockTab.cast)) { ForEach(DockTab.allCases) { Text($0.rawValue).tag($0) } }
            .pickerStyle(.segmented).labelsHidden().padding(.horizontal, 10).padding(.vertical, 8), width: width)
        let full = try PlayFixtureStore.store("3d")  // (a world with all the records of a G3a card: organism, storm, group)
        let cards = [WorldSelection.organism(uid: 5_000_000_100), .storm(id: 7), .kin(id: 17)].compactMap { Inspector.card(for: $0, store: full) }
        let inspectors = cards.map { height(InspectorCardView(card: $0, model: model), width: width) }
        print("measured: strip \(strip), storage strip \(storage), alert row \(row), tab picker \(picker), inspector cards \(inspectors) of \(cards.map(\.rows.count)) rows")
        #expect(cards.count == 3 && inspectors.allSatisfy { $0 <= DockBudget.inspector + 0.5 || $0 <= 340 })
        #expect(strip <= DockBudget.clockStrip, "the strip is \(strip) high; the budget says \(DockBudget.clockStrip)")
        #expect(storage <= DockBudget.storage, "the Storage strip is \(storage) high; the budget says \(DockBudget.storage)")
        #expect(row <= DockBudget.alertRow, "an alert row is \(row) high; the budget says \(DockBudget.alertRow)")
        #expect(row <= AlertRowView.height + 0.5)
        #expect(picker <= DockBudget.tabs, "the tabs are \(picker) high; the budget says \(DockBudget.tabs)")
        // Whole dock, inspector open, the most rows the budget allows for 700 points: it asks for no more than the window has.
        model.setSelection(.storm(id: 7))
        let rows = DockBudget.alertRows(dockHeight: 700, inspectorOpen: model.inspectorCard != nil)
        #expect(rows == 2)
        let dock = height(ControlCenterView(play: play, model: model, collapsed: .constant(false), side: .right, height: 700), width: width)
        print("dock fitting height: \(dock) with \(rows) alert rows and the inspector \(model.inspectorCard != nil)")
        #expect(dock <= 700, "the dock asks for \(dock) points of 700")
    }
}

import CoreGraphics
import Foundation
import Testing

@testable import AlifeCore

@Suite struct DockLayoutTests {
    private let window = CGSize(width: 1280, height: 800)

    @Test func theDockOnTheRightLeavesTheRestToTheMap() {
        let f = DockLayout.frames(in: window, side: .right, extent: 360, collapsed: false)
        #expect(f.free == CGRect(x: 0, y: 0, width: 920, height: 800))
        #expect(f.dock == CGRect(x: 920, y: 0, width: 360, height: 800))
    }

    @Test func theDockOnTheLeftAndAtTheBottom() {
        let l = DockLayout.frames(in: window, side: .left, extent: 360, collapsed: false)
        #expect(l.free == CGRect(x: 360, y: 0, width: 920, height: 800))
        #expect(l.dock == CGRect(x: 0, y: 0, width: 360, height: 800))
        let b = DockLayout.frames(in: window, side: .bottom, extent: 240, collapsed: false)
        #expect(b.free == CGRect(x: 0, y: 0, width: 1280, height: 560))
        #expect(b.dock == CGRect(x: 0, y: 560, width: 1280, height: 240))
    }

    @Test(arguments: DockSide.allCases, [CGFloat(200), 360, 900, 5000])
    func theTwoRectanglesTileTheWindowWithoutOverlap(side: DockSide, extent: CGFloat) {
        for collapsed in [false, true] {
            let f = DockLayout.frames(in: window, side: side, extent: extent, collapsed: collapsed)
            #expect(!f.free.intersects(f.dock) || f.free.intersection(f.dock).isEmpty
                    || f.free.intersection(f.dock).width * f.free.intersection(f.dock).height == 0)
            let area = f.free.width * f.free.height + f.dock.width * f.dock.height
            #expect(abs(area - window.width * window.height) < 1e-6)
            #expect(CGRect(origin: .zero, size: window).contains(f.free) && CGRect(origin: .zero, size: window).contains(f.dock))
        }
    }

    @Test func aCollapsedDockLeavesOnlyItsStrip() {
        let f = DockLayout.frames(in: window, side: .right, extent: 360, collapsed: true)
        #expect(f.dock.width == DockLayout.collapsedExtent)
        #expect(f.free.width == window.width - DockLayout.collapsedExtent)
        #expect(f.divider.isEmpty || f.divider.width == 0)
    }

    @Test func theExtentIsKeptWithinWhatTheWindowAllows() {
        // Never narrower than the minimum, never so wide that the map is left less than its minimum.
        #expect(DockLayout.clamped(extent: 10, side: .right, in: window) == DockLayout.minExtent)
        #expect(DockLayout.clamped(extent: 5000, side: .right, in: window) == window.width - DockLayout.minFree.width)
        #expect(DockLayout.clamped(extent: 5000, side: .bottom, in: window) == window.height - DockLayout.minFree.height)
        #expect(DockLayout.clamped(extent: 10, side: .bottom, in: window) == DockLayout.minBottomExtent)
        let tooWide = DockLayout.frames(in: window, side: .left, extent: 5000, collapsed: false)
        #expect(tooWide.free.width >= DockLayout.minFree.width)
    }

    @Test func theFreeRectangleHasItsOwnAspect() {
        let f = DockLayout.frames(in: window, side: .right, extent: 360, collapsed: false)
        #expect(abs(DockLayout.aspect(of: f.free) - 920.0 / 800) < 1e-6)
        // Changing the side changes the aspect the camera is fitted to.
        let b = DockLayout.frames(in: window, side: .bottom, extent: 240, collapsed: false)
        #expect(abs(DockLayout.aspect(of: b.free) - 1280.0 / 560) < 1e-6)
        #expect(DockLayout.aspect(of: .zero) == 1)
    }

    @Test func draggingTheDividerGivesTheExtentUnderThePointer() {
        #expect(DockLayout.extent(draggingTo: CGPoint(x: 900, y: 10), side: .right, in: window) == 380)
        #expect(DockLayout.extent(draggingTo: CGPoint(x: 400, y: 10), side: .left, in: window) == 400)
        #expect(DockLayout.extent(draggingTo: CGPoint(x: 10, y: 500), side: .bottom, in: window) == 300)
        // Dragged past the limits it stops there.
        #expect(DockLayout.extent(draggingTo: CGPoint(x: 1270, y: 0), side: .right, in: window) == DockLayout.minExtent)
        #expect(DockLayout.extent(draggingTo: CGPoint(x: 0, y: 0), side: .right, in: window)
                == window.width - DockLayout.minFree.width)
    }

    @Test func theDividerSitsOnTheInnerEdge() {
        let f = DockLayout.frames(in: window, side: .right, extent: 360, collapsed: false)
        #expect(f.divider.midX == f.dock.minX && f.divider.height == window.height)
        let b = DockLayout.frames(in: window, side: .bottom, extent: 240, collapsed: false)
        #expect(b.divider.midY == b.dock.minY && b.divider.width == window.width)
    }

    @Test func aTinyWindowStillGivesRectanglesInsideIt() {
        let tiny = CGSize(width: 300, height: 200)
        for side in DockSide.allCases {
            let f = DockLayout.frames(in: tiny, side: side, extent: 360, collapsed: false)
            #expect(f.free.width >= 0 && f.free.height >= 0 && f.dock.width >= 0 && f.dock.height >= 0)
            #expect(CGRect(origin: .zero, size: tiny).contains(f.dock))
        }
    }

    @Test func theDefaultsAreThoseOfTheSpec() {
        #expect(DockLayout.defaultExtent(for: .right) == 360 && DockLayout.defaultExtent(for: .left) == 360)
        #expect(DockLayout.defaultExtent(for: .bottom) == 240)
    }
}

@Suite struct DockBudgetTests {
    @Test func theMinimumWindowWithTheInspectorOpenShowsAtLeastOneRowAndAnEmptierDockShowsThree() {
        // The window is 1100 x 700 at least (DESIGN §34.7). The dock's fixed parts, the inspector card and the list's minimum
        // take what they take; the rows get the rest, never fewer than one and never more than three.
        #expect(DockBudget.alertRows(dockHeight: 700, inspectorOpen: true) == 2, "the spec's cap of two at the minimum window")
        #expect(DockBudget.alertRows(dockHeight: 650, inspectorOpen: true) == 1)
        #expect(DockBudget.alertRows(dockHeight: 700, inspectorOpen: false) == 3)
        // (the Storage strip, §34.10, took 26 points: three rows with the inspector open need 760 now, not 740)
        #expect(DockBudget.alertRows(dockHeight: 740, inspectorOpen: true) == 2)
        #expect(DockBudget.alertRows(dockHeight: 760, inspectorOpen: true) == 3)
        // the strip's second line (older saves wait for the next save) takes a row's room only while it shows
        #expect(DockBudget.alertRows(dockHeight: 700, inspectorOpen: true, storageNote: true) == 1)
        #expect(DockBudget.alertRows(dockHeight: 700, inspectorOpen: false, storageNote: true) == 3)
        #expect(DockBudget.storage >= 26 && DockBudget.storageNote >= 15)
        #expect(DockBudget.alertRows(dockHeight: 100, inspectorOpen: true) == 1, "an alert the player cannot see is not an alert")
        #expect(DockBudget.alertRows(dockHeight: 5000, inspectorOpen: false) == AlertStack.roomyRows)
    }

    @Test func theTabKeepsRoomForTheEditorWhileAPowerIsChosen() {
        // Third review, stage 8: with the inspector open and two alert rows the Powers tab was about 180 points, a slider and
        // nothing else. While a power is chosen the tab keeps `editor` points: the inspector card waits (the view hides it) and
        // the rows give way, never below one.
        #expect(DockBudget.alertRows(dockHeight: 700, inspectorOpen: false, editing: true) == 3)
        #expect(DockBudget.alertRows(dockHeight: 560, inspectorOpen: false, editing: true) == 1)
        #expect(DockBudget.alertRows(dockHeight: 700, inspectorOpen: false, editing: true)
            <= DockBudget.alertRows(dockHeight: 700, inspectorOpen: false), "never more rows than when nothing is being set up")
        for height in stride(from: CGFloat(560), through: 1200, by: 20) {
            let rows = DockBudget.alertRows(dockHeight: height, inspectorOpen: false, editing: true)
            let tab = height - DockBudget.header - DockBudget.clockStrip - DockBudget.storage - DockBudget.tabs
                - CGFloat(rows) * DockBudget.alertRow
            #expect(tab >= DockBudget.editor || rows == 1, "\(rows) rows leave the tab \(tab) of \(height)")
        }
        // the minimum window: 700 points, three rows (144) leave the tab 356, where the old budget with the inspector left 182
        let tab = 700 - DockBudget.header - DockBudget.clockStrip - DockBudget.storage - DockBudget.tabs - 3 * DockBudget.alertRow
        #expect(tab >= DockBudget.editor)
    }

    @Test func theRowsNeverOutgrowTheirBudget() {
        for height in stride(from: CGFloat(500), through: 1200, by: 25) {
            for open in [false, true] {
                let rows = DockBudget.alertRows(dockHeight: height, inspectorOpen: open)
                #expect((1...AlertStack.roomyRows).contains(rows))
                let used = DockBudget.header + DockBudget.clockStrip + DockBudget.tabs + DockBudget.list
                    + (open ? DockBudget.inspector : 0) + CGFloat(rows - 1) * DockBudget.alertRow
                #expect(used <= height || rows == 1, "\(rows) rows in \(height) (inspector \(open))")
            }
        }
    }
}

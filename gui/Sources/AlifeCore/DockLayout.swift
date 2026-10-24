import CoreGraphics
import Foundation

/// Where the control center docks.
package enum DockSide: String, CaseIterable, Sendable, Identifiable {
    case left, right, bottom

    package var id: String { rawValue }
}

/// The two rectangles of the World window: the free rectangle the Metal view draws into (the planet is centred in it,
/// the camera's aspect and every pixel rule are its) and the dock (DESIGN §33.4.4). The dock never overlaps the map. Pure
/// geometry, in the window's coordinates (origin at the top left, as SwiftUI's).
package enum DockLayout {
    package struct Frames: Equatable, Sendable {
        package var free: CGRect
        package var dock: CGRect
        /// The rectangle of the draggable divider along the dock's inner edge (empty when collapsed).
        package var divider: CGRect
    }

    /// The least a window leaves to the map: the dock gives way before it.
    package static let minFree = CGSize(width: 520, height: 360)
    /// The dock's extent (width on a side, height at the bottom) is kept within these.
    package static let minExtent: CGFloat = 280
    package static let defaultExtent: CGFloat = 360
    package static let defaultBottomExtent: CGFloat = 240
    /// The bottom dock's least height: its header and divider (29), and under them the clock column (the clock strip and the Storage
    /// strip, measured 151 in any clock state) and the tabs column (the picker and the list's least, 160); the recording switch is
    /// clipped below that (§34.10, `theBottomDocksClockColumnFitsItsLeastHeight`).
    package static let minBottomExtent: CGFloat = 192
    /// What is left of a collapsed dock: a strip with the button that opens it.
    package static let collapsedExtent: CGFloat = 28
    package static let dividerThickness: CGFloat = 6

    /// `extent` clamped to what `side` allows in a window of `size`.
    package static func clamped(extent: CGFloat, side: DockSide, in size: CGSize) -> CGFloat {
        switch side {
        case .left, .right:
            let upper = max(minExtent, size.width - minFree.width)
            return min(max(extent, minExtent), upper)
        case .bottom:
            let upper = max(minBottomExtent, size.height - minFree.height)
            return min(max(extent, minBottomExtent), upper)
        }
    }

    /// The default extent of a side.
    package static func defaultExtent(for side: DockSide) -> CGFloat {
        side == .bottom ? defaultBottomExtent : defaultExtent
    }

    package static func frames(in size: CGSize, side: DockSide, extent: CGFloat, collapsed: Bool) -> Frames {
        let width = max(size.width, 0), height = max(size.height, 0)
        let e = collapsed ? min(collapsedExtent, side == .bottom ? height : width)
            : min(clamped(extent: extent, side: side, in: size), side == .bottom ? height : width)
        let t = collapsed ? 0 : dividerThickness
        switch side {
        case .right:
            return Frames(free: CGRect(x: 0, y: 0, width: width - e, height: height),
                          dock: CGRect(x: width - e, y: 0, width: e, height: height),
                          divider: CGRect(x: width - e - t / 2, y: 0, width: t, height: collapsed ? 0 : height))
        case .left:
            return Frames(free: CGRect(x: e, y: 0, width: width - e, height: height),
                          dock: CGRect(x: 0, y: 0, width: e, height: height),
                          divider: CGRect(x: e - t / 2, y: 0, width: t, height: collapsed ? 0 : height))
        case .bottom:
            return Frames(free: CGRect(x: 0, y: 0, width: width, height: height - e),
                          dock: CGRect(x: 0, y: height - e, width: width, height: e),
                          divider: CGRect(x: 0, y: height - e - t / 2, width: collapsed ? 0 : width, height: t))
        }
    }

    /// How far along the dock's axis a grab at `point` is from the divider's centre (negative on the map's side): a divider
    /// grabbed at the edge of its strip stays under the pointer instead of jumping to centre on it.
    package static func grab(at point: CGPoint, divider: CGRect, side: DockSide) -> CGFloat {
        switch side {
        case .left, .right: return point.x - divider.midX
        case .bottom: return point.y - divider.midY
        }
    }

    /// The extent a drag of the divider to `point` (in the window's coordinates) asks for, for a divider that was grabbed
    /// `grab` points from its centre (`grab(at:divider:side:)`).
    package static func extent(draggingTo point: CGPoint, side: DockSide, in size: CGSize, grab: CGFloat = 0) -> CGFloat {
        let raw: CGFloat
        switch side {
        case .right: raw = size.width - (point.x - grab)
        case .left: raw = point.x - grab
        case .bottom: raw = size.height - (point.y - grab)
        }
        return clamped(extent: raw, side: side, in: size)
    }

    /// The aspect (width over height) of the free rectangle: the camera's.
    package static func aspect(of free: CGRect) -> Float {
        free.height >= 1 ? Float(free.width / free.height) : 1
    }
}

/// How much of a side dock the fixed parts take, measured through `NSHostingView` at the window's minimum size (the test
/// `theSideDockFitsTheMinimumWindow`), so that the alert rows give way before the list under the tabs does (DESIGN §34.7).
package enum DockBudget {
    /// The dock's header line, the clock strip with the chapter bar (measured: 125 with the 10 above it and the 4 below), the tab
    /// picker (40), the least the list keeps, and what the inspector card takes (measured: 261 for an organism, 189 for a storm, 126
    /// for a group; the dock caps it at 300 and scrolls it). With the Storage strip (§34.10) these are the measured figures plus
    /// one point: the strip took the room the old budget kept as slack (6 in the clock strip's padding, 9 in the inspector's).
    package static let header: CGFloat = 28
    package static let clockStrip: CGFloat = 126
    package static let tabs: CGFloat = 40
    package static let list: CGFloat = 120
    package static let inspector: CGFloat = 262
    /// What the Powers tab keeps while a power is chosen: the grid's first row, the sliders, the preview, its cautions and
    /// Confirm (the tab scrolls to the editor, but a tab of 180 points shows a slider and nothing else). The inspector card
    /// gives way to it (it is hidden while a power is being set up) and the alert rows before it.
    package static let editor: CGFloat = 300
    /// A row of the alert stack and the gap under it.
    package static let alertRow: CGFloat = 48
    /// The Storage strip under the clock strip (one line of 20 and 6 below it: the size and the recording switch; what the switch
    /// does is in its tooltip and the details popover, never in a second line that would cost the dock a row, §34.10).
    package static let storage: CGFloat = 26

    /// The rows of the stack a dock of `dockHeight` can show beside the strip, the tabs and (when one is selected) the
    /// inspector: three when roomy, fewer when tight, never none (an alert the player cannot see is not an alert). While a
    /// power is chosen (`editing`) the tab keeps `editor` points, not `list`.
    package static func alertRows(dockHeight: CGFloat, inspectorOpen: Bool, editing: Bool = false) -> Int {
        let fixed = header + clockStrip + storage + tabs + (editing ? editor : list) + (inspectorOpen ? inspector : 0)
        let fits = Int(((dockHeight - fixed) / alertRow).rounded(.down))
        return min(max(fits, 1), AlertStack.roomyRows)
    }
}

/// Whether the resize cursor is pushed for the dock's divider: while the pointer is on its strip or the strip is being dragged
/// (the strip follows the pointer, but may lag it), and popped exactly once for each push, also when the strip goes away under
/// the pointer (a collapsed dock). `onHover` of a view that is positioned by `.position` covers its whole parent, and a pop
/// without its push takes another view's cursor off the stack; this state is what keeps the two balanced.
package struct DividerCursor: Equatable, Sendable {
    package enum Action: Equatable, Sendable { case push, pop, none }

    package private(set) var hovering = false
    package private(set) var dragging = false
    package private(set) var pushed = false

    package init() {}

    package mutating func hover(_ inside: Bool) -> Action {
        hovering = inside
        return settle()
    }

    package mutating func drag(_ active: Bool) -> Action {
        dragging = active
        return settle()
    }

    /// The strip left the window.
    package mutating func disappear() -> Action {
        hovering = false
        dragging = false
        return settle()
    }

    private mutating func settle() -> Action {
        let wanted = hovering || dragging
        guard wanted != pushed else { return .none }
        pushed = wanted
        return wanted ? .push : .pop
    }
}

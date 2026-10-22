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
    package static let minBottomExtent: CGFloat = 160
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

    /// The extent a drag of the divider to `point` (in the window's coordinates) asks for.
    package static func extent(draggingTo point: CGPoint, side: DockSide, in size: CGSize) -> CGFloat {
        let raw: CGFloat
        switch side {
        case .right: raw = size.width - point.x
        case .left: raw = point.x
        case .bottom: raw = size.height - point.y
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
    /// The dock's header line, the clock strip with the chapter bar (measured: 131 with its padding), the tab picker (40), the least
    /// the list keeps, and what the inspector card takes (measured: 261 for an organism, 189 for a storm, 126 for a group; the dock
    /// caps it at 300 and scrolls it).
    package static let header: CGFloat = 28
    package static let clockStrip: CGFloat = 132
    package static let tabs: CGFloat = 40
    package static let list: CGFloat = 120
    package static let inspector: CGFloat = 270
    /// A row of the alert stack and the gap under it.
    package static let alertRow: CGFloat = 48

    /// The rows of the stack a dock of `dockHeight` can show beside the strip, the tabs and (when one is selected) the
    /// inspector: three when roomy, fewer when tight, never none (an alert the player cannot see is not an alert).
    package static func alertRows(dockHeight: CGFloat, inspectorOpen: Bool) -> Int {
        let fixed = header + clockStrip + tabs + list + (inspectorOpen ? inspector : 0)
        let fits = Int(((dockHeight - fixed) / alertRow).rounded(.down))
        return min(max(fits, 1), AlertStack.roomyRows)
    }
}

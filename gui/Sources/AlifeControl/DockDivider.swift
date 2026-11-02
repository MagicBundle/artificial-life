import AlifeCore
import AppKit
import SwiftUI

/// The draggable strip along a dock's inner edge (DESIGN §33.4.4), shared by the control center of a played world and the Run panel of a
/// research world: one divider, one extent per side (`world.dockExtent`, `world.dockExtentBottom`). The enclosing view names its coordinate
/// space "world".
///
/// It is placed with `.offset`, not `.position`: a positioned view takes its parent's whole size, and its hover and drag would then cover the
/// map under it (the cursor over the whole window, and clicks and scrolls taken from the Metal view). Here the hover and the drag are the 6
/// points of the strip and nothing else.
struct DockDivider: View {
    let rect: CGRect
    let side: DockSide
    let size: CGSize
    @AppStorage("world.dockExtent") private var sideExtent = Double(DockLayout.defaultExtent)
    @AppStorage("world.dockExtentBottom") private var bottomExtent = Double(DockLayout.defaultBottomExtent)
    @State private var dragStamp = Date.distantPast
    /// Where the divider was grabbed, from its centre (`DockLayout.grab`); nil while it is not being dragged.
    @State private var dragGrab: CGFloat?
    @State private var cursor = DividerCursor()

    var body: some View {
        Rectangle()
            .fill(Color.clear)
            .frame(width: rect.width, height: rect.height)
            .contentShape(Rectangle())
            .onHover { inside in apply(cursor.hover(inside)) }
            .gesture(
                DragGesture(minimumDistance: 0, coordinateSpace: .named("world"))
                    .onChanged { value in
                        if dragGrab == nil {
                            dragGrab = DockLayout.grab(at: value.startLocation, divider: rect, side: side)
                            apply(cursor.drag(true))
                        }
                        // The drawable resizes with each step of the divider: at most 30 a second.
                        guard Date().timeIntervalSince(dragStamp) >= 1.0 / 30 else { return }
                        dragStamp = Date()
                        setExtent(value.location, in: size)
                    }
                    .onEnded { value in
                        setExtent(value.location, in: size)  // the last step is never lost to the throttle
                        dragGrab = nil
                        apply(cursor.drag(false))
                    })
            .onDisappear { apply(cursor.disappear()) }
            .offset(x: rect.minX, y: rect.minY)
    }

    private func setExtent(_ location: CGPoint, in size: CGSize) {
        let e = Double(DockLayout.extent(draggingTo: location, side: side, in: size, grab: dragGrab ?? 0))
        if side == .bottom { bottomExtent = e } else { sideExtent = e }
    }

    private func apply(_ action: DividerCursor.Action) {
        switch action {
        case .push: (side == .bottom ? NSCursor.resizeUpDown : NSCursor.resizeLeftRight).push()
        case .pop: NSCursor.pop()
        case .none: break
        }
    }
}

import CoreGraphics

/// The launcher sheet's size rules (DESIGN §33.4.5). The sheet is a header, an optional starting box and the Resume list
/// pinned above and the form's footer pinned below a scrolling middle, so it fits the World window whatever the catalog
/// holds: its height is bounded, and what does not fit scrolls.
package enum LauncherLayout {
    /// The World window's minimum content height (`WorldView`).
    package static let windowMinHeight: CGFloat = 700
    /// The room a sheet needs inside its window: the title bar and the gap a sheet leaves above and below.
    package static let windowMargin: CGFloat = 60

    package static let minWidth: CGFloat = 640
    package static let minHeight: CGFloat = 400
    package static let idealHeight: CGFloat = 560
    /// Never taller than the World window's minimum less the margin, so the sheet fits it at any window size.
    package static let maxHeight: CGFloat = windowMinHeight - windowMargin

    /// Worlds listed before the Resume list scrolls.
    package static let resumeRowsShown = 3
    /// One world's row: the name, the caption under it, and the button.
    package static let resumeRowHeight: CGFloat = 42
    package static let resumeRowSpacing: CGFloat = 6

    /// The height of the Resume list for `count` worlds: all of them up to `resumeRowsShown`, then a window of that many.
    package static func resumeListHeight(count: Int) -> CGFloat {
        let rows = max(0, min(count, resumeRowsShown))
        guard rows > 0 else { return 0 }
        return CGFloat(rows) * resumeRowHeight + CGFloat(rows - 1) * resumeRowSpacing
    }

    package static func resumeListScrolls(count: Int) -> Bool { count > resumeRowsShown }
}

import CoreGraphics
import Foundation

/// What a press at a point of the free rectangle lands on (DESIGN §36.5, the spec's 5.3): the map, or one of the controls the overlay
/// draws over it. The controls are drawn by the Metal pass and the overlay `Canvas`, which take no mouse; the model hit-tests them
/// with `WorldHUD`, so no `NSView` and no hit-testing SwiftUI view lies over the map.
package enum HUDTarget: Equatable, Sendable {
    case map
    /// The square map of the whole torus, bottom right.
    case minimap
    /// The house glyph in the minimap's top left corner.
    case home
    /// The compass: a click turns the map north up, a double click is Home.
    case compass
    case zoomIn, zoomOut
}

/// Where the controls over the map are, in the points of the free rectangle (origin at the top left): the compass and the zoom
/// buttons stacked at the top right, the minimap at the bottom right with the Home glyph in its corner. Pure; the overlay draws
/// them where it says and the model hit-tests them with `target(at:)`.
package struct WorldHUD: Equatable, Sendable {
    package static let margin: CGFloat = 12
    /// The minimap's side is the least of this and `minimapFraction` of the free rectangle's shorter side.
    package static let minimapMaxSide: CGFloat = 220
    package static let minimapFraction: CGFloat = 0.28
    /// The minimap is hidden in a free rectangle shorter than this, and when its side would be under `minimapMinSide` (its labels do not fit).
    package static let minimapMinHeight: CGFloat = 160
    package static let minimapMinSide: CGFloat = 96
    package static let compassSide: CGFloat = 44
    package static let buttonSide: CGFloat = 30
    package static let buttonGap: CGFloat = 6
    /// The Home glyph's hit area, a square in the minimap's top left corner.
    package static let homeSide: CGFloat = 22
    /// The compass and the zoom buttons need this much room.
    package static let controlsMinHeight: CGFloat = 150
    package static let controlsMinWidth: CGFloat = 240

    package var free: CGSize
    package var minimap: CGRect?
    package var home: CGRect?
    package var compass: CGRect?
    package var zoomIn: CGRect?
    package var zoomOut: CGRect?

    /// The controls for a free rectangle of `free` points: `minimap` says whether the world has a map to show and the player has
    /// not turned it off, `compass` whether the world has a north (not a 2D world, which is seen from below with y down).
    package init(free: CGSize, minimap wantsMinimap: Bool, compass wantsCompass: Bool = true) {
        self.free = free
        let m = Self.margin
        let side = Self.minimapSide(free: free)
        if wantsMinimap, free.height >= Self.minimapMinHeight, free.width >= Self.minimapMinHeight, side >= Self.minimapMinSide {
            let box = CGRect(x: free.width - m - side, y: free.height - m - side, width: side, height: side)
            minimap = box
            home = CGRect(x: box.minX + 3, y: box.minY + 3, width: Self.homeSide, height: Self.homeSide)
        }
        if free.height >= Self.controlsMinHeight, free.width >= Self.controlsMinWidth {
            var y = m
            let centreX = free.width - m - Self.compassSide / 2
            if wantsCompass {
                compass = CGRect(x: free.width - m - Self.compassSide, y: y, width: Self.compassSide, height: Self.compassSide)
                y += Self.compassSide + Self.buttonGap + 2
            }
            let b = Self.buttonSide
            zoomIn = CGRect(x: centreX - b / 2, y: y, width: b, height: b)
            zoomOut = CGRect(x: centreX - b / 2, y: y + b + Self.buttonGap, width: b, height: b)
        }
    }

    /// The minimap's side in a free rectangle of `free` points.
    package static func minimapSide(free: CGSize) -> CGFloat {
        min(minimapMaxSide, minimapFraction * min(free.width, free.height)).rounded()
    }

    /// The control under `point`, or the map. The Home glyph is inside the minimap and wins there.
    package func target(at point: CGPoint) -> HUDTarget {
        if let home, home.contains(point) { return .home }
        if let minimap, minimap.contains(point) { return .minimap }
        if let compass, compass.contains(point) { return .compass }
        if let zoomIn, zoomIn.contains(point) { return .zoomIn }
        if let zoomOut, zoomOut.contains(point) { return .zoomOut }
        return .map
    }

    /// Every control's rectangle with its target, for the overlay and the tests.
    package var rects: [(target: HUDTarget, rect: CGRect)] {
        var out: [(HUDTarget, CGRect)] = []
        if let minimap { out.append((.minimap, minimap)) }
        if let home { out.append((.home, home)) }
        if let compass { out.append((.compass, compass)) }
        if let zoomIn { out.append((.zoomIn, zoomIn)) }
        if let zoomOut { out.append((.zoomOut, zoomOut)) }
        return out
    }

    /// The factor on the distance of one press of a zoom button (`-` farther, `+` closer): `e^(-+0.35)`.
    package static let zoomStep: Float = 0.35
}

/// The ease of a press on a zoom button: the distance goes to its target over 0.25 s, smoothly in its logarithm.
package struct ZoomEase: Equatable, Sendable {
    package static let duration: Float = 0.25
    package let from: Float
    package let to: Float
    package var elapsed: Float = 0

    package init(from: Float, to: Float) {
        self.from = max(from, 1e-6)
        self.to = max(to, 1e-6)
    }

    /// The distance `dt` seconds on, and whether the ease is over.
    package mutating func step(dt: Float) -> (distance: Float, done: Bool) {
        elapsed += max(dt, 0)
        let t = min(max(elapsed / Self.duration, 0), 1)
        let e = t * t * (3 - 2 * t)
        return (t >= 1 ? to : from * pow(to / from, e), t >= 1)
    }
}

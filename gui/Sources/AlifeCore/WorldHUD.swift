import CoreGraphics
import Foundation

/// What a press at a point of the free rectangle lands on (DESIGN §36.5, the spec's 5.3): the map, or one of the controls the overlay
/// draws over it. The controls are drawn by the Metal pass and the overlay `Canvas`, which take no mouse; the model hit-tests them
/// with `WorldHUD`, so no `NSView` and no hit-testing SwiftUI view lies over the map.
package enum HUDTarget: Hashable, Sendable {
    case map
    /// The square map of the whole torus, bottom right.
    case minimap
    /// The house glyph in the minimap's top left corner.
    case home
    /// The compass: a click turns the map north up, a double click is Home, a drag turns the view like a dial.
    case compass
    case zoomIn, zoomOut
    /// The 3D button: from the Top view to the angle last used (the Oblique one at first), and back.
    case toggle3D
    /// The tilt handle: a vertical track, straight down at its top and edge-on at its bottom.
    case tilt
    /// A standard view (`ViewPreset`) of the strip over the controls.
    case preset(ViewPreset)
}

/// A button of the strip of standard views.
package struct PresetRect: Equatable, Sendable {
    package var preset: ViewPreset
    package var rect: CGRect
}

/// Where the controls over the map are, in the points of the free rectangle (origin at the top left): a column down the right edge,
/// above the minimap (which is at the bottom right, with the Home glyph in its corner), from the top the strip of standard views and
/// the readout "Tilt 45° · heading 270°", the compass, the 3D button, the tilt handle, and the zoom buttons. Pure; the overlay draws
/// them where it says and the model hit-tests them with `target(at:)`.
///
/// The column is built from the bottom up and keeps what fits, in this order: the zoom buttons, the compass, the 3D button, the tilt
/// handle (from 48 to 96 points tall, as room allows), the strip and the readout. A world with no north (2D) has none of the
/// view controls but the zoom buttons.
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
    /// The column needs this much room.
    package static let controlsMinHeight: CGFloat = 150
    package static let controlsMinWidth: CGFloat = 240
    /// The column's top: under the clock badge of a collapsed dock; its bottom is above the minimap's coordinates tag.
    package static let columnTop: CGFloat = 52
    package static let columnBottomGap: CGFloat = 40
    package static let trackLength: ClosedRange<CGFloat> = 48...96
    package static let thumbDiameter: CGFloat = 20
    package static let stripHeight: CGFloat = 24
    package static let readoutHeight: CGFloat = 22
    /// The widths of the four segments of the strip, in the order of `ViewPreset.allCases`.
    package static let stripWidths: [CGFloat] = [38, 58, 38, 42]
    package static var stripWidth: CGFloat { stripWidths.reduce(0, +) }

    package var free: CGSize
    package var minimap: CGRect?
    package var home: CGRect?
    package var compass: CGRect?
    package var zoomIn: CGRect?
    package var zoomOut: CGRect?
    package var toggle3D: CGRect?
    /// The tilt handle's track (the thumb is drawn inside it).
    package var tiltTrack: CGRect?
    package var presets: [PresetRect] = []
    /// The plate of "Tilt 45° · heading 270°", above the strip. Drawn, not pressed.
    package var readout: CGRect?

    /// The controls for a free rectangle of `free` points: `minimap` says whether the world has a map to show and the player has
    /// not turned it off, `compass` whether the world has a north and a tilt (not a 2D world, which is seen from below with y down).
    package init(free: CGSize, minimap wantsMinimap: Bool, compass wantsCompass: Bool = true) {
        self.free = free
        let m = Self.margin
        let side = Self.minimapSide(free: free)
        if wantsMinimap, free.height >= Self.minimapMinHeight, free.width >= Self.minimapMinHeight, side >= Self.minimapMinSide {
            let box = CGRect(x: free.width - m - side, y: free.height - m - side, width: side, height: side)
            minimap = box
            home = CGRect(x: box.minX + 3, y: box.minY + 3, width: Self.homeSide, height: Self.homeSide)
        }
        guard free.height >= Self.controlsMinHeight, free.width >= Self.controlsMinWidth else { return }
        let right = free.width - m
        let centreX = right - Self.compassSide / 2
        let b = Self.buttonSide, gap = Self.buttonGap
        // The column is built from the bottom up: the zoom buttons first, then what fits of the rest.
        var y = minimap.map { $0.minY - Self.columnBottomGap } ?? free.height - m
        let top = Self.columnTop
        guard y - top >= 2 * b + gap else { return }
        zoomOut = CGRect(x: centreX - b / 2, y: y - b, width: b, height: b)
        zoomIn = CGRect(x: centreX - b / 2, y: y - 2 * b - gap, width: b, height: b)
        y -= 2 * b + 2 * gap
        guard wantsCompass else { return }
        // What fits above them, in the order of priority: the compass, the 3D button, the shortest track, the strip and the readout.
        // (`need` counts the heights and the gap above each.)
        let available = y - top
        var need: CGFloat = 0
        func fits(_ height: CGFloat) -> Bool { need + height <= available }
        let strip = Self.stripHeight + 4 + Self.readoutHeight
        var hasToggle = false, hasTrack = false, hasStrip = false
        guard fits(Self.compassSide) else { return }
        need += Self.compassSide + gap
        if fits(b) {
            hasToggle = true
            need += b + gap
            if fits(Self.trackLength.lowerBound) {
                hasTrack = true
                need += Self.trackLength.lowerBound + gap
                if fits(strip) {
                    hasStrip = true
                    need += strip + gap
                }
            }
        }
        let track = hasTrack ? min(Self.trackLength.lowerBound + max(available - need, 0), Self.trackLength.upperBound) : 0
        if hasTrack {
            tiltTrack = CGRect(x: centreX - b / 2, y: y - track, width: b, height: track)
            y -= track + gap
        }
        if hasToggle {
            toggle3D = CGRect(x: centreX - b / 2, y: y - b, width: b, height: b)
            y -= b + gap
        }
        compass = CGRect(x: right - Self.compassSide, y: y - Self.compassSide, width: Self.compassSide, height: Self.compassSide)
        y -= Self.compassSide + gap
        if hasStrip {
            var x = right - Self.stripWidth
            for (preset, width) in zip(ViewPreset.allCases, Self.stripWidths) {
                presets.append(PresetRect(preset: preset, rect: CGRect(x: x, y: y - Self.stripHeight, width: width, height: Self.stripHeight)))
                x += width
            }
            readout = CGRect(x: right - Self.stripWidth, y: y - strip, width: Self.stripWidth, height: Self.readoutHeight)
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
        for entry in presets where entry.rect.contains(point) { return .preset(entry.preset) }
        if let compass, compass.contains(point) { return .compass }
        if let toggle3D, toggle3D.contains(point) { return .toggle3D }
        if let tiltTrack, tiltTrack.insetBy(dx: -4, dy: -4).contains(point) { return .tilt }
        if let zoomIn, zoomIn.contains(point) { return .zoomIn }
        if let zoomOut, zoomOut.contains(point) { return .zoomOut }
        return .map
    }

    /// Every control's rectangle with its target, for the overlay and the tests. The readout is not here: it is not pressed.
    package var rects: [(target: HUDTarget, rect: CGRect)] {
        var out: [(HUDTarget, CGRect)] = []
        if let minimap { out.append((.minimap, minimap)) }
        if let home { out.append((.home, home)) }
        for entry in presets { out.append((.preset(entry.preset), entry.rect)) }
        if let compass { out.append((.compass, compass)) }
        if let toggle3D { out.append((.toggle3D, toggle3D)) }
        if let tiltTrack { out.append((.tilt, tiltTrack)) }
        if let zoomIn { out.append((.zoomIn, zoomIn)) }
        if let zoomOut { out.append((.zoomOut, zoomOut)) }
        return out
    }

    /// The free rectangle with the controls cut out, as a path to clip with: what the map's overlay draws (storms, rings, names) is
    /// kept off the minimap's box, the compass and the buttons, which are drawn over the picture and are not part of it.
    package var clearOfControls: CGPath {
        var path = CGPath(rect: CGRect(origin: .zero, size: free), transform: nil)
        for (target, rect) in rects where target != .home {  // (the Home glyph is inside the minimap's box)
            path = path.subtracting(CGPath(rect: rect, transform: nil))
        }
        if let readout { path = path.subtracting(CGPath(rect: readout, transform: nil)) }
        return path
    }

    /// The factor on the distance of one press of a zoom button (`-` farther, `+` closer): `e^(-+0.35)`.
    package static let zoomStep: Float = 0.35

    /// The tilt the handle shows for a pointer at `y` (points of the free rectangle) on a track of `track`, held with its thumb
    /// `grab` points from its centre: 0 at the top of the track's travel (straight down), `WorldPose.maxTilt` at the bottom.
    package static func tilt(atY y: CGFloat, track: CGRect, grab: CGFloat = 0) -> Float {
        let travel = max(track.height - thumbDiameter, 1)
        let f = (y - grab - (track.minY + thumbDiameter / 2)) / travel
        return Float(min(max(f, 0), 1)) * WorldPose.maxTilt
    }

    /// The thumb's centre for a `tilt` on a track of `track`.
    package static func thumbCentre(tilt: Float, track: CGRect) -> CGPoint {
        let travel = max(track.height - thumbDiameter, 1)
        let f = CGFloat(min(max(tilt / WorldPose.maxTilt, 0), 1))
        return CGPoint(x: track.midX, y: track.minY + thumbDiameter / 2 + travel * f)
    }
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

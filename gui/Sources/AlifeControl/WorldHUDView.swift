import AlifeCore
import SwiftUI

/// The orientation aids over the map, drawn in a `Canvas` that takes no mouse (DESIGN §36.5, the spec's 4): the readout of where the
/// view is, the column of view controls down the right edge (the strip of standard views and the readout of the angle, the compass, the
/// 3D button, the tilt handle, the zoom buttons), the minimap's frame, footprint, storms and labels over the box the
/// Metal pass drew, the scale bar with the size of an organism, and the legend of the height colour. The model hit-tests the
/// controls (`WorldModel.pressTarget`) from the same `WorldHUD` rectangles.
struct WorldHUDView: View {
    let model: WorldModel
    @ObservedObject var overlay: OverlayState
    @ObservedObject var play: PlayController

    var body: some View {
        let _ = overlay.revision
        Canvas { context, size in
            guard let header = play.store.header, size.width >= 1, size.height >= 1 else { return }
            let hud = WorldHUD(free: size, minimap: model.display.minimap && model.mapShown, compass: !model.pose.faceDown)
            let L = Float(header.size)
            drawReadout(L, &context)
            if let box = hud.minimap { drawMinimap(WorldInset(rect: box, size: L), hud: hud, size: size, &context) }
            drawHeightGuides(size, &context)
            drawControls(hud, &context)
            drawScale(size, &context)
            let heightLegend = model.display.heightColour && model.heightColourAvailable
            if heightLegend { drawLegend(height: Float(header.height), size, &context) }
            drawFeatureLegend(above: heightLegend, size, &context)
        }
        .allowsHitTesting(false)
    }

    // MARK: Pieces

    private static let ink = Color.white
    private static let gold = Color(red: 0.96, green: 0.78, blue: 0.25)
    private static let seam = Color(red: 0.92, green: 0.35, blue: 0.85)
    private static let plate = Color.black.opacity(0.58)

    private func text(_ string: String, size: CGFloat = 11, weight: Font.Weight = .medium, mono: Bool = false, opacity: Double = 0.92)
        -> Text {
        Text(string).font(.system(size: size, weight: weight, design: mono ? .monospaced : .default)).foregroundColor(Self.ink.opacity(opacity))
    }

    private func plate(_ rect: CGRect, radius: CGFloat = 6, _ context: inout GraphicsContext) {
        let path = Path(roundedRect: rect, cornerRadius: radius)
        context.fill(path, with: .color(Self.plate))
        context.stroke(path, with: .color(.white.opacity(0.14)), lineWidth: 0.5)
    }

    /// A line of text on a plate, its top left at `origin`; returns the plate's rectangle.
    @discardableResult
    private func plated(_ string: String, at origin: CGPoint, mono: Bool = false, _ context: inout GraphicsContext) -> CGRect {
        let resolved = context.resolve(text(string, mono: mono))
        let measured = resolved.measure(in: CGSize(width: 900, height: 40))
        let rect = CGRect(x: origin.x, y: origin.y, width: measured.width + 16, height: measured.height + 8)
        plate(rect, &context)
        context.draw(resolved, at: CGPoint(x: rect.minX + 8, y: rect.midY), anchor: .leading)
        return rect
    }

    // MARK: Where the view is

    /// "Northern hemisphere  39 N  235 E   205 units up", and what is under the pointer ("floor 270 E  28 N", or the organism's own place).
    private func drawReadout(_ L: Float, _ context: inout GraphicsContext) {
        var line = WorldReadout.line(pose: model.pose, size: L, height: model.height, mapShown: model.mapShown)
        if let pointer = overlay.pointerText { line += "    \(pointer)" }
        plated(line, at: CGPoint(x: WorldHUD.margin, y: 10), mono: true, &context)
    }

    // MARK: The height guides

    /// The floor, the middle and the ceiling of an edge-on view as faint dashed rows with their heights at the left: "height 80", "40", "0".
    private func drawHeightGuides(_ size: CGSize, _ context: inout GraphicsContext) {
        let guides = model.heightGuides
        for (index, guide) in guides.enumerated() {
            var row = Path()
            row.move(to: CGPoint(x: 0, y: guide.y))
            row.addLine(to: CGPoint(x: size.width, y: guide.y))
            context.stroke(row, with: .color(.white.opacity(0.22)), style: StrokeStyle(lineWidth: 0.7, dash: [3, 5]))
            let words = String(format: "%.0f", guide.height)
            let label = context.resolve(text(index == guides.count - 1 ? "height \(words)" : words, size: 10, mono: true, opacity: 0.8))
            let measured = label.measure(in: CGSize(width: 200, height: 20))
            let tag = CGRect(x: 6, y: guide.y - measured.height / 2 - 2, width: measured.width + 10, height: measured.height + 4)
            plate(tag, radius: 4, &context)
            context.draw(label, at: CGPoint(x: tag.minX + 5, y: tag.midY), anchor: .leading)
        }
    }

    // MARK: The minimap

    private func drawMinimap(_ inset: WorldInset, hud: WorldHUD, size: CGSize, _ context: inout GraphicsContext) {
        let box = inset.rect
        let aspect = Float(size.width / size.height)
        let pose = model.pose
        context.drawLayer { layer in
            layer.clip(to: Path(box))
            // The outside of the footprint is dimmed, so the part of the world in view is the bright one.
            // (at the globe the visible cap, or the border marched onto the planet in the morph: DESIGN §36.10)
            let vantage = model.vantage()
            let globePieces = inset.footprintPieces(of: vantage)
            if let pieces = globePieces ?? inset.footprint(of: pose, aspect: aspect).map({ inset.pieces(of: $0) }) {
                var dim = Path(box)
                for piece in pieces {
                    dim.move(to: piece[0])
                    piece.dropFirst().forEach { dim.addLine(to: $0) }
                    dim.closeSubpath()
                }
                layer.fill(dim, with: .color(.black.opacity(0.35)), style: FillStyle(eoFill: true))
                for piece in pieces {
                    var outline = Path()
                    outline.addLines(piece)
                    outline.closeSubpath()
                    layer.stroke(outline, with: .color(Color(red: 1, green: 0.88, blue: 0.2)), style: StrokeStyle(lineWidth: 2, lineJoin: .round))
                }
            }
            // The equator through the middle (gold), the wrap in y along the top and bottom edges (magenta, dashed).
            var equator = Path()
            equator.move(to: CGPoint(x: box.minX, y: box.midY))
            equator.addLine(to: CGPoint(x: box.maxX, y: box.midY))
            layer.stroke(equator, with: .color(Self.gold.opacity(0.75)), lineWidth: 1)
            var edges = Path()
            for y in [box.minY + 0.5, box.maxY - 0.5] {
                edges.move(to: CGPoint(x: box.minX, y: y))
                edges.addLine(to: CGPoint(x: box.maxX, y: y))
            }
            layer.stroke(edges, with: .color(Self.seam.opacity(0.9)), style: StrokeStyle(lineWidth: 1.2, dash: [4, 3]))
            // The storms, in the two hemispheres' colours, the selected one ringed.
            if let storms = play.store.storms?.storms.storms {
                for storm in storms {
                    let at = inset.point(forTorus: Float(storm.x), y: Float(storm.y))
                    let tint: Color = storm.hemi == 0 ? Color(red: 1, green: 0.85, blue: 0.4) : Color(red: 0.5, green: 0.85, blue: 1)
                    let dot = Path(ellipseIn: CGRect(x: at.x - 2.5, y: at.y - 2.5, width: 5, height: 5))
                    layer.fill(dot, with: .color(tint))
                    layer.stroke(dot, with: .color(.black.opacity(0.7)), lineWidth: 0.6)
                    if model.selection == .storm(id: storm.id) {
                        layer.stroke(Path(ellipseIn: CGRect(x: at.x - 6, y: at.y - 6, width: 12, height: 12)), with: .color(.white), lineWidth: 1.5)
                    }
                }
            }
            // The organism that is selected or followed.
            if let uid = model.selection.organismUID, let frame = play.store.frame,
               let index = play.store.listedIndex(ofUID: uid, hint: model.selectionHint) {
                let p = frame.position(index)
                let at = inset.point(forTorus: p.x, y: p.y)
                layer.stroke(Path(ellipseIn: CGRect(x: at.x - 3.5, y: at.y - 3.5, width: 7, height: 7)), with: .color(.white), lineWidth: 1.5)
            }
            // The centre of the picture.
            let centre = globePieces != nil ? inset.centre(of: vantage) : inset.centre(of: pose, aspect: aspect)
            let side = box.width
            let cx = box.minX + (centre.x - box.minX).truncatingRemainder(dividingBy: side), cy = box.minY + (centre.y - box.minY).truncatingRemainder(dividingBy: side)
            let wrapped = CGPoint(x: cx < box.minX ? cx + side : cx, y: cy < box.minY ? cy + side : cy)
            var cross = Path()
            cross.move(to: CGPoint(x: wrapped.x - 3, y: wrapped.y))
            cross.addLine(to: CGPoint(x: wrapped.x + 3, y: wrapped.y))
            cross.move(to: CGPoint(x: wrapped.x, y: wrapped.y - 3))
            cross.addLine(to: CGPoint(x: wrapped.x, y: wrapped.y + 3))
            layer.stroke(cross, with: .color(.white), lineWidth: 1.5)
            // The words, on a dark halo so they read over the sea and the cloud.
            layer.addFilter(.shadow(color: .black.opacity(0.95), radius: 1.6, x: 0, y: 0))
            let wide = side >= 170
            layer.draw(layer.resolve(text(wide ? "Northern hemisphere" : "North", size: 9, weight: .semibold)),
                       at: CGPoint(x: box.midX, y: box.minY + 8))
            layer.draw(layer.resolve(text(wide ? "Southern hemisphere" : "South", size: 9, weight: .semibold)),
                       at: CGPoint(x: box.midX, y: box.maxY - 8))
            layer.draw(layer.resolve(text("W", size: 9, weight: .semibold)), at: CGPoint(x: box.minX + 7, y: box.midY + 9))
            layer.draw(layer.resolve(text("E", size: 9, weight: .semibold)), at: CGPoint(x: box.maxX - 7, y: box.midY + 9))
        }
        context.stroke(Path(box), with: .color(.white.opacity(0.7)), lineWidth: 1)
        // The Home glyph in the corner.
        if let home = hud.home { drawGlyph(home, "house.fill", target: .home, &context) }
        // Where: the point under the pointer when it is over the box, else the centre of the view.
        var where_ = WorldReadout.coordinates(x: pose.x, y: pose.y, size: inset.size)
        if overlay.hudHover == .minimap, let place = overlay.minimapPoint {
            where_ = WorldReadout.coordinates(x: place.x, y: place.y, size: inset.size)
        }
        let resolved = context.resolve(text(where_, mono: true))
        let measured = resolved.measure(in: CGSize(width: 400, height: 40))
        let tag = CGRect(x: box.maxX - measured.width - 16, y: box.minY - measured.height - 12, width: measured.width + 16, height: measured.height + 6)
        plate(tag, radius: 5, &context)
        context.draw(resolved, at: CGPoint(x: tag.minX + 8, y: tag.midY), anchor: .leading)
    }

    // MARK: Controls

    private func drawGlyph(_ rect: CGRect, _ symbol: String, target: HUDTarget, _ context: inout GraphicsContext) {
        let lit = overlay.pressed == target || overlay.hudHover == target
        let path = Path(roundedRect: rect, cornerRadius: 5)
        context.fill(path, with: .color(Color.black.opacity(0.62)))
        if lit { context.fill(path, with: .color(.white.opacity(overlay.pressed == target ? 0.3 : 0.16))) }
        context.stroke(path, with: .color(.white.opacity(0.3)), lineWidth: 0.5)
        var image = context.resolve(Image(systemName: symbol))
        image.shading = .color(.white.opacity(0.95))
        context.draw(image, in: rect.insetBy(dx: rect.width * 0.2, dy: rect.height * 0.2))
    }

    /// A rounded button's plate, lit while the pointer is on it and brighter while it is pressed.
    private func buttonPlate(_ rect: CGRect, target: HUDTarget, radius: CGFloat = 7, _ context: inout GraphicsContext) {
        let lit = overlay.pressed == target || overlay.hudHover == target
        let path = Path(roundedRect: rect, cornerRadius: radius)
        context.fill(path, with: .color(Color.black.opacity(0.62)))
        if lit { context.fill(path, with: .color(.white.opacity(overlay.pressed == target ? 0.3 : 0.16))) }
        context.stroke(path, with: .color(.white.opacity(0.3)), lineWidth: 0.5)
    }

    private func drawControls(_ hud: WorldHUD, _ context: inout GraphicsContext) {
        if let readout = hud.readout { drawAngle(readout, hud, &context) }
        if let compass = hud.compass { drawCompass(compass, &context) }
        if let toggle = hud.toggle3D {
            buttonPlate(toggle, target: .toggle3D, &context)
            let flat = model.pose.tilt < 0.05
            context.draw(context.resolve(text(flat ? "3D" : "2D", size: 12, weight: .bold)), at: CGPoint(x: toggle.midX, y: toggle.midY))
        }
        if let track = hud.tiltTrack { drawTilt(track, &context) }
        for (target, rect, plus) in [(HUDTarget.zoomIn, hud.zoomIn, true), (.zoomOut, hud.zoomOut, false)] {
            guard let rect else { continue }
            buttonPlate(rect, target: target, &context)
            var sign = Path()
            sign.move(to: CGPoint(x: rect.midX - 6, y: rect.midY))
            sign.addLine(to: CGPoint(x: rect.midX + 6, y: rect.midY))
            if plus {
                sign.move(to: CGPoint(x: rect.midX, y: rect.midY - 6))
                sign.addLine(to: CGPoint(x: rect.midX, y: rect.midY + 6))
            }
            context.stroke(sign, with: .color(.white.opacity(0.95)), style: StrokeStyle(lineWidth: 2, lineCap: .round))
        }
        // The words of the control under the pointer, to its left (for a view of the strip: to the left of the strip).
        let hovered = overlay.hudHover
        var rect: CGRect?
        let words: String
        switch hovered {
        case .compass: (rect, words) = (hud.compass, "North up (N)   double click: Home   drag: turn the view")
        case .zoomIn: (rect, words) = (hud.zoomIn, "Zoom in (pinch, mouse wheel)")
        case .zoomOut: (rect, words) = (hud.zoomOut, "Zoom out")
        case .home: (rect, words) = (hud.home, "Home view (H)")
        case .toggle3D: (rect, words) = (hud.toggle3D, model.pose.tilt < 0.05 ? "Tilt the view: 3D" : "Straight down: 2D")
        case .tilt: (rect, words) = (hud.tiltTrack, "Tilt: drag. Straight down at the top, edge-on at the bottom")
        case .preset(let preset):
            if let entry = hud.presets.first(where: { $0.preset == preset }), let first = hud.presets.first {
                rect = CGRect(x: first.rect.minX, y: entry.rect.minY, width: entry.rect.maxX - first.rect.minX, height: entry.rect.height)
            }
            words = "\(preset.title) (\(preset.key.uppercased())): \(preset.help)"
        default: (rect, words) = (nil, "")
        }
        if let rect {
            let resolved = context.resolve(text(words, size: 11))
            let measured = resolved.measure(in: CGSize(width: 400, height: 40))
            let tag = hovered == .home
                ? CGRect(x: rect.maxX + 6, y: rect.midY - (measured.height + 6) / 2, width: measured.width + 14, height: measured.height + 6)
                : CGRect(x: rect.minX - measured.width - 22, y: rect.midY - (measured.height + 6) / 2, width: measured.width + 14, height: measured.height + 6)
            plate(tag, radius: 5, &context)
            context.draw(resolved, at: CGPoint(x: tag.minX + 7, y: tag.midY), anchor: .leading)
        }
    }

    /// The readout of the angle over the strip of standard views, and the strip: the view whose angle the tilt is at is lit.
    private func drawAngle(_ readout: CGRect, _ hud: WorldHUD, _ context: inout GraphicsContext) {
        plate(readout, radius: 6, &context)
        context.draw(context.resolve(text(WorldReadout.viewText(pose: model.pose), size: 11, mono: true)),
                     at: CGPoint(x: readout.midX, y: readout.midY))
        guard let first = hud.presets.first, let last = hud.presets.last else { return }
        let strip = CGRect(x: first.rect.minX, y: first.rect.minY, width: last.rect.maxX - first.rect.minX, height: first.rect.height)
        let outline = Path(roundedRect: strip, cornerRadius: 7)
        context.fill(outline, with: .color(Color.black.opacity(0.62)))
        let tilt = model.pose.tilt
        for entry in hud.presets {
            let target = HUDTarget.preset(entry.preset)
            let lit = overlay.pressed == target || overlay.hudHover == target
            let current = abs(entry.preset.tilt - tilt) < 1.5 * .pi / 180
            let inner = entry.rect.insetBy(dx: 2, dy: 2)
            if current { context.fill(Path(roundedRect: inner, cornerRadius: 5), with: .color(.white.opacity(0.3))) }
            if lit { context.fill(Path(roundedRect: inner, cornerRadius: 5), with: .color(.white.opacity(overlay.pressed == target ? 0.3 : 0.16))) }
            context.draw(context.resolve(text(entry.preset.title, size: 10.5, weight: current ? .bold : .medium)),
                         at: CGPoint(x: entry.rect.midX, y: entry.rect.midY))
        }
        context.stroke(outline, with: .color(.white.opacity(0.3)), lineWidth: 0.5)
    }

    /// The tilt handle: a track from straight down (top) to edge-on (bottom), ticks at the Oblique and the Low views, and the thumb.
    private func drawTilt(_ track: CGRect, _ context: inout GraphicsContext) {
        let lit = overlay.pressed == .tilt || overlay.hudHover == .tilt
        let plateRect = Path(roundedRect: track, cornerRadius: 15)
        context.fill(plateRect, with: .color(Color.black.opacity(0.62)))
        if lit { context.fill(plateRect, with: .color(.white.opacity(overlay.pressed == .tilt ? 0.2 : 0.1))) }
        context.stroke(plateRect, with: .color(.white.opacity(0.3)), lineWidth: 0.5)
        let top = WorldHUD.thumbCentre(tilt: 0, track: track), bottom = WorldHUD.thumbCentre(tilt: WorldPose.maxTilt, track: track)
        var groove = Path()
        groove.move(to: top)
        groove.addLine(to: bottom)
        context.stroke(groove, with: .color(.white.opacity(0.28)), style: StrokeStyle(lineWidth: 3, lineCap: .round))
        for preset in [ViewPreset.oblique, .low] {
            let at = WorldHUD.thumbCentre(tilt: preset.tilt, track: track)
            var tick = Path()
            tick.move(to: CGPoint(x: track.midX - 6, y: at.y))
            tick.addLine(to: CGPoint(x: track.midX + 6, y: at.y))
            context.stroke(tick, with: .color(.white.opacity(0.5)), lineWidth: 1)
        }
        let thumb = WorldHUD.thumbCentre(tilt: model.pose.tilt, track: track)
        let r = WorldHUD.thumbDiameter / 2
        let disc = Path(ellipseIn: CGRect(x: thumb.x - r, y: thumb.y - r, width: 2 * r, height: 2 * r))
        context.fill(disc, with: .color(.white.opacity(overlay.pressed == .tilt ? 1 : 0.9)))
        context.stroke(disc, with: .color(.black.opacity(0.5)), lineWidth: 0.8)
    }

    /// A disc with a needle that turns with the map: north in red, south pale, and an "N" on the north side.
    private func drawCompass(_ rect: CGRect, _ context: inout GraphicsContext) {
        let lit = overlay.pressed == .compass || overlay.hudHover == .compass
        let disc = Path(ellipseIn: rect)
        context.fill(disc, with: .color(Color.black.opacity(0.62)))
        if lit { context.fill(disc, with: .color(.white.opacity(overlay.pressed == .compass ? 0.28 : 0.14))) }
        context.stroke(disc, with: .color(.white.opacity(0.3)), lineWidth: 0.5)
        let angle = CGFloat(Compass.needleAngle(heading: model.pose.heading, tilt: model.pose.tilt))
        let c = CGPoint(x: rect.midX, y: rect.midY)
        let direction = CGPoint(x: sin(angle), y: -cos(angle))  // north on the screen (y down)
        let across = CGPoint(x: -direction.y, y: direction.x)
        func at(_ along: CGFloat, _ side: CGFloat) -> CGPoint {
            CGPoint(x: c.x + direction.x * along + across.x * side, y: c.y + direction.y * along + across.y * side)
        }
        var north = Path()
        north.move(to: at(10, 0))
        north.addLine(to: at(0, 3.6))
        north.addLine(to: at(0, -3.6))
        north.closeSubpath()
        context.fill(north, with: .color(Color(red: 1, green: 0.4, blue: 0.3)))
        var south = Path()
        south.move(to: at(-10, 0))
        south.addLine(to: at(0, 3.6))
        south.addLine(to: at(0, -3.6))
        south.closeSubpath()
        context.fill(south, with: .color(.white.opacity(0.78)))
        context.draw(context.resolve(text("N", size: 9, weight: .bold)), at: at(16.5, 0))
    }

    // MARK: Scale

    /// The scale plate's height, and its distance from the bottom of the free rectangle.
    static let scalePlateHeight: CGFloat = 58
    static let scalePlateBottomInset: CGFloat = 46
    /// The longest an organism's segment is drawn, in points (it is wider only when the view is zoomed in past a few units).
    private static let organismRoom: CGFloat = 220

    /// A bar of a round number of world units, true on the plane it names (the floor, unless an organism is followed), and under it
    /// the organism as a segment on the same scale: its diameter as long as it is at this scale, never clamped, with what the size
    /// policy draws it as in words ("drawn x0.1").
    private func drawScale(_ size: CGSize, _ context: inout GraphicsContext) {
        guard let scale = model.scaleBar else { return }
        let bar = scale.bar
        let label = context.resolve(text(bar.label(on: scale.plane), size: 11, mono: true))
        let labelSize = label.measure(in: CGSize(width: 400, height: 30))
        var organismText: GraphicsContext.ResolvedText?
        var organismSize = CGSize.zero
        var segment: (points: CGFloat, clipped: Bool) = (0, false)
        if let organism = scale.organism {
            organismText = context.resolve(text(organism.label, size: 10, opacity: 0.82))
            organismSize = organismText?.measure(in: CGSize(width: 400, height: 30)) ?? .zero
            segment = organism.segment(room: Self.organismRoom)
        }
        let rowWidth = max(bar.points, segment.points + (organismText == nil ? 0 : 8 + organismSize.width))
        let plateRect = CGRect(x: 8, y: size.height - Self.scalePlateBottomInset - Self.scalePlateHeight,
                               width: 16 + max(rowWidth, labelSize.width), height: Self.scalePlateHeight)
        plate(plateRect, radius: 6, &context)
        let x0 = plateRect.minX + 8
        context.draw(label, at: CGPoint(x: x0, y: plateRect.minY + 4 + labelSize.height / 2), anchor: .leading)
        let barY = plateRect.minY + 4 + labelSize.height + 9
        func ticked(from x: CGFloat, length: CGFloat, at y: CGFloat, half: CGFloat) -> Path {
            var line = Path()
            line.move(to: CGPoint(x: x, y: y))
            line.addLine(to: CGPoint(x: x + length, y: y))
            for end in [x, x + length] {
                line.move(to: CGPoint(x: end, y: y - half))
                line.addLine(to: CGPoint(x: end, y: y + half))
            }
            return line
        }
        context.stroke(ticked(from: x0, length: bar.points, at: barY, half: 4), with: .color(.white.opacity(0.95)), lineWidth: 1.5)
        if let organismText {
            // The organism: a segment of its diameter on the bar's own scale (a break in it where it is wider than there is room for).
            let y = barY + 15
            let tint = Color(red: 1, green: 0.88, blue: 0.2)
            if segment.clipped {
                var broken = ticked(from: x0, length: segment.points, at: y, half: 3)
                for gap in [x0 + segment.points * 0.5, x0 + segment.points * 0.5 + 5] {
                    broken.move(to: CGPoint(x: gap - 2, y: y + 4))
                    broken.addLine(to: CGPoint(x: gap + 2, y: y - 4))
                }
                context.stroke(broken, with: .color(tint), lineWidth: 1.5)
            } else {
                context.stroke(ticked(from: x0, length: segment.points, at: y, half: 3), with: .color(tint), lineWidth: 2)
            }
            context.draw(organismText, at: CGPoint(x: x0 + segment.points + 8, y: y), anchor: .leading)
        }
    }

    // MARK: Legend

    /// The height colour's ramp, low at the bottom, with the heights at its ticks.
    /// The legend's colour bar in a free rectangle of `size`: above the scale bar, at the left edge.
    static func legendBar(in size: CGSize) -> CGRect {
        CGRect(x: 20, y: size.height - scalePlateBottomInset - scalePlateHeight - 14 - 104, width: 12, height: 104)
    }

    /// The legend of the feature layers in view (`FeatureLegend`): a swatch and a name per row, stacked up from the left edge above the scale bar
    /// (above the height legend when that is shown too).
    private func drawFeatureLegend(above heightLegend: Bool, _ size: CGSize, _ context: inout GraphicsContext) {
        let entries = FeatureLegend.entries(layers: play.store.layers, display: model.display, colourMode: model.currentColourMode)
        guard !entries.isEmpty else { return }
        let bar = Self.legendBar(in: size)
        let rowHeight: CGFloat = 16
        let height = CGFloat(entries.count) * rowHeight + 12
        let bottom = (heightLegend ? bar.minY - 22 - 8 : bar.maxY + 8)
        let plateRect = CGRect(x: 8, y: bottom - height, width: 132, height: height)
        plate(plateRect, radius: 6, &context)
        for (row, entry) in entries.enumerated() {
            let y = plateRect.minY + 6 + CGFloat(row) * rowHeight + rowHeight / 2
            let swatch = CGRect(x: plateRect.minX + 8, y: y - 5, width: 10, height: 10)
            context.fill(Path(roundedRect: swatch, cornerRadius: 2),
                         with: .color(Color(red: Double(entry.colour.x), green: Double(entry.colour.y), blue: Double(entry.colour.z))))
            context.stroke(Path(roundedRect: swatch, cornerRadius: 2), with: .color(.white.opacity(0.5)), lineWidth: 0.5)
            context.draw(context.resolve(text(entry.title, size: 10)), at: CGPoint(x: swatch.maxX + 6, y: y), anchor: .leading)
        }
    }

    private func drawLegend(height: Float, _ size: CGSize, _ context: inout GraphicsContext) {
        let bar = Self.legendBar(in: size)
        let plateRect = CGRect(x: 8, y: bar.minY - 22, width: 74, height: bar.height + 36)
        plate(plateRect, radius: 6, &context)
        let stops = HeightColour.stops.enumerated().map { i, c in
            Gradient.Stop(color: Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z)), location: CGFloat(i) / CGFloat(HeightColour.stops.count - 1))
        }
        context.fill(Path(bar), with: .linearGradient(Gradient(stops: stops), startPoint: CGPoint(x: bar.midX, y: bar.maxY), endPoint: CGPoint(x: bar.midX, y: bar.minY)))
        context.stroke(Path(bar), with: .color(.white.opacity(0.6)), lineWidth: 0.5)
        context.draw(context.resolve(text("height", size: 10, weight: .semibold)), at: CGPoint(x: plateRect.minX + 12, y: plateRect.minY + 10), anchor: .leading)
        for tick in HeightColour.ticks(height: height) {
            let y = bar.maxY - CGFloat(tick.fraction) * bar.height
            var mark = Path()
            mark.move(to: CGPoint(x: bar.maxX, y: y))
            mark.addLine(to: CGPoint(x: bar.maxX + 4, y: y))
            context.stroke(mark, with: .color(.white.opacity(0.9)), lineWidth: 1)
            context.draw(context.resolve(text(tick.label, size: 10, mono: true)), at: CGPoint(x: bar.maxX + 8, y: y), anchor: .leading)
        }
    }
}

/// The one-line hint of the controls, once on the first World window and again from View > Controls…; any input takes it away.
struct ControlsHintView: View {
    @ObservedObject var model: WorldModel

    var body: some View {
        if model.hintVisible {
            // At the top, under the readout and between it and the compass: the bottom corners have the scale bar and the minimap, which a
            // line there lay over in a narrow window.
            Text(WorldModel.hintText)
                .font(.callout)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.primary.opacity(0.12)))
                .frame(maxWidth: 560)
                .padding(.horizontal, WorldHUD.compassSide + WorldHUD.margin + 20)
                .padding(.top, 46)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .allowsHitTesting(false)
                .transition(.opacity)
        }
    }
}

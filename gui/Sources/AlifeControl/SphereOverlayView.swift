import AlifeCore
import SwiftUI
import simd

/// What the planet's view shows over the Metal picture (DESIGN §46.8): the names of the storms, the tooltip, the plate of the clicked cell,
/// the aim of a power, and the legend of the height colour; and the controls the player presses, drawn in one `Canvas` that takes no mouse
/// (`SphereController` hit-tests the same `WorldHUD` rectangles): the readout, the minimap (the Mollweide chart of the whole planet with the
/// footprint of the view, the organisms and the storms), the compass, the zoom buttons, the 3D button, the tilt handle and the strip of
/// standard views.
struct SphereOverlayView: View {
    let model: WorldModel
    @ObservedObject var overlay: OverlayState
    @ObservedObject var play: PlayController

    private static let plate = Color.black.opacity(0.58)
    private static let gold = Color(red: 0.96, green: 0.78, blue: 0.25)

    var body: some View {
        let _ = overlay.revision
        let sphere = model.sphere
        Canvas { context, size in
            guard sphere.isActive, size.width >= 1, size.height >= 1 else { return }
            let hud = sphere.hud
            drawStorms(sphere, size, hud, &context)
            drawAim(sphere, &context)
            drawReadout(sphere, &context)
            if let box = hud.minimap { drawMinimap(sphere, box, &context) }
            drawControls(sphere, hud, &context)
            drawCell(sphere, size, &context)
            drawHover(sphere, size, &context)
            if model.display.heightColour, model.heightColourAvailable, let header = play.store.header {
                drawLegend(height: Float(header.height), size, &context)
            }
        }
        .allowsHitTesting(false)
    }

    // MARK: Text and plates

    private func text(_ string: String, size: CGFloat = 11, weight: Font.Weight = .medium, mono: Bool = false, opacity: Double = 0.92) -> Text {
        Text(string).font(.system(size: size, weight: weight, design: mono ? .monospaced : .default)).foregroundColor(.white.opacity(opacity))
    }

    private func plate(_ rect: CGRect, radius: CGFloat = 6, _ context: inout GraphicsContext) {
        let path = Path(roundedRect: rect, cornerRadius: radius)
        context.fill(path, with: .color(Self.plate))
        context.stroke(path, with: .color(.white.opacity(0.14)), lineWidth: 0.5)
    }

    @discardableResult
    private func plated(_ string: String, at origin: CGPoint, mono: Bool = false, maxWidth: CGFloat = 900, _ context: inout GraphicsContext) -> CGRect {
        let resolved = context.resolve(text(string, mono: mono))
        let measured = resolved.measure(in: CGSize(width: maxWidth, height: 60))
        let rect = CGRect(x: origin.x, y: origin.y, width: measured.width + 16, height: measured.height + 8)
        plate(rect, &context)
        context.draw(resolved, at: CGPoint(x: rect.minX + 8, y: rect.midY), anchor: .leading)
        return rect
    }

    // MARK: The readout

    private func drawReadout(_ sphere: SphereController, _ context: inout GraphicsContext) {
        var line = sphere.readout
        if let pointer = overlay.pointerText { line += "    \(pointer)" }
        plated(line, at: CGPoint(x: WorldHUD.margin, y: 10), mono: true, &context)
    }

    // MARK: Storms

    private func drawStorms(_ sphere: SphereController, _ size: CGSize, _ hud: WorldHUD, _ context: inout GraphicsContext) {
        guard let header = play.store.header?.sphere, let storms = play.store.storms?.storms.storms else { return }
        let rp = Float(header.planetRadius)
        context.drawLayer { layer in
            layer.clip(to: Path(hud.clearOfControls), style: FillStyle(eoFill: false))
            for storm in storms {
                guard let centre = SphereOverlay.direction(of: storm) else { continue }
                let p = centre * rp
                guard !sphere.camera.isOccluded(p), let at = SpherePicker.screen(p, camera: sphere.camera, view: size) else { continue }
                guard at.x > -20, at.y > -20, at.x < size.width + 20, at.y < size.height + 20 else { continue }
                let selected = model.selection == .storm(id: storm.id)
                let pointsPerUnit = SpherePicker.pointsPerUnit(depth: simd_dot(p - sphere.camera.eye, sphere.camera.forward), camera: sphere.camera, view: size)
                let ring = CGFloat(Float(storm.r) * pointsPerUnit)
                // the name: under the ring when it is big enough to read, beside the point when small
                guard storm.named || selected || ring > 14 else { continue }
                let title = Names.storm(storm)
                let resolved = layer.resolve(text(title, size: 11, weight: selected ? .bold : .medium, opacity: selected ? 1 : 0.88))
                let measured = resolved.measure(in: CGSize(width: 300, height: 30))
                let y = at.y + min(ring, 90) + 10
                let tag = CGRect(x: at.x - measured.width / 2 - 6, y: y - measured.height / 2 - 2, width: measured.width + 12, height: measured.height + 4)
                plate(tag, radius: 4, &layer)
                layer.draw(resolved, at: CGPoint(x: tag.midX, y: tag.midY))
            }
        }
    }

    // MARK: The aim of a power

    private func drawAim(_ sphere: SphereController, _ context: inout GraphicsContext) {
        guard let point = overlay.aimPoint, let world = overlay.aimWorld else { return }
        let r: CGFloat = 14
        let ring = Path(ellipseIn: CGRect(x: point.x - r, y: point.y - r, width: 2 * r, height: 2 * r))
        context.stroke(ring, with: .color(Self.gold), lineWidth: 2)
        var cross = Path()
        cross.move(to: CGPoint(x: point.x - r - 5, y: point.y))
        cross.addLine(to: CGPoint(x: point.x + r + 5, y: point.y))
        cross.move(to: CGPoint(x: point.x, y: point.y - r - 5))
        cross.addLine(to: CGPoint(x: point.x, y: point.y + r + 5))
        context.stroke(cross, with: .color(Self.gold.opacity(0.7)), lineWidth: 1)
        let words = SphereFrame.placeText(SphereFrame.direction(lat: Double(world.y), lon: Double(world.x)))
        plated(words, at: CGPoint(x: point.x + r + 8, y: point.y - 10), mono: true, &context)
    }

    // MARK: The clicked cell, the tooltip

    private func drawCell(_ sphere: SphereController, _ size: CGSize, _ context: inout GraphicsContext) {
        guard let words = sphere.cellText else { return }
        let resolved = context.resolve(text(words, size: 11))
        let measured = resolved.measure(in: CGSize(width: min(size.width - 40, 560), height: 80))
        let rect = CGRect(x: WorldHUD.margin, y: size.height - WorldHUD.margin - measured.height - 12, width: measured.width + 18, height: measured.height + 12)
        plate(rect, &context)
        context.draw(resolved, at: CGPoint(x: rect.minX + 9, y: rect.midY), anchor: .leading)
    }

    private func drawHover(_ sphere: SphereController, _ size: CGSize, _ context: inout GraphicsContext) {
        guard let hover = sphere.hover else { return }
        let resolved = context.resolve(text(hover.text, size: 11))
        let measured = resolved.measure(in: CGSize(width: 360, height: 40))
        var x = hover.point.x + 14, y = hover.point.y + 16
        if x + measured.width + 16 > size.width { x = hover.point.x - measured.width - 24 }
        if y + measured.height + 8 > size.height { y = hover.point.y - measured.height - 20 }
        let rect = CGRect(x: x, y: y, width: measured.width + 14, height: measured.height + 8)
        plate(rect, radius: 5, &context)
        context.draw(resolved, at: CGPoint(x: rect.minX + 7, y: rect.midY), anchor: .leading)
    }

    // MARK: The minimap

    private func drawMinimap(_ sphere: SphereController, _ box: CGRect, _ context: inout GraphicsContext) {
        let _ = sphere.minimapSerial
        let ellipse = Path(ellipseIn: box)
        context.fill(ellipse, with: .color(.black.opacity(0.55)))
        guard let header = play.store.header?.sphere else { return }
        context.drawLayer { layer in
            layer.clip(to: ellipse)
            if let image = sphere.minimap {
                layer.draw(Image(decorative: image, scale: 1), in: box)
            }
            // the equator and the prime meridian, faint
            var grid = Path()
            grid.move(to: CGPoint(x: box.minX, y: box.midY))
            grid.addLine(to: CGPoint(x: box.maxX, y: box.midY))
            grid.move(to: CGPoint(x: box.midX, y: box.minY))
            grid.addLine(to: CGPoint(x: box.midX, y: box.maxY))
            layer.stroke(grid, with: .color(.white.opacity(0.18)), lineWidth: 0.5)
            // the organisms, as dots (a sample of them: the minimap is for where the life is)
            if let frame = play.store.frame, frame.n > 0 {
                let stride = max(1, frame.n / 900)
                var dots = Path()
                for i in Swift.stride(from: 0, to: frame.n, by: stride) {
                    let p = frame.position(i)
                    guard simd_length(p) > 0 else { continue }
                    let at = Mollweide.point(simd_normalize(p), centre: 0, in: box.size)
                    dots.addRect(CGRect(x: box.minX + at.x - 0.7, y: box.minY + at.y - 0.7, width: 1.4, height: 1.4))
                }
                layer.fill(dots, with: .color(.white.opacity(0.8)))
            }
            // the storms
            if let storms = play.store.storms?.storms.storms {
                for storm in storms {
                    guard let d = SphereOverlay.direction(of: storm) else { continue }
                    let at = Mollweide.point(d, centre: 0, in: box.size)
                    let r = max(2.5, CGFloat(Float(storm.r) / Float(header.planetRadius)) * box.width / (2 * .pi) * 0.9)
                    let ring = Path(ellipseIn: CGRect(x: box.minX + at.x - r, y: box.minY + at.y - r, width: 2 * r, height: 2 * r))
                    layer.stroke(ring, with: .color(model.selection == .storm(id: storm.id) ? Self.gold : .white.opacity(0.85)), lineWidth: 1)
                }
            }
            // the footprint of the view: the cap the eye sees, as a line broken at the edge of the chart
            let cap = sphere.footprint()
            if !cap.isEmpty {
                var line = Path()
                var last: CGPoint?
                for d in cap {
                    let at = Mollweide.point(d, centre: 0, in: box.size)
                    let p = CGPoint(x: box.minX + at.x, y: box.minY + at.y)
                    if let l = last, abs(p.x - l.x) < box.width * 0.5 { line.addLine(to: p) } else { line.move(to: p) }
                    last = p
                }
                layer.stroke(line, with: .color(Color(red: 1, green: 0.88, blue: 0.2)), style: StrokeStyle(lineWidth: 1.6, lineJoin: .round))
            }
            // the point looked at
            let c = Mollweide.point(sphere.camera.pose.up, centre: 0, in: box.size)
            let centre = CGPoint(x: box.minX + c.x, y: box.minY + c.y)
            layer.fill(Path(ellipseIn: CGRect(x: centre.x - 2.5, y: centre.y - 2.5, width: 5, height: 5)), with: .color(Color(red: 1, green: 0.88, blue: 0.2)))
        }
        context.stroke(ellipse, with: .color(.white.opacity(0.55)), lineWidth: 1)
        if let home = sphere.hud.home { drawGlyph(home, "house.fill", target: .home, &context) }
        let tag = SphereFrame.placeText(sphere.camera.pose.up)
        let resolved = context.resolve(text(tag, size: 10, mono: true, opacity: 0.85))
        context.draw(resolved, at: CGPoint(x: box.midX, y: box.maxY + 12))
    }

    private func drawGlyph(_ rect: CGRect, _ symbol: String, target: HUDTarget, _ context: inout GraphicsContext) {
        let lit = overlay.pressed == target || overlay.hudHover == target
        let disc = Path(roundedRect: rect, cornerRadius: 5)
        context.fill(disc, with: .color(.black.opacity(0.62)))
        if lit { context.fill(disc, with: .color(.white.opacity(overlay.pressed == target ? 0.3 : 0.16))) }
        context.stroke(disc, with: .color(.white.opacity(0.3)), lineWidth: 0.5)
        var image = context.resolve(Image(systemName: symbol))
        image.shading = .color(.white.opacity(0.95))
        context.draw(image, in: rect.insetBy(dx: rect.width * 0.2, dy: rect.height * 0.2))
    }

    // MARK: The controls

    private func buttonPlate(_ rect: CGRect, target: HUDTarget, _ context: inout GraphicsContext) {
        let lit = overlay.pressed == target || overlay.hudHover == target
        let path = Path(roundedRect: rect, cornerRadius: 7)
        context.fill(path, with: .color(.black.opacity(0.62)))
        if lit { context.fill(path, with: .color(.white.opacity(overlay.pressed == target ? 0.3 : 0.16))) }
        context.stroke(path, with: .color(.white.opacity(0.3)), lineWidth: 0.5)
    }

    private func drawControls(_ sphere: SphereController, _ hud: WorldHUD, _ context: inout GraphicsContext) {
        let pose = sphere.camera.pose
        if let readout = hud.readout { drawAngle(sphere, readout, hud, &context) }
        if let compass = hud.compass { drawCompass(sphere, compass, &context) }
        if let toggle = hud.toggle3D {
            buttonPlate(toggle, target: .toggle3D, &context)
            context.draw(context.resolve(text(pose.tilt < 0.05 ? "3D" : "2D", size: 12, weight: .bold)), at: CGPoint(x: toggle.midX, y: toggle.midY))
        }
        if let track = hud.tiltTrack { drawTilt(sphere, track, &context) }
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
        let hovered = overlay.hudHover
        var rect: CGRect?
        let words: String
        switch hovered {
        case .compass: (rect, words) = (hud.compass, "North up (N)   double click: Home   drag: turn the view")
        case .zoomIn: (rect, words) = (hud.zoomIn, "Zoom in (pinch, mouse wheel)")
        case .zoomOut: (rect, words) = (hud.zoomOut, "Zoom out")
        case .home: (rect, words) = (hud.home, "Home view (H): the whole planet, north up")
        case .toggle3D: (rect, words) = (hud.toggle3D, pose.tilt < 0.05 ? "Tilt the view: 3D" : "Straight down: 2D")
        case .tilt: (rect, words) = (hud.tiltTrack, "Tilt: drag. Straight down at the top, near the horizon at the bottom")
        case .minimap: (rect, words) = (hud.minimap, "Click: fly there. Drag: move the view. Double click: dive")
        case .preset(let preset):
            if let entry = hud.presets.first(where: { $0.preset == preset }), let first = hud.presets.first {
                rect = CGRect(x: first.rect.minX, y: entry.rect.minY, width: entry.rect.maxX - first.rect.minX, height: entry.rect.height)
            }
            words = "\(preset.title) (\(preset.key.uppercased())): \(preset.help)"
        default: (rect, words) = (nil, "")
        }
        if let rect, !words.isEmpty {
            let resolved = context.resolve(text(words, size: 11))
            let measured = resolved.measure(in: CGSize(width: 420, height: 40))
            let tag = hovered == .home
                ? CGRect(x: rect.maxX + 6, y: rect.midY - (measured.height + 6) / 2, width: measured.width + 14, height: measured.height + 6)
                : CGRect(x: rect.minX - measured.width - 22, y: rect.midY - (measured.height + 6) / 2, width: measured.width + 14, height: measured.height + 6)
            plate(tag, radius: 5, &context)
            context.draw(resolved, at: CGPoint(x: tag.minX + 7, y: tag.midY), anchor: .leading)
        }
    }

    private func drawAngle(_ sphere: SphereController, _ readout: CGRect, _ hud: WorldHUD, _ context: inout GraphicsContext) {
        plate(readout, radius: 6, &context)
        context.draw(context.resolve(text(sphere.viewText, size: 11, mono: true)), at: CGPoint(x: readout.midX, y: readout.midY))
        guard let first = hud.presets.first, let last = hud.presets.last else { return }
        let strip = CGRect(x: first.rect.minX, y: first.rect.minY, width: last.rect.maxX - first.rect.minX, height: first.rect.height)
        let outline = Path(roundedRect: strip, cornerRadius: 7)
        context.fill(outline, with: .color(.black.opacity(0.62)))
        let tilt = sphere.camera.pose.tilt
        for entry in hud.presets {
            let target = HUDTarget.preset(entry.preset)
            let lit = overlay.pressed == target || overlay.hudHover == target
            let current = abs(min(entry.preset.tilt, SpherePose.maxTilt) - tilt) < 1.5 * .pi / 180
            let inner = entry.rect.insetBy(dx: 2, dy: 2)
            if current { context.fill(Path(roundedRect: inner, cornerRadius: 5), with: .color(.white.opacity(0.3))) }
            if lit { context.fill(Path(roundedRect: inner, cornerRadius: 5), with: .color(.white.opacity(overlay.pressed == target ? 0.3 : 0.16))) }
            context.draw(context.resolve(text(entry.preset.title, size: 10.5, weight: current ? .bold : .medium)), at: CGPoint(x: entry.rect.midX, y: entry.rect.midY))
        }
        context.stroke(outline, with: .color(.white.opacity(0.3)), lineWidth: 0.5)
    }

    private func drawTilt(_ sphere: SphereController, _ track: CGRect, _ context: inout GraphicsContext) {
        let lit = overlay.pressed == .tilt || overlay.hudHover == .tilt
        let plateRect = Path(roundedRect: track, cornerRadius: 15)
        context.fill(plateRect, with: .color(.black.opacity(0.62)))
        if lit { context.fill(plateRect, with: .color(.white.opacity(overlay.pressed == .tilt ? 0.2 : 0.1))) }
        context.stroke(plateRect, with: .color(.white.opacity(0.3)), lineWidth: 0.5)
        let scale = WorldPose.maxTilt / SpherePose.maxTilt
        let top = WorldHUD.thumbCentre(tilt: 0, track: track), bottom = WorldHUD.thumbCentre(tilt: WorldPose.maxTilt, track: track)
        var groove = Path()
        groove.move(to: top)
        groove.addLine(to: bottom)
        context.stroke(groove, with: .color(.white.opacity(0.28)), style: StrokeStyle(lineWidth: 3, lineCap: .round))
        for preset in [ViewPreset.oblique, .low] {
            let at = WorldHUD.thumbCentre(tilt: preset.tilt * scale, track: track)
            var tick = Path()
            tick.move(to: CGPoint(x: track.midX - 6, y: at.y))
            tick.addLine(to: CGPoint(x: track.midX + 6, y: at.y))
            context.stroke(tick, with: .color(.white.opacity(0.5)), lineWidth: 1)
        }
        let thumb = WorldHUD.thumbCentre(tilt: sphere.camera.pose.tilt * scale, track: track)
        let r = WorldHUD.thumbDiameter / 2
        let disc = Path(ellipseIn: CGRect(x: thumb.x - r, y: thumb.y - r, width: 2 * r, height: 2 * r))
        context.fill(disc, with: .color(.white.opacity(overlay.pressed == .tilt ? 1 : 0.9)))
        context.stroke(disc, with: .color(.black.opacity(0.5)), lineWidth: 0.8)
    }

    /// A disc with a needle that turns with the view: north in red, south pale, and an "N" on the north side.
    private func drawCompass(_ sphere: SphereController, _ rect: CGRect, _ context: inout GraphicsContext) {
        let lit = overlay.pressed == .compass || overlay.hudHover == .compass
        let disc = Path(ellipseIn: rect)
        context.fill(disc, with: .color(.black.opacity(0.62)))
        if lit { context.fill(disc, with: .color(.white.opacity(overlay.pressed == .compass ? 0.28 : 0.14))) }
        context.stroke(disc, with: .color(.white.opacity(0.3)), lineWidth: 0.5)
        // north is at the angle of minus the bearing of the screen's up direction, clockwise from up
        let angle = CGFloat(-sphere.camera.pose.bearing)
        let c = CGPoint(x: rect.midX, y: rect.midY)
        let direction = CGPoint(x: sin(angle), y: -cos(angle))
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

    // MARK: The legend of the height colour

    private func drawLegend(height: Float, _ size: CGSize, _ context: inout GraphicsContext) {
        let bar = CGRect(x: 20, y: max(size.height * 0.5 - 70, 80), width: 12, height: 120)
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

/// The one-line hint of the controls of the planet, once on the first World window and again from View > Controls…; any input takes it away.
struct SphereHintView: View {
    @ObservedObject var model: WorldModel

    static let text = "Drag turns the planet. Two-finger scroll pans; pinch or a mouse wheel zooms (View > Scroll zooms swaps them); zoom out for the whole planet, in for the horizon. H home, N north up, I minimap, C colour, T O L S: top, oblique, low and side views; Option-drag tilts and turns. Click a cell, an organism or a storm to see it."

    var body: some View {
        if model.hintVisible {
            Text(Self.text)
                .font(.callout)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.primary.opacity(0.12)))
                .frame(maxWidth: 580)
                .padding(.horizontal, WorldHUD.compassSide + WorldHUD.margin + 20)
                .padding(.top, 46)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .allowsHitTesting(false)
                .transition(.opacity)
        }
    }
}

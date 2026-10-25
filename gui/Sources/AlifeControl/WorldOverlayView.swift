import AlifeCore
import SwiftUI

/// The `Canvas` over the Metal view: the storms (rings, names, tracks), the warm pools, the selection and what the pointer is
/// over, placed with the camera's projection onto the free rectangle at the minimum image of the camera target (DESIGN
/// §33.4.6, §33.4.8). It watches `OverlayState`, which changes on a tick, on a camera edit, on every frame while something is
/// selected and when the pointer moves onto something else, and nothing else: an idle scene draws nothing.
struct WorldOverlayView: View {
    let model: WorldModel
    @ObservedObject var overlay: OverlayState
    @ObservedObject var play: PlayController

    var body: some View {
        let _ = overlay.revision
        Canvas { context, size in
            let projector = model.projector(size: size)
            // The minimap is drawn by the Metal pass under this canvas, and the compass and the buttons by the controls' canvas over it
            // (a fill that lets what is under it through): none of the map's marks (storms, names, rings, territory) are drawn over
            // them, as they would be where the box covers them.
            var map = context
            map.clip(to: Path(WorldHUD(free: size, minimap: model.display.minimap && model.mapShown, compass: !model.pose.faceDown).clearOfControls))
            let store = play.store
            let selection = model.selection
            let hover = overlay.hover
            let castHover = overlay.castHover
            // Territory first: what a kin group owns, under the rings.
            if let territory = store.territory {
                for target in [castHover, selection] {
                    if case .kin(let id)? = target { drawTerritory(of: id, territory, store, projector, &map) }
                }
                if case .territory(_, let kin)? = hover?.pick { drawTerritory(of: kin, territory, store, projector, &map, faint: true) }
            }
            if let environment = store.environment {
                drawPools(environment, projector, selected: selection, &map)
            }
            drawPowerAim(store, projector, &map)
            if let storms = store.storms {
                var marked: Set<Int> = []
                if case .storm(let id) = selection { marked.insert(id) }
                if case .storm(let id)? = castHover { marked.insert(id) }
                if case .storm(let id)? = hover?.pick { marked.insert(id) }
                let glyphs = projector.glyphs(of: storms.storms)
                drawStorms(glyphs, projector.level, selected: selection, marked: marked, names: !projector.isMorphing, &map)
                if projector.isMorphing { drawGlobeLabels(glyphs, projector, size: size, &map) }
            } else if projector.isMorphing {
                drawGlobeLabels([], projector, size: size, &map)
            }
            if let uid = selection.organismUID { drawSelection(uid, store, projector, &map) }
            if case .organism(let uid, let index)? = hover?.pick, uid != selection.organismUID {
                drawHover(uid, hint: index, store, projector, &map)
            }
            if let hover { drawTooltip(hover, in: size, &context) }  // (the tooltip is the pointer's, over everything)
        }
        .allowsHitTesting(false)
    }

    // MARK: Pieces

    private func drawStorms(_ glyphs: [StormGlyph], _ level: WorldLevel, selected: WorldSelection, marked: Set<Int>,
                            names: Bool = true, _ context: inout GraphicsContext) {
        for glyph in glyphs {
            let tint: Color = glyph.hemisphere == 0 ? Color(red: 1, green: 0.85, blue: 0.4) : Color(red: 0.5, green: 0.85, blue: 1)
            let alpha = glyph.named ? 1.0 : 0.55
            if glyph.track.count > 1 {
                var path = Path()
                path.addLines(glyph.track)
                context.stroke(path, with: .color(tint.opacity(0.35 * alpha)), lineWidth: 1.2)
            }
            // (a circle on the floor: flattened towards the horizon)
            let ry = glyph.radius * glyph.flatten
            let ring = Path(ellipseIn: CGRect(x: glyph.center.x - glyph.radius, y: glyph.center.y - ry, width: 2 * glyph.radius, height: 2 * ry))
            context.stroke(ring, with: .color(tint.opacity(alpha)),
                           style: StrokeStyle(lineWidth: glyph.named ? 2 : 1, dash: glyph.coasting ? [4, 3] : []))
            if marked.contains(glyph.id) {
                let isSelected = selected == .storm(id: glyph.id)
                let r = glyph.radius + 4, rv = ry + 4
                let outer = Path(ellipseIn: CGRect(x: glyph.center.x - r, y: glyph.center.y - rv, width: 2 * r, height: 2 * rv))
                context.stroke(outer, with: .color(.white.opacity(isSelected ? 0.95 : 0.55)), lineWidth: isSelected ? 2 : 1)
            }
            if names, level != .organism {
                let label = glyph.name ?? "disturbance \(glyph.id)"
                let text = Text(label).font(.caption2).foregroundColor(tint.opacity(alpha))
                context.draw(context.resolve(text), at: CGPoint(x: glyph.center.x, y: glyph.center.y - ry - 8))
            }
        }
    }

    /// The storms' names (at most 8, named first then by height, 90 points apart, a leader line when one is moved) and the
    /// geography (the centre's hemisphere, EQUATOR, the seam) at the globe and in the morph: placed together so that the geography
    /// never lands on a name (DESIGN §36.10, the spec's 2.4 layer 8).
    private func drawGlobeLabels(_ glyphs: [StormGlyph], _ projector: WorldProjector, size: CGSize, _ context: inout GraphicsContext) {
        var texts: [String: GraphicsContext.ResolvedText] = [:]
        var tints: [String: Color] = [:]
        var candidates: [LabelCandidate] = []
        for glyph in glyphs {
            let tint: Color = glyph.hemisphere == 0 ? Color(red: 1, green: 0.85, blue: 0.4) : Color(red: 0.5, green: 0.85, blue: 1)
            let id = "storm \(glyph.id)"
            let resolved = context.resolve(Text(glyph.name ?? "disturbance \(glyph.id)").font(.caption2).foregroundColor(tint))
            texts[id] = resolved
            tints[id] = tint
            let measured = resolved.measure(in: CGSize(width: 300, height: 30))
            let ry = glyph.radius * glyph.flatten
            candidates.append(LabelCandidate(id: id, kind: .storm, rank: (glyph.named ? 1000 : 0) + glyph.zeta,
                                             anchor: CGPoint(x: glyph.center.x, y: glyph.center.y - ry - 8), size: measured))
        }
        for place in GlobeLabels.places(projector: projector) {
            let resolved = context.resolve(Text(place.text).font(.system(size: 10, weight: .semibold)).foregroundColor(.white.opacity(0.85)))
            texts[place.id] = resolved
            let measured = resolved.measure(in: CGSize(width: 300, height: 30))
            candidates.append(LabelCandidate(id: place.id, kind: .geography, rank: 0, anchor: place.anchor, size: measured))
        }
        let placed = LabelDeclutter.place(candidates, in: CGRect(origin: .zero, size: size))
        for label in placed {
            guard let text = texts[label.id] else { continue }
            if let from = label.leader {
                var leader = Path()
                leader.move(to: from)
                leader.addLine(to: label.centre)
                context.stroke(leader, with: .color((tints[label.id] ?? .white).opacity(0.5)), lineWidth: 0.7)
            }
            var layer = context
            layer.addFilter(.shadow(color: .black.opacity(0.9), radius: 1.5, x: 0, y: 0))
            layer.draw(text, at: label.centre)
        }
    }

    private func drawPools(_ environment: PlayEnvironmentRecord, _ projector: WorldProjector, selected: WorldSelection,
                           _ context: inout GraphicsContext) {
        for (index, pool) in environment.environment.pools.enumerated() {
            let at = SIMD3<Float>(Float(pool.x), Float(pool.y), 0)
            guard let center = projector.screen(at) else { continue }
            let radius = max(projector.points(ofUnits: Float(max(pool.sx, pool.sy)) * 2, at: at) ?? 0, 6)
            let ring = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: 2 * radius, height: 2 * radius))
            let chosen = selected == .pool(index: index)
            context.stroke(ring, with: .color(chosen ? .white : Color(red: 1, green: 0.55, blue: 0.2).opacity(0.45)),
                           style: StrokeStyle(lineWidth: chosen ? 2 : 1, dash: [2, 3]))
        }
    }

    /// The food patches the powers can aim at (while the Powers tab is open or a power is chosen), the one a click would take or
    /// has taken, and a strike's disc at the pointer or its target (DESIGN §34.7).
    private func drawPowerAim(_ store: WorldStore, _ projector: WorldProjector, _ context: inout GraphicsContext) {
        let powers = model.powers
        guard model.tab == .powers || powers.offer != nil else { return }
        let patches = store.patches
        var chosen: Int?
        if case .patch(let i)? = powers.target {
            chosen = i
        } else if powers.isAiming, powers.offer?.scope == .patch, let world = overlay.aimWorld {
            chosen = PowersModel.nearestPatch(to: world, in: patches, size: Float(model.worldSize))
        }
        let green = Color(red: 0.45, green: 0.85, blue: 0.45)
        for patch in patches {
            let at = SIMD3<Float>(Float(patch.x), Float(patch.y), 0)
            guard let center = projector.screen(at) else { continue }
            let radius = max(projector.points(ofUnits: Float(patch.sigma) * 2, at: at) ?? 0, 8)
            let ring = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: 2 * radius, height: 2 * radius))
            let hot = chosen == patch.i
            context.stroke(ring, with: .color(hot ? .white : green.opacity(0.6)),
                           style: StrokeStyle(lineWidth: hot ? 2.5 : 1, dash: hot ? [] : [5, 3]))
            if hot { context.fill(ring, with: .color(green.opacity(0.18))) }
            let label = context.resolve(Text("patch \(patch.i)").font(.caption2).foregroundColor(green.opacity(hot ? 1 : 0.7)))
            context.draw(label, at: CGPoint(x: center.x, y: center.y + radius + 8))
        }
        if powers.offer?.scope == .column {
            var place: SIMD2<Float>?
            if case .point(let x, let y)? = powers.target { place = SIMD2(Float(x), Float(y)) } else { place = overlay.aimWorld }
            if let place, let radiusUnits = powers.values[.radius] {
                let at = SIMD3<Float>(place.x, place.y, 0)
                if let center = projector.screen(at) {
                    let radius = max(projector.points(ofUnits: Float(radiusUnits), at: at) ?? 0, 4)
                    let disc = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: 2 * radius, height: 2 * radius))
                    context.fill(disc, with: .color(.red.opacity(0.16)))
                    context.stroke(disc, with: .color(.red.opacity(0.9)), style: StrokeStyle(lineWidth: 1.5, dash: powers.target == nil ? [4, 3] : []))
                }
            }
        }
    }

    /// The cells a kin group owns, outlined and tinted in its colour, at every copy of the world that shows in the view.
    private func drawTerritory(of kin: Int, _ territory: PlayTerritory, _ store: WorldStore, _ projector: WorldProjector,
                               _ context: inout GraphicsContext, faint: Bool = false) {
        guard let slot = territory.ids.firstIndex(of: kin), slot < Slot.maxTable else { return }
        let bounds = CGRect(origin: .zero, size: projector.view)
        let color = Color(WorldPalette.slotColor(UInt8(slot)) ?? WorldPalette.other)
        var fill = Path()
        for cell in territory.owner.indices where Int(territory.owner[cell]) == slot {
            let cx = cell % territory.nx, cy = cell / territory.nx
            for sx in -1...1 {
                for sy in -1...1 {
                    guard let quad = projector.cellCorners(cx: cx, cy: cy, nx: territory.nx, ny: territory.ny, shift: SIMD2(sx, sy)),
                          quad.contains(where: { bounds.insetBy(dx: -40, dy: -40).contains($0) })
                    else { continue }
                    fill.move(to: quad[0])
                    quad.dropFirst().forEach { fill.addLine(to: $0) }
                    fill.closeSubpath()
                }
            }
        }
        context.fill(fill, with: .color(color.opacity(faint ? 0.16 : 0.3)))
        context.stroke(fill, with: .color(.white.opacity(faint ? 0.35 : 0.8)), lineWidth: faint ? 0.8 : 1.2)
    }

    private func drawSelection(_ uid: Int64, _ store: WorldStore, _ projector: WorldProjector, _ context: inout GraphicsContext) {
        guard let frame = store.frame, let index = store.listedIndex(ofUID: uid, hint: model.selectionHint) else { return }
        model.selectionHint = index
        guard let center = projector.screen(frame.position(index)) else { return }
        let radius = max(projector.points(ofUnits: Float(store.header?.radius ?? 1) * 1.8, at: frame.position(index)) ?? 0, 9)
        let ring = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: 2 * radius, height: 2 * radius))
        context.stroke(ring, with: .color(.black.opacity(0.7)), lineWidth: 4)  // (the keyline)
        context.stroke(ring, with: .color(.white), lineWidth: 2)
        drawPin(uid, index: index, frame.position(index), projector, store, &context)
    }

    /// The pin of the selected or followed organism at the globe: a line from the surface to its height with a ring at the top (white,
    /// a 2 point dark keyline) and a line of words, so that its height over the planet reads (the spec's 2.3). Only in the morph
    /// and where the stalk is long enough to see.
    private func drawPin(_ uid: Int64, index: Int, _ position: SIMD3<Float>, _ projector: WorldProjector, _ store: WorldStore,
                         _ context: inout GraphicsContext) {
        guard projector.isMorphing, position.z > 0.5,
              let foot = projector.screen(SIMD3(position.x, position.y, 0)), let top = projector.screen(position) else { return }
        let length = hypot(top.x - foot.x, top.y - foot.y)
        guard length >= 3 else { return }
        var stalk = Path()
        stalk.move(to: foot)
        stalk.addLine(to: top)
        context.stroke(stalk, with: .color(.black.opacity(0.7)), lineWidth: 3.5)
        context.stroke(stalk, with: .color(.white.opacity(0.9)), lineWidth: 1.4)
        let words = (WorldPicker.describe(.organism(uid: uid, index: index), store: store) ?? "organism \(uid)")
            .replacingOccurrences(of: "organism \(uid) · ", with: "") + String(format: " · z %.0f", position.z)
        let text = context.resolve(Text(words).font(.caption2).foregroundColor(.white))
        let measured = text.measure(in: CGSize(width: 360, height: 30))
        let tag = CGRect(x: top.x + 12, y: top.y - 26 - measured.height / 2, width: measured.width + 10, height: measured.height + 4)
        context.fill(Path(roundedRect: tag, cornerRadius: 4), with: .color(.black.opacity(0.6)))
        context.draw(text, at: CGPoint(x: tag.midX, y: tag.midY))
    }

    private func drawHover(_ uid: Int64, hint: Int, _ store: WorldStore, _ projector: WorldProjector, _ context: inout GraphicsContext) {
        guard let frame = store.frame, let index = store.listedIndex(ofUID: uid, hint: hint),
              let center = projector.screen(frame.position(index)) else { return }
        let radius = max(projector.points(ofUnits: Float(store.header?.radius ?? 1) * 1.6, at: frame.position(index)) ?? 0, 7)
        let ring = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: 2 * radius, height: 2 * radius))
        context.stroke(ring, with: .color(.white.opacity(0.7)), lineWidth: 1)
    }

    /// One line near the pointer, kept inside the view.
    private func drawTooltip(_ hover: HoverInfo, in size: CGSize, _ context: inout GraphicsContext) {
        let text = context.resolve(Text(hover.text).font(.caption).foregroundColor(.white))
        let measured = text.measure(in: CGSize(width: 420, height: 40))
        let box = CGSize(width: measured.width + 12, height: measured.height + 6)
        var origin = CGPoint(x: hover.point.x + 14, y: hover.point.y + 18)
        origin.x = min(max(origin.x, 4), max(size.width - box.width - 4, 4))
        origin.y = min(max(origin.y, 4), max(size.height - box.height - 4, 4))
        let rect = CGRect(origin: origin, size: box)
        context.fill(Path(roundedRect: rect, cornerRadius: 4), with: .color(.black.opacity(0.75)))
        context.draw(text, at: CGPoint(x: rect.midX, y: rect.midY))
    }
}

extension Color {
    /// A palette colour `0xRRGGBB`.
    init(_ rgb: UInt32) {
        let (r, g, b) = WorldPalette.components(rgb)
        self.init(red: Double(r) / 255, green: Double(g) / 255, blue: Double(b) / 255)
    }
}

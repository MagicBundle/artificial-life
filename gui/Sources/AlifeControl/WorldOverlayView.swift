import AlifeCore
import SwiftUI

/// The `Canvas` over the Metal view: the storms (rings, names, tracks), the warm pools and the selected organism, placed with
/// the camera's projection onto the free rectangle at the minimum image of the camera target (DESIGN §33.4.6). It watches
/// `OverlayState`, which changes on a tick, on a camera edit and on every frame while an organism is selected, and nothing
/// else: an idle scene draws nothing.
struct WorldOverlayView: View {
    let model: WorldModel
    @ObservedObject var overlay: OverlayState
    @ObservedObject var play: PlayController

    var body: some View {
        let _ = overlay.revision
        Canvas { context, size in
            let projector = model.projector(size: size)
            let store = play.store
            if let environment = store.environment { drawPools(environment, projector, &context) }
            if let storms = store.storms { drawStorms(projector.glyphs(of: storms.storms), projector.level, &context) }
            if let uid = model.selectedUID { drawSelection(uid, store, projector, &context) }
        }
        .allowsHitTesting(false)
    }

    // MARK: Pieces

    private func drawStorms(_ glyphs: [StormGlyph], _ level: WorldLevel, _ context: inout GraphicsContext) {
        for glyph in glyphs {
            let tint: Color = glyph.hemisphere == 0 ? Color(red: 1, green: 0.85, blue: 0.4) : Color(red: 0.5, green: 0.85, blue: 1)
            let alpha = glyph.named ? 1.0 : 0.55
            if glyph.track.count > 1 {
                var path = Path()
                path.addLines(glyph.track)
                context.stroke(path, with: .color(tint.opacity(0.35 * alpha)), lineWidth: 1.2)
            }
            let ring = Path(ellipseIn: CGRect(x: glyph.center.x - glyph.radius, y: glyph.center.y - glyph.radius,
                                              width: 2 * glyph.radius, height: 2 * glyph.radius))
            context.stroke(ring, with: .color(tint.opacity(alpha)),
                           style: StrokeStyle(lineWidth: glyph.named ? 2 : 1, dash: glyph.coasting ? [4, 3] : []))
            if level != .organism {
                let label = glyph.name ?? "disturbance \(glyph.id)"
                let text = Text(label).font(.caption2).foregroundColor(tint.opacity(alpha))
                context.draw(context.resolve(text), at: CGPoint(x: glyph.center.x, y: glyph.center.y - glyph.radius - 8))
            }
        }
    }

    private func drawPools(_ environment: PlayEnvironmentRecord, _ projector: WorldProjector, _ context: inout GraphicsContext) {
        for pool in environment.environment.pools {
            let at = SIMD3<Float>(Float(pool.x), Float(pool.y), 0)
            guard let center = projector.screen(at) else { continue }
            let radius = max(projector.points(ofUnits: Float(max(pool.sx, pool.sy)) * 2, at: at) ?? 0, 6)
            let ring = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: 2 * radius, height: 2 * radius))
            context.stroke(ring, with: .color(Color(red: 1, green: 0.55, blue: 0.2).opacity(0.45)),
                           style: StrokeStyle(lineWidth: 1, dash: [2, 3]))
        }
    }

    private func drawSelection(_ uid: Int64, _ store: WorldStore, _ projector: WorldProjector, _ context: inout GraphicsContext) {
        guard let frame = store.frame, let index = store.listedIndex(ofUID: uid) else { return }
        guard let center = projector.screen(frame.position(index)) else { return }
        let radius = max(projector.points(ofUnits: Float(store.header?.radius ?? 1) * 1.8, at: frame.position(index)) ?? 0, 9)
        let ring = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: 2 * radius, height: 2 * radius))
        context.stroke(ring, with: .color(.white), lineWidth: 2)
    }
}

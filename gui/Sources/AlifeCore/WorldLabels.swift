import CoreGraphics
import Foundation
import simd

/// A label that wants a place on the globe or in the morph (DESIGN §36.10): a storm's name, a hemisphere's name, EQUATOR, the seam.
package struct LabelCandidate: Equatable, Sendable {
    package enum Kind: Int, Sendable { case storm = 0, geography = 1 }
    package var id: String
    package var kind: Kind
    /// Higher goes first within a kind: a named storm before an unnamed one, then by height (`zeta`).
    package var rank: Double
    /// The point the label belongs to (a storm's centre above its ring, a place on the equator), in the view's points.
    package var anchor: CGPoint
    /// The label's size.
    package var size: CGSize

    package init(id: String, kind: Kind, rank: Double, anchor: CGPoint, size: CGSize) {
        self.id = id
        self.kind = kind
        self.rank = rank
        self.anchor = anchor
        self.size = size
    }
}

/// Where a label went: centred on `centre`; `leader` is the anchor when the label was moved off it (a thin line joins them).
package struct PlacedLabel: Equatable, Sendable {
    package var id: String
    package var centre: CGPoint
    package var leader: CGPoint?
}

/// The declutter rule of the globe's overlay (the spec's 2.4, layer 8): at most 8 storm names, named ones first and then by height,
/// no two names closer than 90 points between their centres, a label that would land on another is moved (a leader line) or left
/// out; the geography (hemisphere, EQUATOR, the seam) comes after the names and never lands on one. Pure and deterministic: the
/// order is the candidates' rank then their id, the moves a fixed ring of offsets.
package enum LabelDeclutter {
    package static let maxNames = 8
    package static let spacing: CGFloat = 90
    /// The moves tried in turn (the anchor itself first): four rings of eight directions.
    package static let offsets: [CGPoint] = {
        var out: [CGPoint] = [.zero]
        for r: CGFloat in [28, 56, 84, 112] {
            for k in 0..<8 {
                let a = CGFloat(k) * .pi / 4
                out.append(CGPoint(x: r * cos(a), y: r * sin(a)))
            }
        }
        return out
    }()

    package static func place(_ candidates: [LabelCandidate], in bounds: CGRect) -> [PlacedLabel] {
        let order = candidates.sorted {
            if $0.kind != $1.kind { return $0.kind.rawValue < $1.kind.rawValue }
            if $0.rank != $1.rank { return $0.rank > $1.rank }
            return $0.id < $1.id
        }
        var placed: [PlacedLabel] = []
        var rects: [CGRect] = []
        var names: [CGPoint] = []
        for label in order {
            if label.kind == .storm, names.count >= maxNames { continue }
            guard bounds.contains(label.anchor) else { continue }
            for offset in offsets {
                let centre = CGPoint(x: label.anchor.x + offset.x, y: label.anchor.y + offset.y)
                let rect = CGRect(x: centre.x - label.size.width / 2, y: centre.y - label.size.height / 2,
                                  width: label.size.width, height: label.size.height)
                guard bounds.insetBy(dx: -label.size.width / 2, dy: -label.size.height / 2).contains(centre) else { continue }
                if rects.contains(where: { $0.insetBy(dx: -3, dy: -2).intersects(rect) }) { continue }
                if label.kind == .storm, names.contains(where: { hypot($0.x - centre.x, $0.y - centre.y) < spacing }) { continue }
                placed.append(PlacedLabel(id: label.id, centre: centre, leader: offset == .zero ? nil : label.anchor))
                rects.append(rect)
                if label.kind == .storm { names.append(centre) }
                break
            }
        }
        return placed
    }
}

/// The places of the globe's geography labels (the spec's 4.2): one hemisphere's name at `+-0.16 L` in the centre's hemisphere,
/// EQUATOR on the equator, "wraps to the other pole" on the seam row; each at the centre's meridian, only where it is on the
/// visible side and in the picture.
package enum GlobeLabels {
    package static let hemisphereRow: Float = 0.16

    package struct Place: Equatable, Sendable {
        package var id: String
        package var text: String
        package var anchor: CGPoint
    }

    package static func places(projector: WorldProjector) -> [Place] {
        guard let vantage = projector.vantage, vantage.isGlobe, vantage.s > 0 else { return [] }
        let L = projector.size
        let x = vantage.frame.x
        let ybar = vantage.frame.ybar
        let bounds = CGRect(origin: .zero, size: projector.view)
        func spot(_ ybarOfLabel: Float) -> CGPoint? {
            let y = Torus.wrap(ybarOfLabel, size: L)
            guard let p = vantage.screen(SIMD3(x, y, 0), in: projector.view), bounds.contains(p) else { return nil }
            return p
        }
        var out: [Place] = []
        let north = ybar >= 0
        if let p = spot((north ? 1 : -1) * hemisphereRow * L) {
            out.append(Place(id: "hemisphere", text: north ? "Northern hemisphere" : "Southern hemisphere", anchor: p))
        }
        if let p = spot(0) { out.append(Place(id: "equator", text: "EQUATOR", anchor: p)) }
        // The seam row of the pole nearer to the centre (the pole caps close it for s >= 0.9; the label stays where the row is).
        if let p = spot(north ? 0.5 * L - 0.001 * L : -0.5 * L + 0.001 * L), vantage.s < 0.99 || abs(ybar) > 0.3 * L {
            out.append(Place(id: "seam", text: "wraps to the other pole", anchor: p))
        }
        return out
    }
}

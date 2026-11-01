import Foundation
import simd

/// What the renderer is given for a world in the store (DESIGN §46.8): the cells' colours, the overlay lines and the Home view, composed from
/// the records by pure functions so that the window, the minimap and the tests agree.
package struct SphereScene: Equatable, Sendable {
    package var floor: [SIMD4<Float>]
    package var cloud: [SIMD4<Float>]
    package var territory: [SIMD4<Float>]
    /// The pools and the wind: under the cloud.
    package var overlay: [SphereLine]
    /// The storms' rings, tracks and predictions: drawn after the cloud shell, so that it never veils them.
    package var top: [SphereLine] = []

    /// The layers of the map as the player sees them.
    package struct Options: Equatable, Sendable {
        package var display = WorldDisplay()
        /// The lattice layer of the built material and the soil (nil: the column's maximum).
        package var layer: Int?
        package var selectedStorm: Int?
        package var selectedPool: Int?
        package var selectedKin: Int?
        /// Draw the wind as streamlines (the weather layer's wind).
        package var wind = true
        package init() {}
    }

    package static func compose(store: WorldStore, options: Options) -> SphereScene? {
        guard let sphere = store.header?.sphere else { return nil }
        let display = options.display
        let centres = SphereCentres.shared.centres(nside: sphere.nsideField)
        let floor = SphereColours.floor(sphere: sphere, map: store.sphereMap, material: store.material, soil: store.soil, display: display,
                                        layer: options.layer, centres: centres)
        let cloud = SphereColours.cloud(sphere: sphere, map: store.sphereMap, display: display)
        let territory = SphereColours.territory(store.territory, nside: store.territory?.nside ?? sphere.nsideTerritory, highlight: options.selectedKin)
        var lines: [SphereLine] = []
        var top: [SphereLine] = []
        if let storms = store.storms?.storms.storms { top = SphereOverlay.storms(storms, sphere: sphere, selected: options.selectedStorm) }
        if let pools = store.environment?.environment.pools {
            lines += SphereOverlay.pools(pools, sphere: sphere, selected: options.selectedPool)
        }
        if options.wind, display.map, let map = store.sphereMap, map.npix == sphere.fieldCells {
            lines += SphereOverlay.streamlines(map, sphere: sphere, layer: 0, seeds: 450, lift: Float(sphere.shellHeight) * 0.1)
        }
        return SphereScene(floor: floor, cloud: cloud, territory: territory, overlay: lines, top: top)
    }

    /// The Home view: the planet whole, north up, centred on the populated side (the mean direction of the organisms, the equator at
    /// longitude 0 for an empty world).
    package static func home(store: WorldStore, aspect: Float, fovY: Float = SpherePose.defaultFovY) -> SpherePose? {
        guard let sphere = store.header?.sphere else { return nil }
        var sum = SIMD3<Float>.zero
        if let frame = store.frame {
            for i in 0..<frame.n {
                let p = frame.position(i)
                let l = simd_length(p)
                if l > 0 { sum += p / l }
            }
        }
        let centre = simd_length(sum) > 1e-3 ? sum : SIMD3<Float>(1, 0, 0)
        return SphereCamera.home(planetRadius: Float(sphere.planetRadius), lidRadius: Float(sphere.lidRadius), centre: centre,
                                 aspect: aspect, fovY: fovY)
    }
}

/// The pixel centres of an Nside, computed once per Nside (a world has one or two).
package final class SphereCentres: @unchecked Sendable {
    package static let shared = SphereCentres()
    private let lock = NSLock()
    private var cache: [Int: [SIMD3<Float>]] = [:]

    package func centres(nside: Int) -> [SIMD3<Float>] {
        lock.lock()
        defer { lock.unlock() }
        if let c = cache[nside] { return c }
        let c = Healpix.centres(nside: nside)
        cache[nside] = c
        return c
    }
}

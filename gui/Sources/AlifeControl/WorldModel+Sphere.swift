import AlifeCore
import CoreGraphics
import Foundation
import simd

/// The planet's share of the World window's model (DESIGN §46.8). The camera, the picking and the renderer feed are the `SphereController`'s;
/// this routes the model's own entry points (the dock's clicks on a storm, a lineage or a chronicle row, the keys, the toolbar's Home) to it
/// while the world in view is a sphere. A box world never reaches any of this: `sphereActive` is false and every guard falls through.
extension WorldModel {
    /// Whether the world in view is a planet.
    var sphereActive: Bool { play.store.header?.isSphere == true }

    /// The direction of a storm of the newest `STRM`.
    private func stormDirection(_ id: Int) -> SIMD3<Float>? {
        play.store.storms?.storms.storms.first { $0.id == id }.flatMap { SphereOverlay.direction(of: $0) }
    }

    func selectOrganismOnPlanet(_ uid: Int64, follow serverFollow: Bool, camera cameraFollow: Bool, fly: Bool) {
        choose(.organism(uid: uid))
        play.send(serverFollow ? .follow(uid) : .focus(uid))
        sphere.clearCell()
        if cameraFollow {
            sphere.setFollow(.organism(uid: uid))
            if fly, sphere.camera.pose.distance > sphere.flyDistanceOrganism() { sphere.fly(toFollowing: sphere.flyDistanceOrganism()) }
        } else {
            sphere.cancelMotion(keepSelection: true)
            if fly, let index = play.store.listedIndex(ofUID: uid), let frame = play.store.frame {
                sphere.flyTo(frame.position(index), distance: sphere.flyDistanceOrganism())
            }
        }
        sphere.startAnimating()
    }

    func selectStormOnPlanet(_ id: Int, fly: Bool, follow: Bool) {
        choose(.storm(id: id))
        sphere.clearCell()
        sphere.setFollow(follow ? .storm(id: id) : .none)
        if fly {
            if follow {
                sphere.fly(toFollowing: sphere.flyDistanceRegion())
            } else if let direction = stormDirection(id) {
                sphere.flyTo(direction, distance: sphere.flyDistanceRegion())
            }
        }
    }

    func selectKinOnPlanet(_ id: Int, fly: Bool, follow: Bool) {
        choose(.kin(id: id))
        sphere.clearCell()
        sphere.setFollow(follow ? .lineage(kinID: id) : .none)
        if fly {
            var state = SphereFollow(target: .lineage(kinID: id))
            if follow {
                sphere.fly(toFollowing: sphere.flyDistanceRegion())
            } else if let goal = state.goal(store: play.store) {
                sphere.flyTo(goal, distance: sphere.flyDistanceRegion())
            }
        }
    }

    func selectPoolOnPlanet(_ index: Int) {
        choose(.pool(index: index))
        sphere.clearCell()
        sphere.setFollow(.none)
        if let pools = play.store.environment?.environment.pools, pools.indices.contains(index), let z = pools[index].z {
            sphere.flyTo(SIMD3(Float(pools[index].x), Float(pools[index].y), Float(z)), distance: sphere.flyDistanceRegion())
        }
    }

    /// A click on a place and an entity (an alert's arrow, a toast, a sentence of a report, a chronicle row), on a planet.
    func activateOnPlanet(_ action: ChronicleAction) {
        let place = action.planetDirection
        switch action.select {
        case .storm(let id):
            if stormDirection(id) != nil {
                selectStormOnPlanet(id, fly: true, follow: false)
            } else {
                choose(.storm(id: id))
                sphere.setFollow(.none)
                if let place { sphere.flyTo(place, distance: sphere.flyDistanceRegion()) }
            }
        case .kin(let id):
            choose(.kin(id: id))
            sphere.setFollow(.none)
            var state = SphereFollow(target: .lineage(kinID: id))
            if let goal = state.goal(store: play.store) ?? place { sphere.flyTo(goal, distance: sphere.flyDistanceRegion()) }
        case .organism(let uid):
            if play.store.listedIndex(ofUID: uid) != nil {
                selectOrganismOnPlanet(uid, follow: true, camera: true, fly: false)
            } else {
                selectOrganismOnPlanet(uid, follow: false, camera: false, fly: false)
                if let place { sphere.flyTo(place, distance: sphere.flyDistanceRegion()) }
            }
        case .pool, .none:
            sphere.setFollow(.none)
            if let place { sphere.flyTo(place, distance: sphere.flyDistanceRegion()) }
        }
    }
}

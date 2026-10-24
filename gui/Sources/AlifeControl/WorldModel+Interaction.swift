import AlifeCore
import CoreGraphics
import Foundation
import simd

/// A fly-to under way on the path of van Wijk and Nuij (`FlightPath`), `elapsed` seconds in; `home` when it is the flight of the Home view.
struct Flight: Equatable {
    var path: FlightPath
    var elapsed: Float = 0
    var home = false
}

/// The north-up ease: the heading goes from `from` to 0 over `duration` seconds, smoothly.
struct HeadingEase: Equatable {
    static let duration: Float = 0.3
    let from: Float
    var elapsed: Float = 0

    /// The heading `dt` seconds on, and whether the ease is over.
    mutating func step(dt: Float) -> (heading: Float, done: Bool) {
        elapsed += dt
        let t = min(max(elapsed / Self.duration, 0), 1)
        return (from * (1 - t * t * (3 - 2 * t)), t >= 1)
    }
}

/// Picking, selection, follow, fly-to, keys, hover and toasts of the World window (DESIGN §33.4.8). The decisions are the
/// pure types of `AlifeCore` (`WorldPicker`, `FollowState`, `CameraGlide`, `WorldSelection`, `ToastQueue`, `WorldKeys`); this
/// is the glue that applies them to the camera, the overlay and the world's control channel.
///
/// What a click means (D10): a single click on an organism **selects and follows** it: the world's `follow UID` (so that the
/// inspector survives its death, the hand-off passing to a child) and the camera on it. A shift-click selects only
/// (`focus UID`, no hand-off, the camera stays). A double click, or `F` on a selection, also flies in to it. A click on a storm
/// glyph selects it and flies there; a double click follows it. A click on a territory cell selects its kin group and changes
/// nothing else (not the global colour mode). A click on nothing clears the selection.
extension WorldModel {
    // MARK: Picking

    /// A click at `point` (in the free rectangle's points, origin at the top left) with `clicks` the number of clicks of
    /// the sequence (2 is the second click of a double click).
    func click(at point: CGPoint, shift: Bool, clicks: Int) {
        // While a power is being aimed the map is its target: a click names the place and selects nothing.
        if powers.isAiming {
            if clicks == 1 { _ = aimClick(at: point) }
            return
        }
        let pick = WorldPicker.pick(at: point, projector: projector(size: freeRect.size), store: play.store, display: display,
                                    maxRadius: drawnRadiusCap)
        switch pick {
        case .organism(let uid, _):
            selectOrganism(uid, follow: !shift, camera: !shift, fly: clicks >= 2)
        case .storm(let id):
            selectStorm(id, fly: true, follow: clicks >= 2)
        case .pool(let index):
            selectPool(index)
        case .territory(_, let kin):
            selectKin(id: kin, fly: false, follow: false)
            if clicks >= 2 { dive(at: point) }
        case .none:
            if clicks == 1 { clearSelection() } else { dive(at: point) }
        }
    }

    /// A double click on an empty point or a territory cell: fly there, and in to the region distance (DESIGN §36.5), unless the camera
    /// is closer already. The point is the one of the plane through the view's centre under the pointer.
    func dive(at point: CGPoint) {
        guard let place = projector(size: freeRect.size).world(at: point, height: pose.z) else { return }
        flyTo(SIMD3(place.x, place.y, 0), distance: min(camera.distance, flyDistance(.region)))
    }

    /// The pointer moved to `point`, or left the view (nil). No round trip to the world: the same hit test as a click.
    func hover(at point: CGPoint?) {
        // Over a control the map has no tooltip and no readout of its own.
        if let point, freeRect.width >= 1 {
            let target = pressTarget(at: point)
            hoverHUD(target, at: point)
            if target != .map {
                overlay.setHover(nil)
                overlay.setPointerText(nil)
                return
            }
        } else {
            hoverHUD(.map, at: .zero)
        }
        if powers.isAiming {
            overlay.setPointerText(pointerReadout(at: point))
            aimHover(at: point)
            overlay.setHover(nil)
            return
        }
        guard let point else {
            overlay.setPointerText(nil)
            overlay.setHover(nil)
            return
        }
        let t = now()
        if lastHoverPick >= 0, t - lastHoverPick < Self.hoverPeriod, t >= lastHoverPick {  // (42k projections a pick)
            overlay.setPointerText(pointerReadout(at: point, over: overlay.hover?.pick))
            return
        }
        lastHoverPick = t
        let store = play.store
        let pick = WorldPicker.pick(at: point, projector: projector(size: freeRect.size), store: store, display: display,
                                    maxRadius: drawnRadiusCap)
        if let text = WorldPicker.describe(pick, store: store) {
            overlay.setHover(HoverInfo(point: point, pick: pick, text: text))
        } else {
            overlay.setHover(nil)
        }
        overlay.setPointerText(pointerReadout(at: point, over: pick))
    }

    static let hoverPeriod = 1.0 / 60

    // MARK: Selection

    /// Changes the selection, and tells the world to stop reporting an organism that is no longer selected.
    func choose(_ new: WorldSelection) {
        let old = selection
        setSelection(new)
        if old.organismUID != nil, new.organismUID == nil { play.send(.focus(nil)) }
    }

    func selectOrganism(_ uid: Int64, follow serverFollow: Bool = true, camera cameraFollow: Bool = true, fly: Bool = false) {
        choose(.organism(uid: uid))
        play.send(serverFollow ? .follow(uid) : .focus(uid))
        if cameraFollow {
            setFollow(.organism(uid: uid))
        } else {
            cancelCameraMotion()
        }
        if fly, camera.distance > flyDistance(.organism) {
            isFlying = true
            flyDistance = flyDistance(.organism)
        }
        if cameraFollow || fly { glide.stop(); startAnimating() }
    }

    func selectStorm(_ id: Int, fly: Bool = true, follow: Bool = false) {
        choose(.storm(id: id))
        let position = play.store.storms?.storms.storms.first { $0.id == id }.map { SIMD3<Float>(Float($0.x), Float($0.y), 0) }
        if follow {
            setFollow(.storm(id: id))
        } else {
            setFollow(.none)
        }
        if fly {
            flyTo(position, distance: min(camera.distance, flyDistance(.region)))
        } else if follow {
            startAnimating()
        }
    }

    /// Selects a kin group: its territory is highlighted. `fly` goes to the centroid of its rows, `follow` keeps the camera on it.
    func selectKin(id: Int, fly: Bool = true, follow: Bool = false) {
        choose(.kin(id: id))
        if follow { setFollow(.lineage(kinID: id)) } else { setFollow(.none) }
        if fly {
            var state = FollowState(target: .lineage(kinID: id))
            flyTo(state.goal(store: play.store, size: worldSize), distance: min(camera.distance, flyDistance(.region)))
        } else if follow {
            startAnimating()
        }
    }

    func selectPool(_ index: Int) {
        choose(.pool(index: index))
        setFollow(.none)
        if let pool = play.store.environment?.environment.pools, pool.indices.contains(index) {
            flyTo(SIMD3(Float(pool[index].x), Float(pool[index].y), 0), distance: min(camera.distance, flyDistance(.region)))
        }
    }

    /// `Esc` or a click on nothing: nothing is selected or followed.
    func clearSelection() {
        choose(.none)
        cancelCameraMotion()
    }

    /// `F`: follow the selection with the camera and fly in to it.
    func followSelection() {
        switch selection {
        case .organism(let uid): selectOrganism(uid, follow: true, camera: true, fly: true)
        case .storm(let id): selectStorm(id, fly: true, follow: true)
        case .kin(let id): selectKin(id: id, fly: true, follow: true)
        case .pool, .none: tell("Select an organism, a storm or a lineage to follow it")
        }
    }

    /// A click on a chronicle row or a toast: fly to where it happened and select what it is about.
    func activate(_ event: WorldEvent) {
        activate(ChronicleAction.of(event))
    }

    /// A click on a place and an entity: an alert's arrow, a toast, a sentence of a report, a chronicle row.
    func activate(_ action: ChronicleAction?) {
        guard let action else { return }
        let place = action.at.map { SIMD3<Float>($0.x, $0.y, 0) }
        /// Where it happened; an action with no place goes nowhere.
        func fly(to point: SIMD3<Float>?) {
            if let point { flyTo(point, distance: min(camera.distance, flyDistance(.region))) }
        }
        switch action.select {
        case .storm(let id):
            if play.store.storms?.storms.storms.contains(where: { $0.id == id }) == true {
                selectStorm(id, fly: true, follow: false)
            } else {
                choose(.storm(id: id))
                setFollow(.none)
                fly(to: place)
            }
        case .kin(let id):
            choose(.kin(id: id))
            setFollow(.none)
            var state = FollowState(target: .lineage(kinID: id))
            fly(to: state.goal(store: play.store, size: worldSize) ?? place)
        case .organism(let uid):
            if play.store.listedIndex(ofUID: uid) != nil {
                selectOrganism(uid, follow: true, camera: true, fly: false)
            } else {
                // Not in the newest frame (it died): select it for the card and go to where it happened. Following it would
                // leave the camera on a target that never has a position.
                selectOrganism(uid, follow: false, camera: false, fly: false)
                fly(to: place)
            }
        case .pool, .none:
            setFollow(.none)
            fly(to: place)
        }
    }

    // MARK: Camera motion

    /// Whether the camera is moving by itself (following or flying).
    var cameraMotionPending: Bool { isFlying || !followState.target.isNone || headingEase != nil || zoomEase != nil }

    func setFollow(_ target: FollowTarget) {
        followState.set(target)
        if followTarget != target { followTarget = target }
        if target.isNone, !isFlying { glide.stop() }
    }

    /// A manual drag, `Esc`, a reset: whatever moved the camera by itself stops. The selection stays (except for `Esc`).
    func cancelCameraMotion() {
        zoomEase = nil
        setFollow(.none)
        isFlying = false
        flyGoal = nil
        flyDistance = nil
        flight = nil
        glide.stop()
    }

    /// A zoom: a flight under way stops where it is (any input does, DESIGN §36.5); a follow goes on, at the user's distance.
    func stopFlight() {
        zoomEase = nil
        if followState.target.isNone {
            isFlying = false
            flyGoal = nil
            flight = nil
            glide.stop()
        }
        flyDistance = nil
    }

    func resetInteraction() {
        selection = .none
        selectionHint = 0
        followState.set(.none)
        if followTarget != .none { followTarget = .none }
        glide = CameraGlide()
        isFlying = false
        flyGoal = nil
        flyDistance = nil
        flight = nil
        headingEase = nil
        zoomEase = nil
        grabAnchor = nil
        scrollLaw = ScrollLaw()
        toastQueue = ToastQueue()
        toasts = []
        ticker = Ticker()
        tickerLine = nil
        powers = PowersModel()
        watch = WatchSet()
        cardCollapsed = false
        chapterNotice = nil
        noticeExpires = nil
        noticeShownFor = 0
        lastCardChapter = 0
        attention.reset()
        setBadge(nil)
        overlay.setAim(point: nil, world: nil)
        successors = [:]
        lastAdvance = nil
        overlay.setHover(nil)
        overlay.setCastHover(nil)
        storage.stop()
    }

    /// The distance a fly ends at on something of `level`.
    func flyDistance(_ level: WorldLevel) -> Float {
        WorldLevels.flyDistance(level, size: worldSize, fovY: camera.fovY, aspect: aspect)
    }

    /// Flies the camera to `point` (when there is one) and to `distance`, ending at rest. Whatever followed stops. Not while
    /// following: a followed organism moves, so the glide of the follow carries the camera (and the distance) there instead of a
    /// path that would have to be planned again every frame. Otherwise the flight is the van Wijk arc (`FlightPath`): zoom out, pan,
    /// zoom in, at an even speed on the screen.
    func flyTo(_ point: SIMD3<Float>?, distance: Float?) {
        guard let point = point ?? flyGoal else { return }  // nowhere to go
        flyGoal = point
        flyDistance = distance.map(clampedDistance)
        isFlying = true
        glide.stop()
        flight = nil
        if followState.target.isNone {
            let goal = FlightPath.Pose(x: point.x, y: point.y, z: point.z, heading: pose.heading, tilt: pose.tilt,
                                       distance: flyDistance ?? pose.distance)
            flight = Flight(path: makePath(to: goal))
        }
        startAnimating()
    }

    /// `h`, `r`, the toolbar's Home: whatever follows or flies stops (the selection stays) and the camera flies to the Home view, the
    /// flat overview north up and facing the populated side (DESIGN §36.5).
    func home() {
        cancelCameraMotion()
        headingEase = nil
        guard let header = play.store.header else { return }
        let goal = homePose(header: header)
        isFlying = true
        flyGoal = SIMD3(goal.x, goal.y, goal.z)
        flyDistance = goal.distance
        flight = Flight(path: makePath(to: FlightPath.Pose(goal)), home: true)
        startAnimating()
    }

    private func makePath(to goal: FlightPath.Pose) -> FlightPath {
        let width = 2 * tan(pose.fovY / 2) * max(1, aspect)
        return FlightPath(from: FlightPath.Pose(pose), to: goal, size: worldSize, widthPerDistance: width, straight: reduceMotion())
    }

    func clampedDistance(_ d: Float) -> Float {
        let limit = WorldLevels.maxDistance(size: worldSize, fovY: pose.fovY, aspect: aspect, heading: pose.heading)
        return min(max(d, WorldLevels.minDistance), max(limit, WorldLevels.minDistance))
    }

    /// One step of the camera's own motion, `dt` seconds. True while it has not arrived.
    func advanceCamera(dt: Float) -> Bool {
        var easing = false
        if var ease = headingEase {
            let step = ease.step(dt: min(dt, 0.1))
            headingEase = step.done ? nil : ease
            editPose { $0.heading = step.heading }
            easing = !step.done
        }
        if var ease = zoomEase {
            let step = ease.step(dt: min(dt, 0.1))
            zoomEase = step.done ? nil : ease
            editPose { $0.distance = step.distance }
            easing = easing || !step.done
        }
        if followState.target.isNone, isFlying, let flight { return advanceFlight(flight, dt: dt) || easing }
        var goal: SIMD3<Float>?
        var distance: Float?
        if !followState.target.isNone {
            goal = followState.goal(store: play.store, size: worldSize)
            distance = isFlying ? flyDistance : nil
            if goal == nil, isFlying { goal = flyGoal }  // nothing to follow yet (a hand-off not in a frame): go where flown
        } else if isFlying {
            goal = flyGoal
            distance = flyDistance
        }
        if goal == nil, distance != nil, followState.target.isNone { goal = camera.target }  // a zoom only
        guard let goal else { return easing }
        glide.timeConstant = isFlying ? CameraGlide.flyConstant : CameraGlide.followConstant
        var moved = camera
        let moving = glide.advance(&moved, toward: goal, distance: distance, dt: dt, size: worldSize)
        editPose {
            $0.x = moved.target.x
            $0.y = moved.target.y
            $0.z = moved.target.z
            $0.distance = moved.distance
        }
        if !moving, isFlying {
            isFlying = false
            flyDistance = nil
            if followState.target.isNone { flyGoal = nil }
        }
        return moving || easing
    }

    /// One step of a flight: the pose `dt` seconds on its path (the end exactly when it is over).
    private func advanceFlight(_ current: Flight, dt: Float) -> Bool {
        var f = current
        f.elapsed += min(max(dt, 0), 0.1)
        let at = f.path.pose(at: f.elapsed)
        // Only the flight to Home turns the view (it goes north up and straight down): a flight to a place keeps the heading and the
        // tilt it had, and so lets a turn the player asked for meanwhile (`n`, an Option drag) stand.
        editPose {
            $0.x = at.x
            $0.y = at.y
            $0.z = at.z
            $0.distance = at.distance
            if f.home {
                $0.heading = at.heading
                $0.tilt = at.tilt
            }
        }
        let done = f.elapsed >= f.path.duration
        flight = done ? nil : f
        if done {
            isFlying = false
            flyDistance = nil
            flyGoal = nil
            if f.home { cameraIsHome = true }
        }
        return !done
    }

    // MARK: Events

    /// The events the world sent that the chronicle had not seen. A live `focus_handoff` from the selected organism moves the
    /// selection (and the camera's follow) to the new one; a toast is raised for the important ones; the backlog loaded at the
    /// start is history and does neither.
    func handle(events fresh: [WorldEvent], backlog: Bool) {
        guard !backlog else { return }
        for event in fresh where event.kind == EventKind.focusHandoff && event.b >= 0 {
            successors[Int64(event.a)] = Int64(event.b)
            if successors.count > 64 { successors = successors.filter { $0.key == Int64(event.a) } }
            var moved = selection
            guard moved.applyHandoff(event) else { continue }  // only when `from` is what is selected now
            setSelection(moved)
            if followState.target == .organism(uid: Int64(event.a)) { setFollow(.organism(uid: Int64(event.b))) }
            startAnimating()
        }
        // The alerts toast in a world that has an alert engine (DESIGN §34.7); an event toasts only where there is none.
        if play.store.play?.alerts == nil {
            let news = ChronicleReducer.toasts(from: fresh, backlog: backlog)
            if !news.isEmpty {
                toastQueue.push(news, at: now())
                toasts = toastQueue.toasts
            }
        }
        ticker.show(events: fresh, at: now())
        tickerLine = ticker.line
        scheduleTimers()
    }

    /// Drops the toasts whose time is up.
    func expireToasts() {
        guard toastQueue.expire(at: now()) else { return }
        toasts = toastQueue.toasts
    }

    func dismissToast(id: UInt64) {
        toastQueue.dismiss(id: id)
        toasts = toastQueue.toasts
    }

    // MARK: Keys

    func perform(_ action: WorldAction) {
        dismissHint()
        switch action {
        case .togglePause:
            if clockStrip.paused { play.resume() } else { play.pause() }
        case .step:
            play.step(1)
        case .cycleColor: cycleColour()
        case .toggleAgents: display.agents.toggle()
        case .toggleBonds: display.bonds.toggle()
        case .cycleTerritory: display.territory = display.territory.next
        case .followSelection: followSelection()
        case .slower, .faster:
            // Along the rungs of a running world; a paused one has no rate to change (space resumes it).
            guard !clockStrip.paused,
                  let current = play.requestedRate ?? play.store.clock.flatMap({ PlayRate(clock: $0.rate) }) else { return }
            if let next = action == .slower ? RateLadder.slower(than: current) : RateLadder.faster(than: current) {
                play.setRate(next)
            }
        case .home: home()
        case .northUp: northUp()
        case .toggleMinimap: display.minimap.toggle()
        case .escape: escape()
        case .showPowers: tab = .powers
        case .flyToAlert: flyToNewestAlert()
        case .acknowledgeAlert: acknowledgeNewest()
        case .continueRun:
            // Continue never pauses; a chapter notice goes with it.
            if clockStrip.paused { play.resume() }
            dismissNotice()
        case .confirmPower: if returnConfirmsPower { confirmPower() }
        }
    }

    /// The clock strip as the dock shows it (what the window asked wins over the last `CLCK`).
    var clockStrip: ClockStrip {
        let store = play.store
        let alertTitle = store.clock.flatMap { $0.reason == .alert ? store.alerts.pausing(atStep: Int($0.head.step))?.title : nil }
        return ClockStrip(clock: store.clock, requested: play.requestedRate, baseRate: store.play?.baseRate ?? 14.4,
                          chapterLabel: chapterBar.label, alertTitle: alertTitle)
    }

    // MARK: The inspector

    var inspectorCard: InspectorCard? {
        Inspector.card(for: selection, store: play.store, successors: successors)
    }
}

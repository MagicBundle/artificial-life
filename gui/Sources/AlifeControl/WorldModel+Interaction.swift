import AlifeCore
import CoreGraphics
import Foundation
import simd

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
        let pick = WorldPicker.pick(at: point, projector: projector(size: freeRect.size), store: play.store, display: display)
        switch pick {
        case .organism(let uid, _):
            selectOrganism(uid, follow: !shift, camera: !shift, fly: clicks >= 2)
        case .storm(let id):
            selectStorm(id, fly: true, follow: clicks >= 2)
        case .pool(let index):
            selectPool(index)
        case .territory(_, let kin):
            selectKin(id: kin, fly: false, follow: false)
        case .none:
            if clicks == 1 { clearSelection() }
        }
    }

    /// The pointer moved to `point`, or left the view (nil). No round trip to the world: the same hit test as a click.
    func hover(at point: CGPoint?) {
        if powers.isAiming {
            aimHover(at: point)
            overlay.setHover(nil)
            return
        }
        guard let point else {
            overlay.setHover(nil)
            return
        }
        let t = now()
        if lastHoverPick >= 0, t - lastHoverPick < Self.hoverPeriod, t >= lastHoverPick { return }  // (42k projections a pick)
        lastHoverPick = t
        let store = play.store
        let pick = WorldPicker.pick(at: point, projector: projector(size: freeRect.size), store: store, display: display)
        if let text = WorldPicker.describe(pick, store: store) {
            overlay.setHover(HoverInfo(point: point, pick: pick, text: text))
        } else {
            overlay.setHover(nil)
        }
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
    var cameraMotionPending: Bool { isFlying || !followState.target.isNone }

    func setFollow(_ target: FollowTarget) {
        followState.set(target)
        if followTarget != target { followTarget = target }
        if target.isNone, !isFlying { glide.stop() }
    }

    /// A manual drag, `Esc`, a reset: whatever moved the camera by itself stops. The selection stays (except for `Esc`).
    func cancelCameraMotion() {
        setFollow(.none)
        isFlying = false
        flyGoal = nil
        flyDistance = nil
        glide.stop()
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
    }

    /// The distance a fly ends at on something of `level`.
    func flyDistance(_ level: WorldLevel) -> Float {
        WorldLevels.flyDistance(level, size: worldSize, fovY: camera.fovY, aspect: aspect)
    }

    /// Flies the camera to `point` (when there is one) and to `distance`, ending at rest. Whatever followed stops.
    func flyTo(_ point: SIMD3<Float>?, distance: Float?) {
        guard let point = point ?? flyGoal else { return }  // nowhere to go
        flyGoal = point
        flyDistance = distance.map(clampedDistance)
        isFlying = true
        glide.stop()
        startAnimating()
    }

    func clampedDistance(_ d: Float) -> Float {
        let limit = WorldLevels.maxDistance(size: worldSize, fovY: camera.fovY, aspect: aspect)
        return min(max(d, WorldLevels.minDistance), max(limit, WorldLevels.minDistance))
    }

    /// One step of the camera's own motion, `dt` seconds. True while it has not arrived.
    func advanceCamera(dt: Float) -> Bool {
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
        guard let goal else { return false }
        glide.timeConstant = isFlying ? CameraGlide.flyConstant : CameraGlide.followConstant
        var moved = camera
        let moving = glide.advance(&moved, toward: goal, distance: distance, dt: dt, size: worldSize)
        edit { $0 = moved }
        if !moving, isFlying {
            isFlying = false
            flyDistance = nil
            if followState.target.isNone { flyGoal = nil }
        }
        return moving
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
        switch action {
        case .togglePause:
            if clockStrip.paused { play.resume() } else { play.pause() }
        case .step:
            play.step(1)
        case .cycleColor:
            let store = play.store
            let current = store.frame.flatMap { store.header?.colorMode(index: $0.colorIndex) }
            if let next = WorldKeys.nextColorMode(after: current, in: store.header?.colorModes ?? []) { play.setColor(next) }
        case .toggleAgents:
            if level == .planet { display.agentPoints.toggle() } else { display.agents.toggle() }
        case .toggleBonds: display.bonds.toggle()
        case .toggleTerritory: display.territory.toggle()
        case .followSelection: followSelection()
        case .slower, .faster:
            // Along the rungs of a running world; a paused one has no rate to change (space resumes it).
            guard !clockStrip.paused,
                  let current = play.requestedRate ?? play.store.clock.flatMap({ PlayRate(clock: $0.rate) }) else { return }
            if let next = action == .slower ? RateLadder.slower(than: current) : RateLadder.faster(than: current) {
                play.setRate(next)
            }
        case .resetCamera: resetCamera()
        case .escape: escape()
        case .showPowers: tab = .powers
        case .flyToAlert: flyToNewestAlert()
        case .acknowledgeAlert: acknowledgeNewest()
        case .continueRun:
            // Continue never pauses; a chapter notice goes with it.
            if clockStrip.paused { play.resume() }
            dismissNotice()
        case .confirmPower: confirmPower()
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

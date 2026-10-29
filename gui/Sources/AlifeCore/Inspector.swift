import Foundation

/// The card pinned at the bottom of the dock whenever something is selected (DESIGN §33.4.8): the words and numbers of one
/// selection, built from the records so that the card is tested without a window.
package struct InspectorCard: Equatable, Sendable {
    package struct Row: Equatable, Sendable {
        package var label: String
        package var value: String
        /// Tagged "provisional": a label of the observers, whose thresholds are not validated for storm worlds (C8 to C10).
        package var provisional = false

        package init(_ label: String, _ value: String, provisional: Bool = false) {
            self.label = label
            self.value = value
            self.provisional = provisional
        }
    }

    package var title: String
    package var subtitle: String?
    package var rows: [Row] = []
    /// An organism's energy over the cap, as a fraction in 0...1, with its words.
    package var energy: (fraction: Double, text: String)?
    /// Notes under the card: "dynasties approximate", "provisional (labels v2)".
    package var notes: [String] = []
    /// An organism's parent, when it can be selected: alive in the newest frame.
    package var selectableParent: Int64?
    /// The organism the follow passed to when this one died, once known.
    package var successor: Int64?
    package var alive = true
    /// The voice of the organism (comm feature): what it says and hears, per channel.
    package var voice: VoicePanel?

    package static func == (a: InspectorCard, b: InspectorCard) -> Bool {
        a.title == b.title && a.subtitle == b.subtitle && a.rows == b.rows && a.notes == b.notes
            && a.selectableParent == b.selectableParent && a.successor == b.successor && a.alive == b.alive
            && a.energy?.fraction == b.energy?.fraction && a.energy?.text == b.energy?.text && a.voice == b.voice
    }
}

/// The bars of an organism's voice (DESIGN §39.8, §45): W channels of what it says (after the step) and of what it hears (what drove the
/// step), and how many neighbours can hear it. The heard half is there only when the world is played with the observer channel.
package struct VoicePanel: Equatable, Sendable {
    package var says: [Double]
    /// Nil when the heard half is not valid: a panel then shows the spoken half only.
    package var hears: [Double]?
    package var audible: Int?

    package init(_ voice: PlayVoice) {
        says = voice.says.map(FeatureWords.fraction)
        hears = voice.heardValid ? voice.hears.map(FeatureWords.fraction) : nil
        audible = voice.heardValid ? voice.audible : nil
    }

    package var channels: Int { says.count }
    package var isSilent: Bool { says.allSatisfy { $0 == 0 } }
}

package enum Inspector {
    /// The card of the selection; nil for `.none`, a kin group that is no longer listed, or a storm that is gone (the card
    /// of a storm that ended says so).
    package static func card(for selection: WorldSelection, store: WorldStore, successors: [Int64: Int64] = [:]) -> InspectorCard? {
        switch selection {
        case .none: return nil
        case .organism(let uid): return organism(uid: uid, store: store, successor: successors[uid])
        case .storm(let id): return storm(id: id, store: store)
        case .kin(let id): return kin(id: id, store: store)
        case .pool(let index): return pool(index: index, store: store)
        }
    }

    // MARK: An organism

    /// The organism's card from its `FOCS` (about four a second, and now when it is picked) with `IDNT` for the clade and the
    /// tables its slots index, and `INTN`'s labels only through `FOCS` (a newborn is complete). A `FOCS` of another organism
    /// is not this one's: then the card has what the frame knows (that it is alive, and where).
    package static func organism(uid: Int64, store: WorldStore, successor: Int64? = nil) -> InspectorCard {
        let header = store.play
        let tables = store.identities?.tables
        var card = InspectorCard(title: "Organism \(uid)")
        let frameIndex = store.listedIndex(ofUID: uid)
        let focus = store.focus.flatMap { $0.uid == uid ? $0 : nil }

        if let focus, !focus.isAlive {
            card.alive = false
            card.subtitle = "no longer alive"
            card.successor = successor
            if let successor { card.rows.append(.init("Follow passed to", "organism \(successor)")) }
            return card
        }
        if frameIndex == nil, store.frame != nil, focus == nil {
            card.alive = false
            card.subtitle = "not in the world"
            card.successor = successor
            if let successor { card.rows.append(.init("Follow passed to", "organism \(successor)")) }
            return card
        }

        if let focus, let tables {
            let kin = Names.kin(slot: focus.kin, tables: tables)
            let house = Names.house(slot: focus.house, tables: tables)
            card.subtitle = "\(kin) · House \(house)" + (tables.exact ? "" : " (approximate)")
            if !tables.exact { card.notes.append("dynasties approximate: this world's houses are the clades of the fork") }
        } else if let focus {
            card.subtitle = "kin slot \(focus.kin), house slot \(focus.house)"
        }

        if let row = store.identity(ofUID: uid) {
            card.rows.append(.init("Clade", String(row.clade)))
        }
        if let focus {
            if let labels = header?.labels {
                let validity = LabelValidity(labels: labels, environment: store.environment?.environment)
                let evaluated = focus.labelFlags & 1 != 0
                let archValid = evaluated && validity.archetypeIsValid(focus.archPrimary)
                    && (focus.archSecondary == 0 || validity.archetypeIsValid(focus.archSecondary))
                let intentValid = evaluated && validity.intentIsValid(focus.intent)
                card.rows.append(
                    .init("Archetype", archetype(focus, names: labels.arch), provisional: !archValid))
                card.rows.append(.init("Intent", intent(focus, names: labels.intent), provisional: !intentValid))
                card.notes.append(validity.note(allValid: archValid && intentValid))
            }
            let cap = Double(header?.energyCap ?? 0)
            if cap > 0 {
                let fraction = min(max(Double(focus.energy) / cap, 0), 1)
                card.energy = (fraction, "\(focus.energy) of \(Int(cap))")
            }
            card.rows.append(.init("Age", ageText(steps: Int(focus.age), baseRate: header?.baseRate ?? 14.4)))
            card.rows.append(.init("Generation", String(focus.generation)))
            card.rows.append(.init("Bonds", String(focus.nBonds)))
            if focus.parentUid >= 0 {
                card.rows.append(.init("Parent", "organism \(focus.parentUid)"))
                if store.listedIndex(ofUID: focus.parentUid) != nil { card.selectableParent = focus.parentUid }
            } else {
                card.rows.append(.init("Parent", "none (a founder or an immigrant)"))
            }
            if focus.isFollowed { card.rows.append(.init("Follow", "passes to a child at its death")) }
            addFeatureRows(&card, uid: uid, frameIndex: frameIndex, store: store)
        } else if let index = frameIndex, let frame = store.frame {
            let p = frame.position(index)
            card.subtitle = "waiting for its report"
            card.rows.append(.init("Position", String(format: "%.1f, %.1f", p.x, p.y)))
        }
        card.successor = successor
        if let successor { card.rows.append(.init("Follow passed to", "organism \(successor)")) }
        return card
    }

    /// The rows and the voice of the feature layers the world carries (DESIGN §45): the role (read from the colour the world paints in the
    /// role mode), the net flow along its bonds, the material and the soil under it, and its voice.
    package static func addFeatureRows(_ card: inout InspectorCard, uid: Int64, frameIndex: Int?, store: WorldStore) {
        let layers = store.layers
        guard layers.any, let header = store.header else { return }
        if let index = frameIndex, let frame = store.frame {
            if layers.roles {
                if header.colorMode(index: frame.colorIndex) == RoleColours.mode {
                    if let role = FeatureWords.role(forColour: frame.rgb[index]) {
                        card.rows.append(.init("Role", role))
                    } else {
                        card.rows.append(.init("Role", "hurt in the last step (red)"))
                    }
                } else {
                    card.notes.append("colour by role (C) to see its role")
                }
            }
            if let flow = frame.flow, let scale = store.play?.bonds?.scale {
                var net = 0.0, count = 0
                for k in 0..<frame.nBonds {
                    let (i, j) = frame.bond(k)
                    guard i == index || j == index else { continue }
                    count += 1
                    let q = BondFlowLook.quanta(flow[k], scale: scale)
                    net += j == index ? q : -q  // positive flow runs from the lower row to the higher
                }
                if count > 0 {
                    card.rows.append(.init("Bond flow", String(format: "%+.0f q/step net over %d bond%@", net, count, count == 1 ? "" : "s")))
                }
            }
            let p = frame.position(index)
            if let grid = store.play?.structures, let material = store.material {
                let level = material.level(atX: p.x, y: p.y, z: p.z, cell: grid.cell)
                card.rows.append(.init("Built here", level == 0 ? "nothing" : FeatureWords.cellText(level: level, cap: grid.cap)))
            }
            if let grid = store.play?.niche, let soil = store.soil {
                let level = soil.level(atX: p.x, y: p.y, z: p.z, cell: grid.cell)
                card.rows.append(.init("Soil here", level == 0 ? "none" : FeatureWords.cellText(level: level, cap: grid.cap)))
            }
        }
        if let voice = store.focusVoice, voice.uid == uid { card.voice = VoicePanel(voice) }
    }

    /// `"Gatherer"` or `"Gatherer / Hunter"`; "warming up" before the labels are evaluated. Flag bit 0 is the *intent*
    /// warm-up (24 observed steps); the archetype warms up longer (192), and until then its byte is 0 (once warm the
    /// primary is never 0: no match is Generalist), so a 0 primary is "warming up" whatever the flag says.
    package static func archetype(_ focus: PlayFocus, names: [String]) -> String {
        guard focus.labelFlags & 1 != 0, focus.archPrimary != 0 else { return "warming up" }
        let first = Names.label(focus.archPrimary, in: names)
        guard focus.archSecondary != 0 else { return first }
        return "\(first) / \(Names.label(focus.archSecondary, in: names))"
    }

    /// The intent's name; the Budding overlay (bit 1) is a forecast and says so.
    package static func intent(_ focus: PlayFocus, names: [String]) -> String {
        guard focus.labelFlags & 1 != 0 else { return "warming up" }
        let name = Names.label(focus.intent, in: names)
        return focus.labelFlags & 2 != 0 ? "\(name) (forecast)" : name
    }

    /// `"1,250 steps (1.4 game minutes)"`.
    package static func ageText(steps: Int, baseRate: Double) -> String {
        let minutes = CastPolicy.ageMinutes(steps: steps, baseRate: baseRate)
        return "\(steps) steps (" + String(format: "%.1f", minutes) + " game min)"
    }

    // MARK: A storm, a kin group, a pool

    package static func storm(id: Int, store: WorldStore) -> InspectorCard {
        let baseRate = store.play?.baseRate ?? 14.4
        guard let storm = store.storms?.storms.storms.first(where: { $0.id == id }) else {
            var card = InspectorCard(title: "Storm \(id)")
            card.subtitle = "dissipated or merged"
            card.alive = false
            return card
        }
        var card = InspectorCard(title: Names.storm(storm).capitalizedFirst)
        card.subtitle = (storm.hemi == 0 ? "northern" : "southern") + " hemisphere"
            + (storm.state == "coasting" ? ", coasting" : "")
        card.rows = [
            .init("Peak vorticity", String(format: "%.2f f0", storm.zeta)),
            .init("Vorticity now", String(format: "%.2f f0", storm.zetaNow)),
            .init("Wind", String(format: "%.3f", storm.wind)),
            .init("Rain", String(format: "%.4f", storm.rain)),
            .init("Radius", String(format: "%.1f units", storm.r)),
            .init("Age", ageText(steps: storm.age, baseRate: baseRate)),
        ]
        if storm.parent > 0 { card.rows.append(.init("Split from", "storm \(storm.parent)")) }
        if !storm.named { card.notes.append("a disturbance: it has not reached named strength (0.5 f0)") }
        return card
    }

    package static func kin(id: Int, store: WorldStore) -> InspectorCard? {
        guard let cast = CastPolicy.lineages(store.identities, territory: store.territory).first(where: { $0.id == id }) else {
            var card = InspectorCard(title: "Kin group \(id)")
            card.subtitle = "not among the groups listed"
            card.alive = false
            return card
        }
        var card = InspectorCard(title: cast.name)
        card.subtitle = cast.namedAfter.map { "named after House \($0)" }
        card.rows = [
            .init("Members", "\(cast.size) (" + String(format: "%.1f", 100 * cast.share) + "% of the population)"),
            .init("Territory", "\(cast.territoryCells) cells (dominance)"),
        ]
        if let births = cast.births { card.rows.append(.init("Births", Numbers.grouped(births))) }
        if let deaths = CastPolicy.deathSummary(cast.deaths) { card.rows.append(.init("Deaths", deaths)) }
        return card
    }

    package static func pool(index: Int, store: WorldStore) -> InspectorCard {
        var card = InspectorCard(title: "Warm pool")
        guard let pools = store.environment?.environment.pools, pools.indices.contains(index) else {
            card.alive = false
            return card
        }
        let pool = pools[index]
        card.subtitle = "a warm patch of the sea that feeds storms"
        card.rows = [
            .init("Position", String(format: "%.0f, %.0f", pool.x, pool.y)),
            .init("Strength", String(format: "%.2f", pool.amp)),
        ]
        return card
    }
}

extension String {
    /// The first letter in capitals ("disturbance 9" -> "Disturbance 9"); the rest as it is.
    package var capitalizedFirst: String {
        guard let first else { return self }
        return first.uppercased() + dropFirst()
    }
}

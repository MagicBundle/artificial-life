import Foundation

// The records `alife play` adds to the stream (ALSTRM v1 additions), decoded from DESIGN.md §33.1, which is normative.
// A record is a 4-byte tag, a `u32` payload length (a multiple of 4) and a payload that starts with the common head
// `u64 step, u32 seq, u16 version, u16 flags`. `FrameDecoder` hands the payload of a known tag with version 1 to
// `RecordDecoder`; a record of an unknown tag or version is skipped by its length (and a version skip is counted).
// Value types only: the frames of 1.3 MB never pass through a published property, they are read by the renderer.

// MARK: - The play header key

/// `header["play"]`: what the writer sends and the constants the client needs (absent in a plain `alife stream`).
package struct PlayHeader: Equatable, Sendable, Codable {
    package struct Weather: Equatable, Sendable, Codable {
        package struct Refs: Equatable, Sendable, Codable {
            package var p: Double
            package var dc: Double
            package var u: Double
            package var zeta: Double
        }
        package var enabled: Bool
        package var cells: Int
        package var dx: Double
        package var f0: Double
        package var coarse: Int
        package var nameZeta: Double
        package var refs: Refs

        private enum CodingKeys: String, CodingKey {
            case enabled, cells, dx, f0, coarse, refs
            case nameZeta = "name_zeta"
        }
    }

    package struct Labels: Equatable, Sendable, Codable {
        package var intentVersion: Int
        /// Display names by byte value (index 0 is "none").
        package var arch: [String]
        package var intent: [String]

        private enum CodingKeys: String, CodingKey {
            case intentVersion = "intent_version"
            case arch, intent
        }
    }

    package struct Cast: Equatable, Sendable, Codable {
        package var kin: Int
        package var house: Int
        package var colony: Int
    }

    package struct Planet: Equatable, Sendable, Codable {
        package var source: String?
        /// `"exact"`, `"clade"` or `"none"`.
        package var houses: String
    }

    /// The tags the writer actually sends (a 2D or dry world has no `MAP1` and `STRM`).
    package var records: [String]
    package var versions: [String: Int]
    /// Steps per second at 1x.
    package var baseRate: Double
    package var rates: [Double]
    /// The detection cadence of the storm tracker, in steps.
    package var tick: Int
    package var snapshotEvery: Int
    package var frameHz: Int
    /// Steps per chapter (G3b).
    package var chapter: Int
    package var weather: Weather
    package var labels: Labels
    package var cast: Cast
    package var energyCap: Int
    /// The event kind registry: kind number (as text) to name.
    package var events: [String: String]
    package var planet: Planet

    private enum CodingKeys: String, CodingKey {
        case records, versions, rates, tick, chapter, weather, labels, cast, events, planet
        case baseRate = "base_rate"
        case snapshotEvery = "snapshot_every"
        case frameHz = "frame_hz"
        case energyCap = "energy_cap"
    }

    /// Refuses values that cannot describe a world (a bad rate, a tick of zero, a negative cap).
    func validate() throws {
        func bad(_ why: String) -> StreamError { .malformedHeader("play: \(why)") }
        guard baseRate.isFinite, baseRate > 0 else { throw bad("base_rate \(baseRate) is not positive") }
        guard rates.allSatisfy({ $0.isFinite && $0 > 0 }) else { throw bad("a rate is not positive") }
        guard tick > 0, snapshotEvery > 0, frameHz > 0, chapter > 0 else { throw bad("a cadence is not positive") }
        guard snapshotEvery % tick == 0 else { throw bad("snapshot_every \(snapshotEvery) is not a multiple of tick \(tick)") }
        guard energyCap > 0 else { throw bad("energy_cap \(energyCap) is not positive") }
        guard weather.cells > 0, weather.coarse > 0, weather.cells % weather.coarse == 0 else {
            throw bad("weather.cells \(weather.cells) is not a multiple of weather.coarse \(weather.coarse)")
        }
        guard weather.dx.isFinite, weather.dx > 0 else { throw bad("weather.dx is not positive") }
        guard Set(records).count == records.count else { throw bad("records lists a tag twice") }
    }
}

// MARK: - Common pieces

/// `u64 step, u32 seq, u16 version, u16 flags`: the head of every record.
package struct RecordHead: Equatable, Sendable {
    /// The step of the state the record describes.
    package var step: UInt64
    /// Per tag, from 0; a gap means a lost record (none in a pipe).
    package var seq: UInt32
    package var version: UInt16
    /// Tag-specific; bits a version does not define are zero and ignored.
    package var flags: UInt16

    package init(step: UInt64, seq: UInt32, version: UInt16, flags: UInt16) {
        self.step = step
        self.seq = seq
        self.version = version
        self.flags = flags
    }
}

/// A JSON value of a shape the record schema leaves open (a lever value, an effect's parameters).
package enum JSONValue: Equatable, Sendable, Codable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    package init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let v = try? c.decode(Bool.self) {
            self = .bool(v)
        } else if let v = try? c.decode(Double.self) {
            self = .number(v)
        } else if let v = try? c.decode(String.self) {
            self = .string(v)
        } else if let v = try? c.decode([JSONValue].self) {
            self = .array(v)
        } else {
            self = .object(try c.decode([String: JSONValue].self))
        }
    }

    package func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }

    package var number: Double? { if case .number(let v) = self { v } else { nil } }
    package var string: String? { if case .string(let v) = self { v } else { nil } }
}

/// The slot byte values of `IDNT`, `TERR` and `FOCS`: a slot below `MAX_TABLE` (in `IDNT` the `slot` field of a table
/// entry, which need not be the entry's position because a slot is sticky and a group can leave a gap; in `TERR` an
/// index into `ids`), or one of the reserved values.
package enum Slot {
    package static let maxTable = 16
    /// The house of an immigrant lineage of an observer attached to a world already running.
    package static let unknown: UInt8 = 253
    /// Alive but listed in no top-16 table.
    package static let other: UInt8 = 254
    package static let none: UInt8 = 255

    package static func isReserved(_ slot: UInt8) -> Bool { slot >= 253 }
    /// Whether `slot` is valid for a table of `size` rows.
    package static func isValid(_ slot: UInt8, tableSize size: Int) -> Bool { Int(slot) < size || isReserved(slot) }
    /// Whether `slot` is the slot of one of a table's entries, or reserved.
    package static func isValid(_ slot: UInt8, among slots: Set<Int>) -> Bool {
        slots.contains(Int(slot)) || isReserved(slot)
    }
}

// MARK: - FRMA

/// `FRMA`: the alive rows of the state at `head.step`. Per-agent arrays hold `n` entries in ascending row order;
/// `rows[i]` is the state row of listed index `i`. Bonds index the listed rows, not slots.
package struct PlayFrame: Equatable, Sendable {
    package var head: RecordHead
    /// Alive rows listed (the popcount of `bitmap`).
    package var n: Int
    /// `N`: the state's capacity (equal to the header's).
    package var capacity: Int
    package var colorIndex: Int
    /// Steps since the previous `FRMA` written (0 for the first); informational.
    package var span: Int
    /// `ceil(N/32)` words, bit `r` (LSB first) set when row `r` is alive.
    package var bitmap: [UInt32]
    /// The state row of each listed index.
    package var rows: [UInt32]
    /// `3n` floats: x, y, z (z is 0 in 2D).
    package var pos: [Float]
    /// `2n` binary16 bit patterns: yaw then pitch.
    package var orient: [UInt16]
    /// `n` colours `0x00RRGGBB` (hurt agents red in every mode).
    package var rgb: [UInt32]
    package var uid: [Int64]
    /// `2·nBonds` listed indices: pairs `(i, j)` with `i < j < n`.
    package var bonds: [UInt32]

    package init(head: RecordHead, n: Int, capacity: Int, colorIndex: Int, span: Int, bitmap: [UInt32], rows: [UInt32], pos: [Float],
                 orient: [UInt16], rgb: [UInt32], uid: [Int64], bonds: [UInt32]) {
        self.head = head
        self.n = n
        self.capacity = capacity
        self.colorIndex = colorIndex
        self.span = span
        self.bitmap = bitmap
        self.rows = rows
        self.pos = pos
        self.orient = orient
        self.rgb = rgb
        self.uid = uid
        self.bonds = bonds
    }

    package var nBonds: Int { bonds.count / 2 }
    /// Bit 0: the state has an invariant (error) diag bit set.
    package var hasDiagError: Bool { head.flags & 1 != 0 }
    package var step: UInt64 { head.step }

    package func position(_ i: Int) -> SIMD3<Float> { SIMD3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]) }
    package func yaw(_ i: Int) -> Float { Half.float(orient[2 * i]) }
    package func pitch(_ i: Int) -> Float { Half.float(orient[2 * i + 1]) }
    package func bond(_ k: Int) -> (i: Int, j: Int) { (Int(bonds[2 * k]), Int(bonds[2 * k + 1])) }

    package func isAlive(row: Int) -> Bool {
        row >= 0 && row < capacity && bitmap[row >> 5] & (1 << UInt32(row & 31)) != 0
    }

    /// The listed index of state row `row`: the number of alive rows below it. Nil when the row is dead.
    package func listedIndex(ofRow row: Int) -> Int? {
        guard isAlive(row: row) else { return nil }
        var below = 0
        for w in 0..<(row >> 5) { below += bitmap[w].nonzeroBitCount }
        let mask = (UInt32(1) << UInt32(row & 31)) &- 1
        return below + (bitmap[row >> 5] & mask).nonzeroBitCount
    }
}

// MARK: - IDNT

/// A row of `IDNT`: an alive row's identity (16 bytes on the wire).
package struct IdentityRow: Equatable, Sendable {
    package var uid: Int64
    package var clade: Int32
    /// Slots into the record's tables; 253 to 255 are reserved (`Slot`).
    package var house: UInt8
    package var kin: UInt8
    package var colony: UInt8
    /// Bit 0: born after the last snapshot (its slots are approximate).
    package var flags: UInt8

    package var bornSinceSnapshot: Bool { flags & 1 != 0 }
}

/// The JSON tables of `IDNT`: the top-16 houses, kin groups and colonies of the last snapshot.
package struct IdentityTables: Equatable, Sendable, Codable {
    package struct House: Equatable, Sendable, Codable {
        package var slot: Int
        package var id: Int
        package var name: String
        package var size: Int
        package var clade: Int
    }

    package struct Kin: Equatable, Sendable, Codable {
        package var slot: Int
        package var id: Int
        package var name: String
        /// The id of the dominant house the group is named after, if any.
        package var namedAfter: Int?
        package var size: Int
        package var clade: Int
        /// Nil only for a group the observer lists with no live row (a derived table).
        package var cladeShare: Double?
        /// Nil while the group has too few members to sample.
        package var simSelf: Double?
        package var x: Double?
        package var y: Double?
        package var z: Double?
        /// The mean archetype shares (seven values), nil while the labels warm up.
        package var arch: [Double]?
        /// Births in the last window, nil for a group that held no slot then.
        package var births: Int?
        /// Deaths in the last window by cause: starved, killed, aged, catastrophe (and storm in a weather world).
        package var deaths: [Int]?

        private enum CodingKeys: String, CodingKey {
            case slot, id, name, size, clade, x, y, z, arch, births, deaths
            case namedAfter = "named_after"
            case cladeShare = "clade_share"
            case simSelf = "sim_self"
        }
    }

    package struct Colony: Equatable, Sendable, Codable {
        package var slot: Int
        package var id: Int
        package var size: Int
        package var kin: Int
        package var x: Double?
        package var y: Double?
    }

    package var houses: [House]
    /// Whether the houses are the exact lineages (true) or the clade approximation of a fork.
    package var exact: Bool
    package var origin: Int
    package var kin: [Kin]
    package var colonies: [Colony]
    package var unknownImmigrant: Int

    private enum CodingKeys: String, CodingKey {
        case houses, exact, origin, kin, colonies
        case unknownImmigrant = "unknown_immigrant"
    }
}

/// `IDNT`: who is who, at the last snapshot (rows) with the tables the slots index.
package struct PlayIdentities: Equatable, Sendable {
    package var head: RecordHead
    package var snapshotStep: UInt32
    package var tables: IdentityTables
    /// Alive rows, ascending row order (the client matches them by uid).
    package var rows: [IdentityRow]
}

// MARK: - INTN

package struct LabelRow: Equatable, Sendable {
    package var uid: Int64
    /// `rules.Archetype` values (0 = none).
    package var archPrimary: UInt8
    package var archSecondary: UInt8
    /// `rules.Intent` value (0 = none).
    package var intent: UInt8
    package var flags: UInt8

    /// Bit 0: evaluated (past warm-up).
    package var evaluated: Bool { flags & 1 != 0 }
    /// Bit 1: the Budding overlay (a forecast).
    package var buddingOverlay: Bool { flags & 2 != 0 }
}

/// `INTN`: the observer's labels (provisional: `intentVersion`).
package struct PlayLabels: Equatable, Sendable {
    package var head: RecordHead
    /// The last evaluation (at most the record's step).
    package var labelsStep: UInt32
    package var intentVersion: UInt32
    package var rows: [LabelRow]
}

// MARK: - TERR

/// `TERR`: the dominance map, `ny` rows of `nx` cells, row-major (`owner[y * nx + x]`).
package struct PlayTerritory: Equatable, Sendable {
    package var head: RecordHead
    package var nx: Int
    package var ny: Int
    package var snapshotStep: UInt32
    /// The kin group id of each owner slot.
    package var ids: [Int]
    /// A slot of `ids`, or 255 for no owner.
    package var owner: [UInt8]
    /// `round(purity_q16 / 257)`.
    package var purity: [UInt8]
}

// MARK: - STRM

/// `STRM`: the storm tracker's state at a detection tick.
package struct PlayStorms: Equatable, Sendable, Codable {
    package struct Storm: Equatable, Sendable, Codable {
        package var id: Int
        /// Nil for a disturbance: only storms that reached 0.5 f0 are named.
        package var name: String?
        package var named: Bool
        package var hemi: Int
        /// Wrapped into `[0, L)`.
        package var x: Double
        package var y: Double
        package var r: Double
        package var ext: [Double]
        /// Peak lower-layer vorticity over f0 so far.
        package var zeta: Double
        package var zetaNow: Double
        package var wind: Double
        package var rain: Double
        /// Steps since the track began.
        package var age: Int
        package var parent: Int
        /// `"active"` or `"coasting"`.
        package var state: String
        /// `[step, ux, uy]` with unwrapped coordinates, at most 24.
        package var track: [[Double]]
        /// `[lead, x, y]` for the leads 250, 500 and 1000.
        package var pred: [[Double]]

        private enum CodingKeys: String, CodingKey {
            case id, name, named, hemi, x, y, r, ext, zeta, wind, rain, age, parent, state, track, pred
            case zetaNow = "zeta_now"
        }
    }

    package struct Ended: Equatable, Sendable, Codable {
        package var id: Int
        /// `"dissipated"` or `"merged"`.
        package var reason: String
        package var into: Int
    }

    package var detectStep: Int
    package var storms: [Storm]
    package var ended: [Ended]

    private enum CodingKeys: String, CodingKey {
        case storms, ended
        case detectStep = "detect_step"
    }
}

/// `STRM` with its head and the tick it was made at.
package struct PlayStormRecord: Equatable, Sendable {
    package var head: RecordHead
    package var tick: UInt32
    package var storms: PlayStorms
}

// MARK: - MAP1

/// `MAP1`: the weather maps, planes row-major `[cy][cx]` with `n x n` cells (and winds as `4 x nw x nw`).
package struct PlayWeatherMap: Equatable, Sendable {
    package var head: RecordHead
    package var n: Int
    package var nw: Int
    package var f0: Float
    package var pRef: Float
    package var dcRef: Float
    package var uRef: Float
    package var zetaRef: Float
    /// Rain rate, companded: `rain = (byte / 255)^2 * pRef`.
    package var p: [UInt8]
    /// Cloud depth, `byte / 255 * dcRef`.
    package var dc: [UInt8]
    /// Layer-1 vorticity, `byte / 127 * zetaRef`.
    package var zeta: [Int8]
    /// The warm-pool (sea) map, `byte / 255`.
    package var ts: [UInt8]
    /// Block means of `u1, v1, u2, v2`, `byte / 127 * uRef`; plane `k` starts at `k * nw * nw`.
    package var winds: [Int8]

    /// Companded decode of a rain byte.
    package func rain(_ byte: UInt8) -> Float { let u = Float(byte) / 255; return u * u * pRef }
}

// MARK: - EVNT

/// One event of the chronicle (the closed template table of the server writes `text`).
package struct WorldEvent: Equatable, Sendable, Codable {
    /// Monotone per run (`step * 1000 + index`), so a resume keeps order; the client deduplicates by it.
    package var id: UInt64
    package var step: Int
    /// A key of `PlayHeader.events`; an unknown kind is shown by its `text`.
    package var kind: Int
    /// `"kin"`, `"house"`, `"storm"`, `"focus"`, `"director"` or `"system"`.
    package var fam: String
    package var a: Int
    package var b: Int
    package var x: Double?
    package var y: Double?
    package var z: Double?
    /// 2 = a toast, 1 = the chronicle, 0 = the log only.
    package var imp: Int
    package var text: String

    package init(id: UInt64, step: Int, kind: Int, fam: String, a: Int, b: Int, x: Double?, y: Double?, z: Double?, imp: Int, text: String) {
        self.id = id
        self.step = step
        self.kind = kind
        self.fam = fam
        self.a = a
        self.b = b
        self.x = x
        self.y = y
        self.z = z
        self.imp = imp
        self.text = text
    }
}

/// `EVNT`: up to 64 events; queued (never dropped) by the client.
package struct PlayEventBatch: Equatable, Sendable {
    package var head: RecordHead
    package var events: [WorldEvent]
    /// Bit 0 of the head's flags: the history rebuilt at start.
    package var isBacklog: Bool { head.flags & 1 != 0 }
}

// MARK: - ENVS

/// `ENVS`: the environment (warm pools at this step, levers, the director's effects and cooldowns).
package struct PlayEnvironment: Equatable, Sendable, Codable {
    package struct Pool: Equatable, Sendable, Codable {
        package var x: Double
        package var y: Double
        package var amp: Double
        package var sx: Double
        package var sy: Double
        package var vx: Double
        package var vy: Double
    }

    package struct Season: Equatable, Sendable, Codable {
        package var period: Double
        package var amp: Double
        package var phase: Double
    }

    package struct Effect: Equatable, Sendable, Codable {
        package var action: String
        package var params: [String: JSONValue]
        package var start: Int
        package var end: Int?
        /// `"rule"`, `"llm"` or `"player"`.
        package var source: String
    }

    package struct Cooldown: Equatable, Sendable, Codable {
        package var action: String
        package var ready: Int
    }

    package var pools: [Pool]
    package var season: Season?
    package var levers: [String: JSONValue]
    package var effects: [Effect]
    package var cooldowns: [Cooldown]
    /// Null in G3a.
    package var influence: JSONValue?
}

package struct PlayEnvironmentRecord: Equatable, Sendable {
    package var head: RecordHead
    package var environment: PlayEnvironment
}

// MARK: - CLCK and FOCS

/// `CLCK`: the clock and the scheduler's counters.
package struct PlayClock: Equatable, Sendable {
    package enum Reason: UInt8, Sendable {
        case running = 0, userPause, endOfSteps, chapter, alert, error
    }

    package var head: RecordHead
    /// The requested multiplier: 0 paused, -1 max.
    package var rate: Float
    /// Steps per second over the last 2 s.
    package var achieved: Float
    package var stepMs: Float
    package var frameMs: Float
    package var lateSteps: UInt32
    package var saveStallSteps: UInt32
    package var reason: Reason
    /// A checkpoint is being written.
    package var saving: Bool
    package var flags: UInt16
    /// Steps between frames.
    package var fe: UInt32
    package var chapter: UInt32
    package var inChapter: UInt32
    package var chapterLen: UInt32
}

/// `FOCS`: the focused organism's own row with the observer's current slots.
package struct PlayFocus: Equatable, Sendable {
    package var head: RecordHead
    package var uid: Int64
    package var row: UInt32
    /// Bit 0 alive, bit 1 hand-off this frame, bit 2 followed.
    package var flags: UInt32
    package var x: Float
    package var y: Float
    package var z: Float
    package var vx: Float
    package var vy: Float
    package var vz: Float
    package var energy: Int32
    package var age: UInt32
    package var generation: UInt32
    package var parentUid: Int64
    package var nBonds: UInt8
    package var house: UInt8
    package var kin: UInt8
    package var colony: UInt8
    package var archPrimary: UInt8
    package var archSecondary: UInt8
    package var intent: UInt8
    package var labelFlags: UInt8

    package var isAlive: Bool { flags & 1 != 0 }
    package var handedOff: Bool { flags & 2 != 0 }
    package var isFollowed: Bool { flags & 4 != 0 }
}

// MARK: - The record enum

/// A decoded play record.
package enum StreamRecord: Equatable, Sendable {
    case frma(PlayFrame)
    case idnt(PlayIdentities)
    case intn(PlayLabels)
    case terr(PlayTerritory)
    case strm(PlayStormRecord)
    case map1(PlayWeatherMap)
    case evnt(PlayEventBatch)
    case envs(PlayEnvironmentRecord)
    case clck(PlayClock)
    case focs(PlayFocus)

    /// The record's tag as on the wire.
    package var tag: String {
        switch self {
        case .frma: "FRMA"
        case .idnt: "IDNT"
        case .intn: "INTN"
        case .terr: "TERR"
        case .strm: "STRM"
        case .map1: "MAP1"
        case .evnt: "EVNT"
        case .envs: "ENVS"
        case .clck: "CLCK"
        case .focs: "FOCS"
        }
    }

    package var head: RecordHead {
        switch self {
        case .frma(let r): r.head
        case .idnt(let r): r.head
        case .intn(let r): r.head
        case .terr(let r): r.head
        case .strm(let r): r.head
        case .map1(let r): r.head
        case .evnt(let r): r.head
        case .envs(let r): r.head
        case .clck(let r): r.head
        case .focs(let r): r.head
        }
    }
}

// MARK: - Bytes

/// Little-endian reads from a record's payload; `offset` is relative to the payload's first byte (the common head).
struct ByteReader {
    let raw: UnsafeRawBufferPointer
    /// Where the payload starts in `raw`.
    let start: Int

    func u8(_ offset: Int) -> UInt8 { raw[start + offset] }
    func u16(_ offset: Int) -> UInt16 { raw.loadUnaligned(fromByteOffset: start + offset, as: UInt16.self) }
    func u32(_ offset: Int) -> UInt32 { raw.loadUnaligned(fromByteOffset: start + offset, as: UInt32.self) }
    func u64(_ offset: Int) -> UInt64 { raw.loadUnaligned(fromByteOffset: start + offset, as: UInt64.self) }
    func i64(_ offset: Int) -> Int64 { Int64(bitPattern: u64(offset)) }
    func f32(_ offset: Int) -> Float { Float(bitPattern: u32(offset)) }

    /// `count` values of `T` copied from `offset` (the host is little endian).
    func copy<T: BitwiseCopyable>(count n: Int, at offset: Int) -> [T] {
        [T](unsafeUninitializedCapacity: n) { target, initialized in
            let bytes = n * MemoryLayout<T>.stride
            if bytes > 0 {
                UnsafeMutableRawBufferPointer(target).copyMemory(
                    from: UnsafeRawBufferPointer(rebasing: raw[(start + offset)..<(start + offset + bytes)]))
            }
            initialized = n
        }
    }

    func bytes(at offset: Int, count n: Int) -> UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(rebasing: raw[(start + offset)..<(start + offset + n)])
    }
}

// MARK: - Decoding

/// The decoder of the play records: one pure function per question the stream decoder asks.
enum RecordDecoder {
    static let version: UInt16 = 1
    static let headBytes = 16
    /// The largest JSON a record may carry; a corrupt length is refused before the payload is waited for.
    static let maxJSON = 1 << 20
    static let maxEvents = 64
    static let maxStorms = 32

    /// How many bytes after the 8-byte record prefix and the common head `expectedLength` needs, or nil for a
    /// tag this decoder does not know.
    static func prefixBytes(_ tag: String) -> Int? {
        switch tag {
        case "FRMA": 20
        case "IDNT", "INTN", "STRM": 16
        case "TERR": 16
        case "MAP1": 32
        case "EVNT": 8
        case "ENVS": 4
        case "CLCK", "FOCS": 0
        default: nil
        }
    }

    private static func pad4(_ n: Int) -> Int { (n + 3) & ~3 }

    private static func malformed(_ tag: String, _ why: String) -> StreamError { .malformedRecord(tag: tag, why) }

    /// The payload length a record of `tag` must have, from the counts in its fixed prefix (the common head and
    /// `prefixBytes(tag)` bytes are available). Throws for counts the header or the format rule out.
    static func expectedLength(_ tag: String, _ r: ByteReader, header: StreamHeader) throws -> Int {
        let head = headBytes
        switch tag {
        case "FRMA":
            let n = Int(r.u32(head)), nBonds = Int(r.u32(head + 4)), cap = Int(r.u32(head + 16))
            guard cap == header.capacity else {
                throw malformed(tag, "N \(cap) differs from the header's \(header.capacity)")
            }
            guard n <= cap else { throw malformed(tag, "n \(n) exceeds N \(cap)") }
            guard nBonds <= n * header.bondSlots else {
                throw malformed(tag, "n_bonds \(nBonds) exceeds the \(header.bondSlots) bond slots of \(n) rows")
            }
            return 36 + 4 * ((cap + 31) / 32) + 28 * n + 8 * nBonds
        case "IDNT":
            let rows = Int(r.u32(head)), jsonLength = Int(r.u32(head + 8))
            try checkRows(tag, rows, header)
            try checkJSON(tag, jsonLength)
            return head + 16 + pad4(jsonLength) + 16 * rows
        case "INTN":
            let rows = Int(r.u32(head))
            try checkRows(tag, rows, header)
            return head + 16 + 12 * rows
        case "TERR":
            let nx = Int(r.u16(head)), ny = Int(r.u16(head + 2)), jsonLength = Int(r.u32(head + 8))
            guard (1...256).contains(nx), (1...256).contains(ny) else { throw malformed(tag, "map \(nx) x \(ny)") }
            try checkJSON(tag, jsonLength)
            return head + 16 + pad4(jsonLength) + pad4(2 * nx * ny)
        case "STRM":
            let jsonLength = Int(r.u32(head)), count = Int(r.u32(head + 8))
            guard count <= maxStorms else { throw malformed(tag, "\(count) storms exceed the cap of \(maxStorms)") }
            try checkJSON(tag, jsonLength)
            return head + 16 + pad4(jsonLength)
        case "MAP1":
            let n = Int(r.u16(head)), nw = Int(r.u16(head + 2))
            guard (1...1024).contains(n), (1...n).contains(nw), n % nw == 0 else {
                throw malformed(tag, "map \(n) cells with \(nw) wind blocks")
            }
            return head + 32 + pad4(4 * n * n + 4 * nw * nw)
        case "EVNT":
            let count = Int(r.u32(head)), jsonLength = Int(r.u32(head + 4))
            guard count <= maxEvents else { throw malformed(tag, "\(count) events exceed the cap of \(maxEvents)") }
            try checkJSON(tag, jsonLength)
            return head + 8 + pad4(jsonLength)
        case "ENVS":
            let jsonLength = Int(r.u32(head))
            try checkJSON(tag, jsonLength)
            return head + 4 + pad4(jsonLength)
        case "CLCK": return head + 44
        case "FOCS": return head + 68
        default: throw malformed(tag, "not a play record")
        }
    }

    private static func checkRows(_ tag: String, _ rows: Int, _ header: StreamHeader) throws {
        guard rows <= header.capacity else { throw malformed(tag, "\(rows) rows exceed N \(header.capacity)") }
    }

    private static func checkJSON(_ tag: String, _ length: Int) throws {
        guard length <= maxJSON else { throw malformed(tag, "json_len \(length) exceeds 1 MiB") }
    }

    /// The record whose payload `r` holds (its length has been checked against `expectedLength`).
    static func decode(_ tag: String, _ r: ByteReader, header: StreamHeader) throws -> StreamRecord {
        let head = RecordHead(step: r.u64(0), seq: r.u32(8), version: r.u16(12), flags: r.u16(14))
        switch tag {
        case "FRMA": return .frma(try frma(head, r, header))
        case "IDNT": return .idnt(try idnt(head, r, header))
        case "INTN": return .intn(try intn(head, r))
        case "TERR": return .terr(try terr(head, r))
        case "STRM": return .strm(try strm(head, r))
        case "MAP1": return .map1(try map1(head, r))
        case "EVNT": return .evnt(try evnt(head, r))
        case "ENVS": return .envs(try envs(head, r))
        case "CLCK": return .clck(try clck(head, r))
        case "FOCS": return .focs(try focs(head, r))
        default: throw malformed(tag, "not a play record")
        }
    }

    // MARK: Records

    private static func frma(_ head: RecordHead, _ r: ByteReader, _ header: StreamHeader) throws -> PlayFrame {
        let tag = "FRMA"
        let n = Int(r.u32(16)), nBonds = Int(r.u32(20)), colorIndex = Int(r.u32(24)), span = Int(r.u32(28))
        let cap = Int(r.u32(32))
        guard colorIndex < header.colorModes.count else {
            throw malformed(tag, "color_index \(colorIndex) is outside the \(header.colorModes.count) colour modes")
        }
        let words = (cap + 31) / 32
        let bitmap: [UInt32] = r.copy(count: words, at: 36)
        var rows: [UInt32] = []
        rows.reserveCapacity(n)
        for (w, word) in bitmap.enumerated() {
            var bits = word
            while bits != 0 {
                rows.append(UInt32(w * 32 + bits.trailingZeroBitCount))
                bits &= bits - 1
            }
        }
        guard rows.count == n else { throw malformed(tag, "n \(n) but the bitmap lists \(rows.count) rows") }
        guard rows.last.map({ Int($0) < cap }) ?? true else { throw malformed(tag, "a bitmap bit at or past N is set") }
        var offset = 36 + 4 * words
        let pos: [Float] = r.copy(count: 3 * n, at: offset)
        offset += 12 * n
        let orient: [UInt16] = r.copy(count: 2 * n, at: offset)
        offset += 4 * n
        let rgb: [UInt32] = r.copy(count: n, at: offset)
        offset += 4 * n
        let uid: [Int64] = r.copy(count: n, at: offset)
        offset += 8 * n
        let bonds: [UInt32] = r.copy(count: 2 * nBonds, at: offset)
        for k in 0..<nBonds where !(bonds[2 * k] < bonds[2 * k + 1] && Int(bonds[2 * k + 1]) < n) {
            throw malformed(tag, "bond \(k) is (\(bonds[2 * k]), \(bonds[2 * k + 1])), not i < j < n")
        }
        return PlayFrame(head: head, n: n, capacity: cap, colorIndex: colorIndex, span: span, bitmap: bitmap,
                         rows: rows, pos: pos, orient: orient, rgb: rgb, uid: uid, bonds: bonds)
    }

    private static func idnt(_ head: RecordHead, _ r: ByteReader, _ header: StreamHeader) throws -> PlayIdentities {
        let tag = "IDNT"
        let n = Int(r.u32(16)), snapshotStep = r.u32(20), jsonLength = Int(r.u32(24))
        guard r.u32(28) == 0 else { throw malformed(tag, "the reserved word is not zero") }
        let tables: IdentityTables = try json(tag, r, at: 32, length: jsonLength)
        guard tables.houses.count <= Slot.maxTable, tables.kin.count <= Slot.maxTable,
            tables.colonies.count <= Slot.maxTable
        else { throw malformed(tag, "a table lists more than \(Slot.maxTable) rows") }
        let slots = (
            houses: Set(tables.houses.map(\.slot)), kin: Set(tables.kin.map(\.slot)),
            colonies: Set(tables.colonies.map(\.slot))
        )
        guard slots.houses.count == tables.houses.count, slots.kin.count == tables.kin.count,
            slots.colonies.count == tables.colonies.count,
            (slots.houses.union(slots.kin).union(slots.colonies)).allSatisfy({ 0 <= $0 && $0 < Slot.maxTable })
        else { throw malformed(tag, "a table entry has a repeated slot or one outside 0..<\(Slot.maxTable)") }
        let base = 32 + pad4(jsonLength)
        var rows: [IdentityRow] = []
        rows.reserveCapacity(n)
        for i in 0..<n {
            let o = base + 16 * i
            let row = IdentityRow(uid: r.i64(o), clade: Int32(bitPattern: r.u32(o + 8)), house: r.u8(o + 12),
                                  kin: r.u8(o + 13), colony: r.u8(o + 14), flags: r.u8(o + 15))
            guard Slot.isValid(row.house, among: slots.houses) else {
                throw malformed(tag, "house slot \(row.house) is not the slot of a listed house")
            }
            guard Slot.isValid(row.kin, among: slots.kin) else {
                throw malformed(tag, "kin slot \(row.kin) is not the slot of a listed kin group")
            }
            guard Slot.isValid(row.colony, among: slots.colonies) else {
                throw malformed(tag, "colony slot \(row.colony) is not the slot of a listed colony")
            }
            rows.append(row)
        }
        return PlayIdentities(head: head, snapshotStep: snapshotStep, tables: tables, rows: rows)
    }

    private static func intn(_ head: RecordHead, _ r: ByteReader) throws -> PlayLabels {
        let n = Int(r.u32(16))
        guard r.u32(28) == 0 else { throw malformed("INTN", "the reserved word is not zero") }
        var rows: [LabelRow] = []
        rows.reserveCapacity(n)
        for i in 0..<n {
            let o = 32 + 12 * i
            rows.append(LabelRow(uid: r.i64(o), archPrimary: r.u8(o + 8), archSecondary: r.u8(o + 9),
                                 intent: r.u8(o + 10), flags: r.u8(o + 11)))
        }
        return PlayLabels(head: head, labelsStep: r.u32(20), intentVersion: r.u32(24), rows: rows)
    }

    private static func terr(_ head: RecordHead, _ r: ByteReader) throws -> PlayTerritory {
        let tag = "TERR"
        let nx = Int(r.u16(16)), ny = Int(r.u16(18)), snapshotStep = r.u32(20), jsonLength = Int(r.u32(24))
        guard r.u32(28) == 0 else { throw malformed(tag, "the reserved word is not zero") }
        struct Ids: Decodable { var ids: [Int] }
        let ids = try json(tag, r, at: 32, length: jsonLength, as: Ids.self).ids
        guard ids.count <= Slot.maxTable else { throw malformed(tag, "\(ids.count) kin ids exceed \(Slot.maxTable)") }
        let base = 32 + pad4(jsonLength)
        let owner: [UInt8] = r.copy(count: nx * ny, at: base)
        let purity: [UInt8] = r.copy(count: nx * ny, at: base + nx * ny)
        for slot in owner where !Slot.isValid(slot, tableSize: ids.count) {
            throw malformed(tag, "owner slot \(slot) is outside a table of \(ids.count)")
        }
        try checkZero(tag, r, from: base + 2 * nx * ny, to: pad4(base + 2 * nx * ny))
        return PlayTerritory(head: head, nx: nx, ny: ny, snapshotStep: snapshotStep, ids: ids, owner: owner,
                             purity: purity)
    }

    private static func strm(_ head: RecordHead, _ r: ByteReader) throws -> PlayStormRecord {
        let tag = "STRM"
        let jsonLength = Int(r.u32(16)), tick = r.u32(20), count = Int(r.u32(24))
        guard r.u32(28) == 0 else { throw malformed(tag, "the reserved word is not zero") }
        let storms: PlayStorms = try json(tag, r, at: 32, length: jsonLength)
        guard storms.storms.count == count else {
            throw malformed(tag, "n_storms \(count) but the JSON lists \(storms.storms.count)")
        }
        return PlayStormRecord(head: head, tick: tick, storms: storms)
    }

    private static func map1(_ head: RecordHead, _ r: ByteReader) throws -> PlayWeatherMap {
        let tag = "MAP1"
        let n = Int(r.u16(16)), nw = Int(r.u16(18))
        guard head.flags & 3 == 3 else { throw malformed(tag, "a version 1 record carries T_s and the winds") }
        let cells = n * n
        var offset = 48
        let p: [UInt8] = r.copy(count: cells, at: offset)
        offset += cells
        let dc: [UInt8] = r.copy(count: cells, at: offset)
        offset += cells
        let zeta: [Int8] = r.copy(count: cells, at: offset)
        offset += cells
        let ts: [UInt8] = r.copy(count: cells, at: offset)
        offset += cells
        let winds: [Int8] = r.copy(count: 4 * nw * nw, at: offset)
        offset += 4 * nw * nw
        try checkZero(tag, r, from: offset, to: pad4(offset))
        guard (40..<48).allSatisfy({ r.u8($0) == 0 }) else { throw malformed(tag, "reserved bytes are not zero") }
        return PlayWeatherMap(head: head, n: n, nw: nw, f0: r.f32(20), pRef: r.f32(24), dcRef: r.f32(28),
                              uRef: r.f32(32), zetaRef: r.f32(36), p: p, dc: dc, zeta: zeta, ts: ts, winds: winds)
    }

    private static func evnt(_ head: RecordHead, _ r: ByteReader) throws -> PlayEventBatch {
        let tag = "EVNT"
        let count = Int(r.u32(16)), jsonLength = Int(r.u32(20))
        let events: [WorldEvent] = try json(tag, r, at: 24, length: jsonLength)
        guard events.count == count else { throw malformed(tag, "count \(count) but the JSON lists \(events.count)") }
        for e in events where e.text.count > 120 {
            throw malformed(tag, "an event text has \(e.text.count) characters, the limit is 120")
        }
        return PlayEventBatch(head: head, events: events)
    }

    private static func envs(_ head: RecordHead, _ r: ByteReader) throws -> PlayEnvironmentRecord {
        let jsonLength = Int(r.u32(16))
        let environment: PlayEnvironment = try json("ENVS", r, at: 20, length: jsonLength)
        return PlayEnvironmentRecord(head: head, environment: environment)
    }

    private static func clck(_ head: RecordHead, _ r: ByteReader) throws -> PlayClock {
        guard let reason = PlayClock.Reason(rawValue: r.u8(40)) else {
            throw malformed("CLCK", "reason \(r.u8(40)) is not 0 to 5")
        }
        guard r.u8(41) <= 1 else { throw malformed("CLCK", "saving is \(r.u8(41)), not 0 or 1") }
        return PlayClock(head: head, rate: r.f32(16), achieved: r.f32(20), stepMs: r.f32(24), frameMs: r.f32(28),
                         lateSteps: r.u32(32), saveStallSteps: r.u32(36), reason: reason, saving: r.u8(41) == 1,
                         flags: r.u16(42), fe: r.u32(44), chapter: r.u32(48), inChapter: r.u32(52),
                         chapterLen: r.u32(56))
    }

    private static func focs(_ head: RecordHead, _ r: ByteReader) throws -> PlayFocus {
        let focus = PlayFocus(
            head: head, uid: r.i64(16), row: r.u32(24), flags: r.u32(28), x: r.f32(32), y: r.f32(36), z: r.f32(40),
            vx: r.f32(44), vy: r.f32(48), vz: r.f32(52), energy: Int32(bitPattern: r.u32(56)), age: r.u32(60),
            generation: r.u32(64), parentUid: r.i64(68), nBonds: r.u8(76), house: r.u8(77), kin: r.u8(78),
            colony: r.u8(79), archPrimary: r.u8(80), archSecondary: r.u8(81), intent: r.u8(82), labelFlags: r.u8(83))
        for (name, slot) in [("house", focus.house), ("kin", focus.kin), ("colony", focus.colony)]
        where !(Int(slot) < Slot.maxTable || Slot.isReserved(slot)) {
            throw malformed("FOCS", "\(name) slot \(slot) is not a slot or 253 to 255")
        }
        return focus
    }

    // MARK: JSON

    private static func json<T: Decodable>(
        _ tag: String, _ r: ByteReader, at offset: Int, length: Int, as type: T.Type = T.self
    ) throws -> T {
        try checkZero(tag, r, from: offset + length, to: pad4(offset + length))
        let data = Data(r.bytes(at: offset, count: length))
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw malformed(tag, "json: \(describe(error))")
        }
    }

    private static func checkZero(_ tag: String, _ r: ByteReader, from: Int, to: Int) throws {
        for i in from..<to where r.u8(i) != 0 { throw malformed(tag, "padding byte \(i) is not zero") }
    }

    private static func describe(_ error: Error) -> String {
        guard let error = error as? DecodingError else { return "\(error)" }
        func key(_ path: [CodingKey]) -> String { path.map(\.stringValue).joined(separator: ".") }
        switch error {
        case .keyNotFound(let k, let c): return "the key \((c.codingPath + [k]).map(\.stringValue).joined(separator: ".")) is missing"
        case .typeMismatch(_, let c), .valueNotFound(_, let c): return "the key \(key(c.codingPath)) has the wrong type"
        case .dataCorrupted: return "not valid JSON"
        @unknown default: return "\(error)"
        }
    }
}

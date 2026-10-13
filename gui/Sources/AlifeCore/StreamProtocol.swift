import Foundation

// The binary stream of `alife stream` (version 1), decoded from DESIGN.md §27.11 "Protocol", which is normative.
// All integers are little endian, as is every host the app runs on; sections are copied without swapping bytes.
#if !_endian(little)
#error("StreamProtocol copies little-endian sections as they are and needs a little-endian host")
#endif

// MARK: - Header

/// The header object of a stream: the JSON after the 12-byte preamble. A key a later compatible version adds is
/// ignored; a missing key is an error (every key is always present).
package struct StreamHeader: Equatable, Sendable, Decodable {
    /// `protocol`: the major version, equal to the last magic byte.
    package var protocolVersion: Int
    /// 2 or 3. A 2D stream carries z = 0 and pitch = 0.
    package var dims: Int
    /// `L`: `world.size`; x and y are periodic on `[0, L)`.
    package var size: Double
    /// `height`: z is bounded on `[0, height]` (0.0 when `dims` is 2).
    package var height: Double
    /// `N`: the rows in every per-agent section of a frame (the state's agent capacity).
    package var capacity: Int
    /// `B`: bond slots per agent.
    package var bondSlots: Int
    /// `radius`: the agent radius.
    package var radius: Double
    package var fieldCell: Double
    /// `FG`: field cells per side, and `FZ`: field layers (1 when `dims` is 2).
    package var fieldGrid: Int
    package var fieldLayers: Int
    /// `NCH`: chemical channels.
    package var channels: Int
    /// The index of a mode is a frame's `colorIndex`.
    package var colorModes: [String]
    /// The view of the first frame: colour mode, field section on or off, field layer.
    package var color: String
    package var fields: Bool
    package var fieldZ: Int
    /// The step of the first frame.
    package var step: Int
    /// The `RUN` or `FILE` argument as given; nil for a preset.
    package var run: String?
    package var configHash: String
    /// `"simulate"` or `"follow"`.
    package var mode: String
    /// Steps per frame; 0 in follow mode.
    package var every: Int
    /// `"run"`, `"file"` or `"preset"`.
    package var source: String

    /// Decodes every key explicitly. The synthesized decoder reads an optional property with `decodeIfPresent`, which
    /// would let the `run` key be absent; the protocol says it is a string or null and always present.
    package init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        protocolVersion = try c.decode(Int.self, forKey: .protocolVersion)
        dims = try c.decode(Int.self, forKey: .dims)
        size = try c.decode(Double.self, forKey: .size)
        height = try c.decode(Double.self, forKey: .height)
        capacity = try c.decode(Int.self, forKey: .capacity)
        bondSlots = try c.decode(Int.self, forKey: .bondSlots)
        radius = try c.decode(Double.self, forKey: .radius)
        fieldCell = try c.decode(Double.self, forKey: .fieldCell)
        fieldGrid = try c.decode(Int.self, forKey: .fieldGrid)
        fieldLayers = try c.decode(Int.self, forKey: .fieldLayers)
        channels = try c.decode(Int.self, forKey: .channels)
        colorModes = try c.decode([String].self, forKey: .colorModes)
        color = try c.decode(String.self, forKey: .color)
        fields = try c.decode(Bool.self, forKey: .fields)
        fieldZ = try c.decode(Int.self, forKey: .fieldZ)
        step = try c.decode(Int.self, forKey: .step)
        run = try c.decode(String?.self, forKey: .run)  // throws keyNotFound when absent, nil for null
        configHash = try c.decode(String.self, forKey: .configHash)
        mode = try c.decode(String.self, forKey: .mode)
        every = try c.decode(Int.self, forKey: .every)
        source = try c.decode(String.self, forKey: .source)
    }

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol"
        case dims
        case size = "L"
        case height
        case capacity = "N"
        case bondSlots = "B"
        case radius
        case fieldCell = "field_cell"
        case fieldGrid = "FG"
        case fieldLayers = "FZ"
        case channels = "NCH"
        case colorModes = "color_modes"
        case color
        case fields
        case fieldZ = "field_z"
        case step
        case run
        case configHash = "config_hash"
        case mode
        case every
        case source
    }

    /// Limits that keep a corrupt header from asking for absurd memory; the engine's own are tighter
    /// (`population.n_max` ≤ 2^20, FG ≤ 1024).
    static let maxCapacity = 1 << 22
    static let maxFieldGrid = 1 << 10

    /// Refuses a header whose values cannot describe a stream (`magicVersion` is the preamble's last byte): a value out of
    /// its range, or one that breaks a relation §27.11 states (`FG = L / field_cell`, `FZ = height / field_cell` and
    /// `height = 0` in 2D, the initial view inside the header's own colour modes and field layers, `mode` and `source`
    /// among the values of version 1). The viewer copies the initial view into its state, so it must be one it can show.
    func validate(magicVersion: Int) throws {
        func bad(_ why: String) -> StreamError { .malformedHeader(why) }
        guard protocolVersion == magicVersion else {
            throw bad("protocol \(protocolVersion) does not match the version \(magicVersion) of the magic")
        }
        guard dims == 2 || dims == 3 else { throw bad("dims \(dims) is not 2 or 3") }
        guard size.isFinite, size > 0, height.isFinite, height >= 0 else { throw bad("L or height is not a size") }
        guard radius.isFinite, radius > 0 else { throw bad("radius \(radius) is not a positive number") }
        guard fieldCell.isFinite, fieldCell > 0 else { throw bad("field_cell \(fieldCell) is not a positive number") }
        guard (1...Self.maxCapacity).contains(capacity) else { throw bad("N \(capacity) is outside [1, 2^22]") }
        guard (1...64).contains(bondSlots) else { throw bad("B \(bondSlots) is outside [1, 64]") }
        guard (1...Self.maxFieldGrid).contains(fieldGrid), (1...Self.maxFieldGrid).contains(fieldLayers) else {
            throw bad("FG \(fieldGrid) or FZ \(fieldLayers) is outside [1, 1024]")
        }
        guard Self.agrees(Double(fieldGrid) * fieldCell, size) else {
            throw bad("FG \(fieldGrid) cells of \(fieldCell) do not make L = \(size)")
        }
        if dims == 2 {
            guard height == 0, fieldLayers == 1 else {
                throw bad("a 2D stream has height 0 and FZ 1, not \(height) and \(fieldLayers)")
            }
        } else {
            guard height > 0, Self.agrees(Double(fieldLayers) * fieldCell, height) else {
                throw bad("FZ \(fieldLayers) layers of \(fieldCell) do not make height \(height)")
            }
        }
        guard !colorModes.isEmpty else { throw bad("color_modes is empty") }
        guard colorModes.contains(color) else { throw bad("color \"\(color)\" is not one of color_modes") }
        guard (0..<fieldLayers).contains(fieldZ) else { throw bad("field_z \(fieldZ) is outside [0, \(fieldLayers))") }
        guard every >= 0 else { throw bad("every \(every) is negative") }
        guard mode == "simulate" || mode == "follow" else { throw bad("mode \"\(mode)\" is not simulate or follow") }
        guard ["run", "file", "preset"].contains(source) else {
            throw bad("source \"\(source)\" is not run, file or preset")
        }
    }

    /// Whether two sizes written by the engine as binary64 agree, allowing for the rounding of a product.
    private static func agrees(_ a: Double, _ b: Double) -> Bool { abs(a - b) <= 1e-6 * max(abs(b), 1) }

    /// The name of the colour mode `frame` carries, or nil for an index outside `colorModes`.
    package func colorMode(of frame: StreamFrame) -> String? {
        colorModes.indices.contains(frame.colorIndex) ? colorModes[frame.colorIndex] : nil
    }
}

// MARK: - Frame

/// One `FRM1` record: the streamed state at `step`. Rows of a per-agent section are agent slots; the values of a row
/// with `alive == 0` are unspecified and are to be ignored.
package struct StreamFrame: Equatable, Sendable {
    package var step: UInt64
    /// The number of rows with `alive == 1`.
    package var nAlive: Int
    /// Bit 0: a field section follows; bit 1: the state's diagnostics have an error bit (an invariant violation);
    /// bits 2 to 15 are reserved; bits 16 to 31 are the field layer. A bit this version does not know is kept.
    package var flags: UInt32
    package var nBonds: Int
    /// The colour mode of `rgb`: an index into the header's `colorModes`.
    package var colorIndex: Int
    /// `N` bytes, 1 for an alive row.
    package var alive: [UInt8]
    /// `3N` floats: x, y, z of each row (z is 0 in a 2D stream).
    package var pos: [Float]
    /// `2N` binary16 values as their bit patterns: yaw then pitch of each row. A renderer can upload them as they
    /// are (a `half2` per row); `yaw(row)` and `pitch(row)` convert.
    package var orient: [UInt16]
    /// `N` colours, `0x00RRGGBB`.
    package var rgb: [UInt32]
    /// `2·nBonds` row indices: the pairs `(i, j)` with `i < j`, both alive, each bond once.
    package var bonds: [UInt32]
    /// `FG²` colours `0x00RRGGBB` of one field layer, row-major `[cy][cx]`; nil without a field section.
    package var field: [UInt32]?

    package init(step: UInt64, nAlive: Int, flags: UInt32, nBonds: Int, colorIndex: Int, alive: [UInt8], pos: [Float],
                 orient: [UInt16], rgb: [UInt32], bonds: [UInt32], field: [UInt32]?) {
        self.step = step
        self.nAlive = nAlive
        self.flags = flags
        self.nBonds = nBonds
        self.colorIndex = colorIndex
        self.alive = alive
        self.pos = pos
        self.orient = orient
        self.rgb = rgb
        self.bonds = bonds
        self.field = field
    }

    package var rows: Int { alive.count }
    package var hasField: Bool { flags & 1 != 0 }
    /// An invariant violation in the streamed state (the pygame HUD shows it in red).
    package var hasDiagError: Bool { flags & 2 != 0 }
    /// The lattice layer of the field section; nil without one.
    package var fieldLayer: Int? { hasField ? Int(flags >> 16) : nil }

    package func isAlive(_ row: Int) -> Bool { alive[row] != 0 }

    package func position(_ row: Int) -> SIMD3<Float> {
        SIMD3(pos[3 * row], pos[3 * row + 1], pos[3 * row + 2])
    }

    /// The heading, in radians in `[-π, π)`.
    package func yaw(_ row: Int) -> Float { Half.float(orient[2 * row]) }

    /// The elevation of the forward axis, in radians (within ±80° in 3D, 0 in 2D).
    package func pitch(_ row: Int) -> Float { Half.float(orient[2 * row + 1]) }

    package func bond(_ index: Int) -> (i: Int, j: Int) { (Int(bonds[2 * index]), Int(bonds[2 * index + 1])) }
}

/// IEEE 754 binary16, which Swift's `Float16` has only on Apple silicon.
package enum Half {
    /// The exact `Float` value of the half-precision bit pattern `bits`.
    package static func float(_ bits: UInt16) -> Float {
        let sign: FloatingPointSign = bits & 0x8000 != 0 ? .minus : .plus
        let exponent = Int((bits >> 10) & 0x1F)
        let fraction = Float(bits & 0x03FF)
        switch exponent {
        case 0:  // zero and subnormals: fraction · 2^-24
            return Float(sign: sign, exponent: -24, significand: fraction)
        case 31:
            return fraction != 0 ? .nan : sign == .minus ? -.infinity : .infinity
        default:
            return Float(sign: sign, exponent: exponent - 15, significand: 1 + fraction / 1024)
        }
    }
}

// MARK: - Events and errors

/// What the decoder reports, in stream order.
package enum StreamEvent: Equatable, Sendable {
    case header(StreamHeader)
    case frame(StreamFrame)
    /// A record of a tag this version does not know, skipped by its length. The tag is its four bytes as Latin-1.
    case unknownRecord(tag: String, length: Int)
}

/// Why a stream was refused. The cases are what DESIGN §27.11 tells a decoder to reject.
package enum StreamError: Error, Equatable, Sendable, CustomStringConvertible {
    /// The first seven bytes are not `ALSTRM` and a zero byte.
    case badMagic
    /// The major version (the last magic byte) is not 1.
    case unsupportedProtocol(Int)
    case malformedHeader(String)
    /// A record's length is not what its content requires.
    case lengthMismatch(tag: String, declared: Int, expected: Int)
    /// A frame whose content contradicts the header or the specification.
    case malformedFrame(String)
    /// The stream ended inside the header or a record: the process died.
    case truncated(String)

    package var description: String {
        switch self {
        case .badMagic: return "not an alife stream (bad magic bytes)"
        case .unsupportedProtocol(let v): return "unsupported stream protocol version \(v) (this app reads version 1)"
        case .malformedHeader(let why): return "malformed stream header: \(why)"
        case .lengthMismatch(let tag, let declared, let expected):
            return "stream record \(tag) declares \(declared) bytes but its content takes \(expected)"
        case .malformedFrame(let why): return "malformed stream frame: \(why)"
        case .truncated(let why): return "truncated stream: \(why)"
        }
    }
}

// MARK: - Decoder

/// An incremental decoder of the `alife stream` protocol. Feed it the bytes of the pipe as they arrive, in any
/// chunking: it owns the partial record, and the events it returns are the same for every way of cutting the
/// stream. It is strict: a bad magic, an unknown major version, a frame whose length disagrees with its content,
/// a frame that contradicts the header, and (at `finish()`) a stream that ends inside a record are errors.
///
/// Records are 4-byte tag, `u32` length, payload. `FRM1` is a frame; a record of any other tag is skipped by its
/// length without being kept. An error is sticky: the decoder reports the events completed before it, and throws
/// from that call if there were none, else from the next `feed` or `finish`, so a caller always sees them in order.
package struct FrameDecoder: Sendable {
    private static let magic: [UInt8] = Array("ALSTRM".utf8) + [0]
    private static let version = 1
    private static let maxHeaderBytes = 1 << 20
    private static let frameTag = "FRM1"
    private static let frameHeadBytes = 32

    private var buffer: [UInt8] = []
    /// The first unconsumed byte of `buffer`.
    private var head = 0
    /// An unknown record being skipped: its tag, declared length and the bytes still to drop.
    private var skipping: (tag: String, length: Int, remaining: Int)?
    /// The error that ended the stream, from the `feed` that met it on. `feed` reports the events before the error and
    /// throws the error from the next call, so a caller that must act on it at once reads it here after each `feed`.
    package private(set) var failure: StreamError?
    /// The header, once it has been decoded.
    package private(set) var header: StreamHeader?

    package init() {}

    private var available: Int { buffer.count - head }

    /// Adds `data` and returns the events it completed.
    package mutating func feed(_ data: Data) throws -> [StreamEvent] {
        if let failure { throw failure }
        data.withUnsafeBytes { buffer.append(contentsOf: $0) }
        var events: [StreamEvent] = []
        do {
            try parse(into: &events)
        } catch let error as StreamError {
            failure = error
            discardInput()
            if events.isEmpty { throw error }
        }
        compact()
        return events
    }

    /// Call at the end of the input. Throws the error that ended the stream, or `truncated` if it ends inside the
    /// header or a record (an input that ends between two records is complete).
    package func finish() throws {
        if let failure { throw failure }
        if header == nil {
            throw StreamError.truncated(buffer.isEmpty ? "the stream ended before its header"
                                                       : "the stream ended inside its header")
        }
        if let skipping {
            throw StreamError.truncated(
                "the stream ended inside a \(skipping.tag) record, "
                    + "\(skipping.remaining) of \(skipping.length) bytes short")
        }
        if available > 0 {
            throw StreamError.truncated("the stream ended inside a record (\(available) bytes of it arrived)")
        }
    }

    // MARK: Parsing

    private mutating func parse(into events: inout [StreamEvent]) throws {
        while true {
            if let skip = skipping {
                let dropped = min(available, skip.remaining)
                head += dropped
                guard dropped == skip.remaining else {
                    skipping?.remaining -= dropped
                    return
                }
                skipping = nil
                events.append(.unknownRecord(tag: skip.tag, length: skip.length))
                continue
            }
            guard let header = self.header else {
                guard let decoded = try parseHeader() else { return }
                self.header = decoded
                events.append(.header(decoded))
                continue
            }
            guard available >= 8 else { return }
            let tag = latin1(at: head, count: 4)
            let length = Int(u32(at: head + 4))
            guard tag == Self.frameTag else {
                head += 8
                skipping = (tag, length, length)
                continue
            }
            // The head of the payload fixes the length it must have, so a corrupt length is refused before the
            // payload is waited for (and before a few gigabytes are buffered).
            guard length >= Self.frameHeadBytes else {
                throw StreamError.lengthMismatch(tag: tag, declared: length, expected: Self.frameHeadBytes)
            }
            guard available >= 8 + Self.frameHeadBytes else { return }
            let expected = try expectedFrameLength(header, payload: head + 8)
            guard length == expected else {
                throw StreamError.lengthMismatch(tag: tag, declared: length, expected: expected)
            }
            guard available >= 8 + length else { return }
            events.append(.frame(try decodeFrame(header, payload: head + 8)))
            head += 8 + length
        }
    }

    /// The preamble and the header object, or nil if more bytes are needed.
    private mutating func parseHeader() throws -> StreamHeader? {
        // Refuse a stream that is not ours as soon as the bytes seen stop matching the magic.
        for i in 0..<min(available, Self.magic.count) where buffer[head + i] != Self.magic[i] {
            throw StreamError.badMagic
        }
        guard available >= 8 else { return nil }
        let version = Int(buffer[head + 7])
        guard version == Self.version else { throw StreamError.unsupportedProtocol(version) }
        guard available >= 12 else { return nil }
        let jsonLength = Int(u32(at: head + 8))
        guard jsonLength % 4 == 0, jsonLength <= Self.maxHeaderBytes else {
            throw StreamError.malformedHeader("json_len \(jsonLength) is not a multiple of 4 up to 1 MiB")
        }
        guard available >= 12 + jsonLength else { return nil }
        let json = Data(buffer[(head + 12)..<(head + 12 + jsonLength)])
        let decoded: StreamHeader
        do {
            decoded = try JSONDecoder().decode(StreamHeader.self, from: json)
        } catch {
            throw StreamError.malformedHeader(Self.describe(error))
        }
        try decoded.validate(magicVersion: version)
        head += 12 + jsonLength
        return decoded
    }

    /// `32 + 4·ceil(N/4) + 20·N + 8·n_bonds`, plus `4·FG²` with a field section; throws if `n_bonds` cannot be.
    private func expectedFrameLength(_ header: StreamHeader, payload: Int) throws -> Int {
        let flags = u32(at: payload + 12)
        let nBonds = Int(u32(at: payload + 16))
        guard nBonds <= header.capacity * header.bondSlots else {
            throw StreamError.malformedFrame(
                "n_bonds \(nBonds) exceeds the \(header.bondSlots) bond slots of \(header.capacity) rows")
        }
        var length = Self.frameHeadBytes + 4 * ((header.capacity + 3) / 4) + 20 * header.capacity + 8 * nBonds
        if flags & 1 != 0 { length += 4 * header.fieldGrid * header.fieldGrid }
        return length
    }

    /// The frame whose payload starts at `payload`; the whole record is buffered and its length has been checked.
    private func decodeFrame(_ header: StreamHeader, payload: Int) throws -> StreamFrame {
        let n = header.capacity
        let step = u64(at: payload)
        let nAlive = Int(u32(at: payload + 8))
        let flags = u32(at: payload + 12)
        let nBonds = Int(u32(at: payload + 16))
        let colorIndex = Int(u32(at: payload + 20))
        // Bytes 24 to 31 are reserved: a later version may use them, so they are not checked.
        guard colorIndex < header.colorModes.count else {
            throw StreamError.malformedFrame(
                "color_index \(colorIndex) is outside the \(header.colorModes.count) colour modes")
        }
        var offset = payload + Self.frameHeadBytes
        let alive: [UInt8] = copy(count: n, at: offset)
        offset += 4 * ((n + 3) / 4)
        var aliveCount = 0
        for flag in alive {
            guard flag <= 1 else { throw StreamError.malformedFrame("an alive byte is \(flag), not 0 or 1") }
            aliveCount += Int(flag)
        }
        guard aliveCount == nAlive else {
            throw StreamError.malformedFrame("n_alive \(nAlive) but \(aliveCount) rows are alive")
        }
        let pos: [Float] = copy(count: 3 * n, at: offset)
        offset += 12 * n
        let orient: [UInt16] = copy(count: 2 * n, at: offset)
        offset += 4 * n
        let rgb: [UInt32] = copy(count: n, at: offset)
        offset += 4 * n
        let bonds: [UInt32] = copy(count: 2 * nBonds, at: offset)
        offset += 8 * nBonds
        for k in 0..<nBonds where !(bonds[2 * k] < bonds[2 * k + 1] && Int(bonds[2 * k + 1]) < n) {
            throw StreamError.malformedFrame("bond \(k) is (\(bonds[2 * k]), \(bonds[2 * k + 1])), not i < j < N")
        }
        var field: [UInt32]?
        if flags & 1 != 0 {
            guard Int(flags >> 16) < header.fieldLayers else {
                throw StreamError.malformedFrame(
                    "field layer \(flags >> 16) is outside the \(header.fieldLayers) layers")
            }
            field = copy(count: header.fieldGrid * header.fieldGrid, at: offset)
        }
        return StreamFrame(step: step, nAlive: nAlive, flags: flags, nBonds: nBonds, colorIndex: colorIndex,
                           alive: alive, pos: pos, orient: orient, rgb: rgb, bonds: bonds, field: field)
    }

    // MARK: Bytes

    private func u32(at offset: Int) -> UInt32 {
        UInt32(buffer[offset]) | UInt32(buffer[offset + 1]) << 8 | UInt32(buffer[offset + 2]) << 16
            | UInt32(buffer[offset + 3]) << 24
    }

    private func u64(at offset: Int) -> UInt64 {
        UInt64(u32(at: offset)) | UInt64(u32(at: offset + 4)) << 32
    }

    private func latin1(at offset: Int, count: Int) -> String {
        String(String.UnicodeScalarView(buffer[offset..<(offset + count)].map { Unicode.Scalar($0) }))
    }

    /// `count` values of `T` read from the bytes at `offset` (a plain copy: the host is little endian).
    private func copy<T: BitwiseCopyable>(count: Int, at offset: Int) -> [T] {
        buffer.withUnsafeBytes { source in
            [T](unsafeUninitializedCapacity: count) { target, initialized in
                let bytes = count * MemoryLayout<T>.stride
                if bytes > 0 {
                    UnsafeMutableRawBufferPointer(target).copyMemory(
                        from: UnsafeRawBufferPointer(rebasing: source[offset..<(offset + bytes)]))
                }
                initialized = count
            }
        }
    }

    /// Drops the consumed bytes (a partial record, at most, is left).
    private mutating func compact() {
        guard head > 0 else { return }
        buffer.removeSubrange(0..<head)
        head = 0
    }

    private mutating func discardInput() {
        buffer.removeAll()
        head = 0
        skipping = nil
    }

    private static func describe(_ error: Error) -> String {
        guard let error = error as? DecodingError else { return "\(error)" }
        func key(_ path: [CodingKey]) -> String { path.map(\.stringValue).joined(separator: ".") }
        switch error {
        case .keyNotFound(let k, _): return "the key \(k.stringValue) is missing"
        case .typeMismatch(_, let c), .valueNotFound(_, let c): return "the key \(key(c.codingPath)) has the wrong type"
        case .dataCorrupted: return "not valid JSON"
        @unknown default: return "\(error)"
        }
    }
}

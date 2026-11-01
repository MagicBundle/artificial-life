import AlifeCore
import AppKit
import Foundation
import Metal
import Testing
import UniformTypeIdentifiers

@testable import AlifeControl

/// Draws the preset guide's snapshots (`gui/PresetSnapshots/<preset>.jpg`) with the app's own Metal renderers, offscreen. It is a tool, not a
/// test of anything: `gui/scripts/make-preset-snapshots.sh render` runs it, and it is skipped unless `ALIFE_PRESET_SNAPSHOTS=1`, so the
/// normal `swift test` never creates a Metal device or writes a file.
///
/// Its input is, per preset, `$ALIFE_PRESET_SNAPSHOT_WORK/<preset>.stream.bin`: what `alife stream RUN --checkpoint latest --steps 0` writes
/// for the run of that plain preset, the header and one frame, which is what the Lab's viewer would receive. The view is the Lab's: the
/// stream's own home camera (the face-on square for a flat world, the whole box from the south for a 3D one, the whole planet for a sphere),
/// with the agents, bonds and frame on and the field slice off (the viewer's defaults; a 3D box is drawn a little closer, `boxZoom`). Output: `$ALIFE_PRESET_SNAPSHOT_WORK/render/<preset>.png`
/// (the 1280 x 800 render), `<preset>.json` (the step and the population of the frame shown) and the JPEG at
/// `$ALIFE_PRESET_SNAPSHOT_OUT` (default `gui/PresetSnapshots`). `$ALIFE_PRESET_SNAPSHOT_ONLY` is a comma-separated list of presets.
@MainActor
@Suite(
    "Preset snapshot tool",
    .serialized,
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_PRESET_SNAPSHOTS"] == "1",
        "set ALIFE_PRESET_SNAPSHOTS=1 to draw the preset snapshots (gui/scripts/make-preset-snapshots.sh render)")
)
struct PresetSnapshotTool {
    static let renderSize = (width: 1280, height: 800)
    static let imageSize = (width: 640, height: 400)
    /// The JPEG must stay under this (the brief: about 120 KB).
    static let byteLimit = 115_000
    /// The fraction of the home distance at which a 3D box is drawn.
    static let boxZoom: Float = 0.72

    private let environment = ProcessInfo.processInfo.environment

    private var work: URL {
        URL(fileURLWithPath: environment["ALIFE_PRESET_SNAPSHOT_WORK"]
            ?? NSString(string: "~/alife-preset-snapshots").expandingTildeInPath)
    }

    private var output: URL {
        environment["ALIFE_PRESET_SNAPSHOT_OUT"].map { URL(fileURLWithPath: $0) }
            ?? Repo.root.appendingPathComponent("gui/PresetSnapshots", isDirectory: true)
    }

    @Test func drawEverySnapshot() throws {
        let only = environment["ALIFE_PRESET_SNAPSHOT_ONLY"].map { Set($0.split(separator: ",").map(String.init)) }
        var drawn = 0
        for preset in Preset.allCases where only?.contains(preset.rawValue) ?? true {
            let input = work.appendingPathComponent("\(preset.rawValue).stream.bin")
            guard FileManager.default.fileExists(atPath: input.path) else {
                Issue.record("no stream for \(preset.rawValue): \(input.path)")
                continue
            }
            try draw(preset, stream: input)
            drawn += 1
        }
        #expect(drawn > 0, "nothing was drawn")
    }

    private func draw(_ preset: Preset, stream: URL) throws {
        var decoder = FrameDecoder()
        let events = try decoder.feed(try Data(contentsOf: stream))
        try decoder.finish()
        let header = try #require(events.compactMap { if case .header(let h) = $0 { h } else { nil } }.first)
        let frame = try #require(events.compactMap { if case .frame(let f) = $0 { f } else { nil } }.last,
                                 "\(preset.rawValue): the stream has no frame")
        let device = try #require(MTLCreateSystemDefaultDevice(), "no Metal device")
        let (w, h) = Self.renderSize
        let aspect = Float(w) / Float(h)
        let image: OffscreenImage
        var view = "the whole world"
        if header.isSphere {
            image = try sphere(device: device, header: header, frame: frame, aspect: aspect)
        } else {
            let renderer = try Renderer(device: device)
            renderer.configure(header: header, aspect: aspect)
            renderer.show(frame: frame)
            renderer.display = ViewerDisplay()
            // The home pose of a 3D box fits the whole box in a margin of dark; the snapshot is a small picture, so the eye comes in a
            // little (the same pose, 0.72 of the distance: the box still fits the 8:5 frame whole).
            if header.dims == 3 { renderer.camera.distance *= Self.boxZoom }
            // A flat world is drawn whole while it is small; a large one is a dust of sub-pixel organisms, so the eye comes down to a
            // window of one scale for every flat preset (`Self.flatWindow`), at the place of typical density (`typicalWindow`).
            if header.dims == 2, let window = Self.typicalWindow(frame: frame, size: Float(header.size), aspect: aspect) {
                var camera = renderer.camera
                camera.distance = Camera.topDown(size: window.height, aspect: aspect).distance
                camera.target = SIMD3(window.centre.x, window.centre.y, 0)
                renderer.camera = camera
                view = "a \(Int(window.width.rounded())) x \(Int(window.height.rounded())) window of typical density"
            }
            image = try renderer.renderOffscreen(width: w, height: h)
        }
        let renders = work.appendingPathComponent("render", isDirectory: true)
        try PNG.write(image, to: renders.appendingPathComponent("\(preset.rawValue).png"))
        let small = try Self.downscaled(image, to: Self.imageSize)
        let jpeg = try Self.jpeg(small, limit: Self.byteLimit)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try jpeg.write(to: output.appendingPathComponent("\(preset.rawValue).jpg"))
        let info: [String: Any] = ["preset": preset.rawValue, "step": Int(frame.step), "alive": frame.nAlive, "bytes": jpeg.count, "view": view]
        try JSONSerialization.data(withJSONObject: info, options: [.sortedKeys])
            .write(to: renders.appendingPathComponent("\(preset.rawValue).json"))
        print("PresetSnapshotTool: \(preset.rawValue) step \(frame.step) alive \(frame.nAlive) -> \(jpeg.count) bytes")
    }

    /// The Lab's planet view (`SphereLabModel`): the home pose at longitude 0, the floor without a field layer, no weather.
    private func sphere(device: any MTLDevice, header: StreamHeader, frame: StreamFrame, aspect: Float) throws -> OffscreenImage {
        let sphere = try #require(header.sphere)
        let renderer = try SphereRenderer(device: device)
        let pose = SphereCamera.home(planetRadius: Float(sphere.planetRadius), lidRadius: Float(sphere.lidRadius), centre: SIMD3(1, 0, 0),
                                     aspect: aspect)
        let camera = SphereCamera(pose: pose, planetRadius: Float(sphere.planetRadius), lidRadius: Float(sphere.lidRadius))
        renderer.configure(sphere: sphere, bodyRadius: Float(header.radius), camera: camera)
        var shown = WorldDisplay()
        let viewer = ViewerDisplay()
        shown.agents = viewer.agents
        shown.bonds = viewer.bonds
        shown.frame = viewer.frame
        shown.territory = .off
        let centres = SphereCentres.shared.centres(nside: sphere.nsideField)
        renderer.setFloor(SphereColours.floor(sphere: sphere, map: nil, material: nil, soil: nil, display: shown, layer: nil, centres: centres))
        renderer.setCloud([SIMD4<Float>](repeating: .zero, count: sphere.fieldCells))
        renderer.setTerritory([SIMD4<Float>](repeating: .zero, count: sphere.territoryCells))
        renderer.show(streamFrame: frame)
        var draw = SphereDraw(camera: camera)
        draw.display = shown
        draw.pixelsPerPoint = 2
        renderer.setDraw(draw)
        return try renderer.renderOffscreen(width: Self.renderSize.width, height: Self.renderSize.height)
    }

    // MARK: The flat window

    /// The height of the window of a flat world, in world units; its width is this times the aspect.
    static let flatWindow: Float = 320

    /// The window (`flatWindow` high, as wide as the 8:5 frame needs) that holds about the average number of organisms of all the windows
    /// that fit inside the world (grid of 32 units), or nil when the world is no larger than the window and is drawn whole. Typical, not
    /// the busiest: a snapshot should not flatter.
    static func typicalWindow(frame: StreamFrame, size: Float, aspect: Float) -> (centre: SIMD2<Float>, width: Float, height: Float)? {
        let height = flatWindow, width = flatWindow * aspect
        guard size > height + 1 else { return nil }
        let xs = Array(stride(from: width / 2, through: max(size - width / 2, width / 2), by: 32))
        let ys = Array(stride(from: height / 2, through: size - height / 2, by: 32))
        var alive: [SIMD2<Float>] = []
        for row in 0..<frame.rows where frame.isAlive(row) { alive.append(SIMD2(frame.pos[3 * row], frame.pos[3 * row + 1])) }
        var candidates: [(SIMD2<Float>, Int)] = []
        for cx in xs {
            for cy in ys {
                let n = alive.reduce(0) { $0 + (abs($1.x - cx) <= width / 2 && abs($1.y - cy) <= height / 2 ? 1 : 0) }
                candidates.append((SIMD2(cx, cy), n))
            }
        }
        let mean = Double(candidates.reduce(0) { $0 + $1.1 }) / Double(max(candidates.count, 1))
        guard let best = candidates.min(by: { abs(Double($0.1) - mean) < abs(Double($1.1) - mean) }) else { return nil }
        return (best.0, min(width, size), height)
    }

    // MARK: Images

    static func cgImage(_ image: OffscreenImage) throws -> CGImage {
        guard let provider = CGDataProvider(data: Data(image.bgra) as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB),
              let picture = CGImage(
                  width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                  bytesPerRow: 4 * image.width, space: space,
                  bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue),
                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { throw PNG.Failure.encode("cannot create the image") }
        return picture
    }

    /// `image` drawn into a `size` bitmap with high-quality interpolation (a 2x supersampled render, averaged down).
    static func downscaled(_ image: OffscreenImage, to size: (width: Int, height: Int)) throws -> CGImage {
        let source = try cgImage(image)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: size.width, height: size.height, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw PNG.Failure.encode("cannot create the context") }
        context.interpolationQuality = .high
        context.draw(source, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        guard let result = context.makeImage() else { throw PNG.Failure.encode("cannot make the image") }
        return result
    }

    /// The JPEG of `image`, at the highest quality of a ladder that fits `limit` bytes.
    static func jpeg(_ image: CGImage, limit: Int) throws -> Data {
        var last = Data()
        for quality in stride(from: 0.92, through: 0.5, by: -0.04) {
            let data = NSMutableData()
            guard let destination = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else {
                throw PNG.Failure.encode("cannot create the JPEG")
            }
            CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
            guard CGImageDestinationFinalize(destination) else { throw PNG.Failure.encode("cannot write the JPEG") }
            last = data as Data
            if last.count <= limit { return last }
        }
        return last
    }
}

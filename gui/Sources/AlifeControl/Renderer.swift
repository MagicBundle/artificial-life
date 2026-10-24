import AlifeCore
import AppKit
import ImageIO
import Metal
import MetalKit
import UniformTypeIdentifiers
import simd

// MARK: - Uniforms

/// The per-frame constants of every shader (`Scene` in Shaders.swift). Only 16-byte types, so the Swift layout is the
/// Metal one: 160 bytes.
struct SceneUniforms: Equatable {
    var viewProjection: simd_float4x4
    /// xyz: the eye.
    var eye: SIMD4<Float>
    /// xyz: the screen's +x and +y in the world.
    var right: SIMD4<Float>
    var up: SIMD4<Float>
    /// x: L, y: height, z: the agent radius.
    var world: SIMD4<Float>
    /// x: rows, y: bonds, z: dims.
    var counts: SIMD4<UInt32>
    /// x, y: the drawable size in pixels.
    var viewport: SIMD4<Float>
}

/// A plane's constants (`Plane` in Shaders.swift): 32 bytes.
struct PlaneUniforms: Equatable {
    var color: SIMD4<Float>
    /// x: the z of the plane.
    var info: SIMD4<Float>
}

/// The World window's constants (`WorldU` in Shaders.swift), buffer 7 of the world pass: 128 bytes.
struct WorldUniforms: Equatable {
    /// xy: the camera target (everything is drawn at its minimum image), z: L, w: 1 / L.
    var target: SIMD4<Float>
    /// x: map blend (previous to current), y: map opacity, z: territory tint alpha, w: the flat-disc threshold in pixels.
    var params: SIMD4<Float>
    /// x: p_ref, y: dc_ref, z: zeta_ref, w: cot(fovY / 2).
    var refs: SIMD4<Float>
    /// x: pass (0 spheres, 1 flat discs), y: rows listed, z: bonds, w: bit 0 map, bit 1 territory, bit 2 equators.
    var flags: SIMD4<UInt32>
    /// x: point alpha, y: point size in pixels, z: the smallest drawn radius in pixels, w: the outline's width in pixels.
    var style: SIMD4<Float>
    /// x: the cap on the drawn radius in pixels (`unbounded` at the physical size), y, z: the alpha ramp's low and high radius in
    /// pixels, w: the depth cue's slope.
    var size: SIMD4<Float>
    /// x: exposure, y: the territory borders' alpha, z: the depth cue's base, w: the hemisphere tint's weight.
    var look: SIMD4<Float>
    /// The one-world patch of the map and its copies across the wrap in y (`SeamCopies`): x: the index `k` of the cell the view's
    /// centre is in (the cell `y` in `[kL - L/2, kL + L/2]`), y, z, w: the alpha of the copies `k - 1`, `k`, `k + 1` (0: not drawn).
    var patch: SIMD4<Float>

    /// What the shader gets for "no cap" (a float that is larger than any radius).
    static let unbounded: Float = 1e9

    static let mapBit: UInt32 = 1
    static let territoryBit: UInt32 = 2
    static let equatorBit: UInt32 = 4
}

/// A translucent quad of the scene: the floor, the surface, or the field slice.
struct ScenePlane: Equatable {
    var z: Float
    var color: SIMD4<Float>
    var textured: Bool

    /// The planes ordered so that the farthest from the eye is drawn first (translucent planes blend correctly only
    /// back to front; they are all horizontal, so the distance is that of the heights).
    static func backToFront(_ planes: [ScenePlane], eyeZ: Float) -> [ScenePlane] {
        planes.sorted { abs(eyeZ - $0.z) > abs(eyeZ - $1.z) }
    }
}

/// A rendered image: BGRA, 8 bits per channel, row 0 at the top.
struct OffscreenImage: Sendable, Equatable {
    var width: Int
    var height: Int
    var bgra: [UInt8]

    func pixel(_ x: Int, _ y: Int) -> (r: UInt8, g: UInt8, b: UInt8) {
        let at = 4 * (y * width + x)
        return (bgra[at + 2], bgra[at + 1], bgra[at])
    }
}

// MARK: - Draw gate

/// Which buffer set the next draw uses, and who may draw: the ring of `slotCount` sets, a semaphore that lets that many
/// draws be in flight, and a second one that lets `windowDraws` of them go into the window (see `Renderer`).
///
/// The ring moves on only when a draw has been committed (`commit`). A draw that fails before that has used no set, so
/// it gives its ticket back (`release`) and the next one uses the same set: moving the ring on before the commit would
/// let the ring and the semaphore drift apart, until a draw wrote a set the GPU was still reading.
@MainActor
final class DrawGate {
    /// The right to one draw with the buffer set `slot`. `giveBack` is what the GPU's completion handler calls.
    struct Ticket: Sendable {
        let slot: Int
        let giveBack: GiveBack
    }

    /// Returns a ticket's permissions; safe to call from any thread, once.
    struct GiveBack: Sendable {
        let sets: DispatchSemaphore
        let window: DispatchSemaphore?

        func callAsFunction() {
            sets.signal()
            window?.signal()
        }
    }

    let slotCount: Int
    private let sets: DispatchSemaphore
    private let window: DispatchSemaphore
    private(set) var next = 0

    init(slotCount: Int, windowDraws: Int) {
        self.slotCount = slotCount
        sets = DispatchSemaphore(value: slotCount)
        window = DispatchSemaphore(value: windowDraws)
    }

    /// A ticket if a buffer set (and for a draw into the window, a place for its drawable) is free within `patience`; nil
    /// if not, and then nothing was taken. A draw into the window never waits for the place.
    func acquire(intoWindow: Bool, patience: DispatchTimeInterval) -> Ticket? {
        if intoWindow, window.wait(timeout: .now()) != .success { return nil }
        guard sets.wait(timeout: .now() + patience) == .success else {
            if intoWindow { window.signal() }
            return nil
        }
        return Ticket(slot: next, giveBack: GiveBack(sets: sets, window: intoWindow ? window : nil))
    }

    /// The draw of `ticket` is committed: the GPU has it, and it gives the ticket back when it has finished.
    func commit(_ ticket: Ticket) {
        next = (ticket.slot + 1) % slotCount
    }

    /// The draw of `ticket` failed before it was committed.
    func release(_ ticket: Ticket) {
        ticket.giveBack()
    }
}

// MARK: - Draw timing

/// What one draw cost (DESIGN §33.5.3): the CPU time to refresh the buffer set and encode the pass, and the GPU's time for
/// the pass (`gpuEndTime - gpuStartTime` of its command buffer). Drawable acquisition and the compositor are not in either.
struct DrawTiming: Sendable, Equatable {
    var cpuMs: Double
    var gpuMs: Double
}

/// A texture pair to draw into off the screen, kept between draws (`Renderer.makeOffscreenTarget`).
final class OffscreenTarget {
    let color: any MTLTexture
    let depth: any MTLTexture
    let width: Int
    let height: Int

    init(color: any MTLTexture, depth: any MTLTexture, width: Int, height: Int) {
        self.color = color
        self.depth = depth
        self.width = width
        self.height = height
    }
}

// MARK: - Renderer

/// Draws the stream's frames with Metal: instanced sphere impostors for the agents (with a nose for the heading), bond
/// half segments, the edges of the box, translucent floor, surface and field-slice planes. It is the `MTKViewDelegate`
/// of the viewer's view, and it can also draw into a texture (`renderOffscreen`, used by the snapshot and the GPU test).
///
/// Drawing is on demand: the view is not on a display link, so the GPU is idle until a frame arrives, the camera moves
/// or the window changes. The frame's arrays are copied into one of three shared buffer sets and a semaphore lets at
/// most three draws be in flight, so the CPU never writes a set the GPU is still reading. A set remembers which frame it
/// holds, so redrawing the same frame from another angle copies nothing once all three have it.
///
/// A draw into the window takes its buffer set first and its drawable second. `CAMetalLayer` has three drawables, one of
/// which is on screen, so a second semaphore lets two draws into the window be in flight; the third would wait inside
/// `nextDrawable()` (for up to a second, on the main thread) for the GPU to finish one of them. A draw into the window
/// never waits for either semaphore: one that finds no free set or drawable is tried again shortly. (A snapshot, which
/// the user asked for, waits for a set for up to 10 s.)
@MainActor
final class Renderer: NSObject, MTKViewDelegate {
    enum Failure: Error, CustomStringConvertible {
        case noDevice
        case shaders(String)
        case missingFunction(String)
        case pipeline(String, String)
        case noQueue
        case allocation(String)
        case encoding
        case busy

        var description: String {
            switch self {
            case .noDevice: return "this Mac has no Metal device"
            case .shaders(let log): return "the viewer's shaders do not compile: \(log)"
            case .missingFunction(let name): return "the shader function \(name) is missing"
            case .pipeline(let name, let why): return "cannot build the \(name) pipeline: \(why)"
            case .noQueue: return "cannot create a Metal command queue"
            case .allocation(let what): return "cannot allocate \(what)"
            case .encoding: return "cannot encode the drawing commands"
            case .busy: return "the GPU is too busy to draw"
            }
        }
    }

    static let slotCount = 3
    /// Draws into the window in flight at once: one fewer than the layer's three drawables (its
    /// `maximumDrawableCount` defaults to 3 and `MTKView` leaves it), because the one on screen is not available.
    static let drawablesInFlight = 2
    static let colorFormat = MTLPixelFormat.bgra8Unorm
    static let depthFormat = MTLPixelFormat.depth32Float
    static let background = MTLClearColor(red: 0.035, green: 0.04, blue: 0.055, alpha: 1)

    let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let spherePipeline: any MTLRenderPipelineState
    private let boxPipeline: any MTLRenderPipelineState
    private let bondPipeline: any MTLRenderPipelineState
    private let planePipeline: any MTLRenderPipelineState
    private let slicePipeline: any MTLRenderPipelineState
    // The World window's pipelines (DESIGN 33.4.6).
    private let agentPipeline: any MTLRenderPipelineState
    private let flatPipeline: any MTLRenderPipelineState
    private let worldBondPipeline: any MTLRenderPipelineState
    private let pointPipeline: any MTLRenderPipelineState
    private let mapPipeline: any MTLRenderPipelineState
    /// One-texel stand-ins for the planes and the territory of a world without them.
    private let blankPlane: any MTLTexture
    private let blankTerritory: any MTLTexture
    /// Depth tested and written (agents, bonds, box) and depth tested only (translucent planes, the 2D background).
    private let opaqueDepth: any MTLDepthStencilState
    private let overlayDepth: any MTLDepthStencilState

    private var slots: [Slot]
    private let gate = DrawGate(slotCount: Renderer.slotCount, windowDraws: Renderer.drawablesInFlight)

    /// The view; edited by the controls, then the owner asks the view to redraw. While it is still the home pose, a
    /// resize of the window fits the world to the new shape again (`cameraIsHome`).
    var camera = Camera.overview(size: 1, height: 1) {
        didSet { if !settingHome { cameraIsHome = false } }
    }
    private(set) var cameraIsHome = true
    private var settingHome = false
    var display = ViewerDisplay()
    private(set) var header: StreamHeader?
    private var frame: StreamFrame?
    /// Increases with every frame shown; a slot that holds an older one is refreshed before it is drawn.
    private var frameSerial = 0
    private(set) var frameStep: UInt64?
    /// Told, from the GPU's completion thread, what each committed draw cost; nil costs nothing (the bench sets it).
    var timingHandler: (@Sendable (DrawTiming) -> Void)?
    /// Told, from the display's thread, when a draw into the window was presented (`MTLDrawable.presentedTime`, seconds):
    /// the intervals between two are what the user sees.
    var presentedHandler: (@Sendable (CFTimeInterval) -> Void)?
    /// Told when a draw fails (once per distinct message), so the window can say so instead of staying blank.
    var failureHandler: (@MainActor (String) -> Void)?
    private var lastFailure: String?

    // MARK: World state

    /// What the renderer draws: the Lab's `StreamFrame`s, or the World window's records.
    enum Mode: Equatable { case lab, world }
    private(set) var mode = Mode.lab
    /// What the world pass draws besides the camera: the level, the layers, the fade between two maps.
    struct WorldDraw: Equatable {
        var level = WorldLevel.planet
        var display = WorldDisplay()
        /// The weight of the current map against the previous one (`MapFade`).
        var mapBlend: Float = 1
        /// The size policy's coverage radius for the rows in view, in drawable pixels (`OrganismSizePolicy.coverRadius`, counted
        /// by the model and smoothed); infinite where it does not apply.
        var coverRadius: Float = .infinity
        /// The drawable's pixels per point of the free rectangle (4.5 pt is the absolute cap of a symbol): 2 on a Retina display.
        var pixelsPerPoint: Float = 2
        /// The copies of the map's patch to draw (`SeamCopies.plan`): the cell of the view's centre, and its neighbours across the wrap
        /// in y when the picture reaches them. Nil draws all three, which is right for any view and a little more than a flat one needs.
        var copies: [SeamCopies.Copy]?
    }
    var worldDraw = WorldDraw()
    /// Told when the drawable's size changes in the World (the free rectangle changed), in pixels.
    var onResize: (@MainActor (CGSize) -> Void)?
    private var playFrame: PlayFrame?
    private var playFrameSerial = 0
    private var mapNow: PlayWeatherMap?
    private var mapPrevious: PlayWeatherMap?
    private var mapNowSerial = -1
    private var mapPreviousSerial = -1
    private var playTerritory: PlayTerritory?
    private var playTerritorySerial = -1

    /// A set of the GPU buffers one draw reads.
    private final class Slot {
        var alive: (any MTLBuffer)?
        var pos: (any MTLBuffer)?
        var orient: (any MTLBuffer)?
        var rgb: (any MTLBuffer)?
        var bonds: (any MTLBuffer)?
        var field: (any MTLTexture)?
        /// The `frameSerial` this slot holds (-1: nothing).
        var serial = -1
        /// The World window's buffers and textures. They live here, one set per draw in flight, with the agents': a texture
        /// written while two or three committed draws still sample it would be a data race (a torn map).
        let world = WorldSlot()
    }

    /// The five textures of one delivery of the weather map.
    struct MapTextures {
        var rain: (any MTLTexture)?
        var cloud: (any MTLTexture)?
        var zeta: (any MTLTexture)?
        var sea: (any MTLTexture)?
        var winds: (any MTLTexture)?
    }

    /// What a buffer set holds for the world pass: the alive-only arrays of the frame, the current and previous weather map
    /// (the serials of the deliveries they hold), and the territory.
    private final class WorldSlot {
        var pos: (any MTLBuffer)?
        var orient: (any MTLBuffer)?
        var rgb: (any MTLBuffer)?
        var bonds: (any MTLBuffer)?
        var frameSerial = -1
        var current = MapTextures()
        var previous = MapTextures()
        var currentSerial = -1
        var previousSerial = -1
        var territory: (any MTLTexture)?
        var territorySerial = -1
    }

    init(device: any MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw Failure.noQueue }
        self.queue = queue

        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        let library: any MTLLibrary
        do {
            library = try device.makeLibrary(source: Shaders.source, options: options)
        } catch {
            throw Failure.shaders("\(error)")
        }
        func function(_ name: String) throws -> any MTLFunction {
            guard let f = library.makeFunction(name: name) else { throw Failure.missingFunction(name) }
            return f
        }
        func pipeline(_ name: String, vertex: String, fragment: String, blend: Bool) throws
            -> any MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.label = name
            d.vertexFunction = try function(vertex)
            d.fragmentFunction = try function(fragment)
            d.colorAttachments[0].pixelFormat = Renderer.colorFormat
            d.depthAttachmentPixelFormat = Renderer.depthFormat
            if blend {
                let c = d.colorAttachments[0]!
                c.isBlendingEnabled = true
                c.rgbBlendOperation = .add
                c.alphaBlendOperation = .add
                c.sourceRGBBlendFactor = .sourceAlpha
                c.destinationRGBBlendFactor = .oneMinusSourceAlpha
                c.sourceAlphaBlendFactor = .one
                c.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            }
            do {
                return try device.makeRenderPipelineState(descriptor: d)
            } catch {
                throw Failure.pipeline(name, "\(error)")
            }
        }
        spherePipeline = try pipeline("agents", vertex: Shaders.sphereVertex, fragment: Shaders.sphereFragment,
                                      blend: false)
        bondPipeline = try pipeline("bonds", vertex: Shaders.bondVertex, fragment: Shaders.lineFragment, blend: false)
        boxPipeline = try pipeline("box", vertex: Shaders.boxVertex, fragment: Shaders.lineFragment, blend: false)
        planePipeline = try pipeline("planes", vertex: Shaders.planeVertex, fragment: Shaders.planeFragment,
                                     blend: true)
        slicePipeline = try pipeline("field slice", vertex: Shaders.planeVertex, fragment: Shaders.sliceFragment,
                                     blend: true)
        // The impostors blend: their opacity is a ramp of their drawn radius (the colour is unchanged where it is 1).
        agentPipeline = try pipeline("world agents", vertex: Shaders.agentVertexA, fragment: Shaders.agentFragmentA,
                                     blend: true)
        flatPipeline = try pipeline("world flat agents", vertex: Shaders.agentVertexA,
                                    fragment: Shaders.agentFragmentFlat, blend: true)
        worldBondPipeline = try pipeline("world bonds", vertex: Shaders.bondVertexA, fragment: Shaders.lineFragment,
                                         blend: false)
        pointPipeline = try pipeline("world points", vertex: Shaders.pointVertexA, fragment: Shaders.pointFragment,
                                     blend: true)
        mapPipeline = try pipeline("world map", vertex: Shaders.mapVertex, fragment: Shaders.mapFragment, blend: false)
        func blank(_ format: MTLPixelFormat) throws -> any MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: 1, height: 1, mipmapped: false)
            d.usage = .shaderRead
            d.storageMode = .shared
            guard let texture = device.makeTexture(descriptor: d) else { throw Failure.allocation("a blank texture") }
            var zero: [UInt8] = [0, 0, 0, 0]
            texture.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &zero, bytesPerRow: 4)
            return texture
        }
        blankPlane = try blank(.r8Unorm)
        blankTerritory = try blank(.rgba8Unorm)

        func depthState(write: Bool) throws -> any MTLDepthStencilState {
            let d = MTLDepthStencilDescriptor()
            d.depthCompareFunction = .lessEqual
            d.isDepthWriteEnabled = write
            guard let state = device.makeDepthStencilState(descriptor: d) else { throw Failure.allocation("a depth state") }
            return state
        }
        opaqueDepth = try depthState(write: true)
        overlayDepth = try depthState(write: false)
        slots = (0..<Renderer.slotCount).map { _ in Slot() }
        super.init()
    }

    // MARK: Content

    /// A new stream: forget the old frame and start from the home pose of the world (`aspect` is the view's).
    func configure(header: StreamHeader?, aspect: Float) {
        mode = .lab
        self.header = header
        frame = nil
        frameStep = nil
        frameSerial += 1
        goHome(aspect: aspect)
    }

    /// The frame to draw from now on (nil: none).
    func show(frame: StreamFrame?) {
        self.frame = frame
        frameStep = frame?.step
        frameSerial += 1
    }

    func resetCamera(aspect: Float) { goHome(aspect: aspect) }

    // MARK: World content

    /// A new world: forget the old frame, map and territory. The camera is the owner's (`WorldModel`).
    func configureWorld(header: StreamHeader?) {
        mode = .world
        self.header = header
        frame = nil
        playFrame = nil
        playFrameSerial += 1
        mapNow = nil
        mapPrevious = nil
        mapNowSerial = -1
        mapPreviousSerial = -1
        playTerritory = nil
        playTerritorySerial = -1
        frameStep = nil
    }

    /// The `FRMA` to draw from now on (nil: none). `serial` increases with every frame.
    func showWorld(frame newFrame: PlayFrame?) {
        playFrame = newFrame
        playFrameSerial += 1
        frameStep = newFrame?.step
    }

    /// A weather map was delivered. The previous one is kept for the fade: `start` says which image the fade starts from
    /// when a delivery lands during one (`MapFade.deliver`). The first map has no previous one.
    func deliverWorld(map: PlayWeatherMap, serial: Int, start: MapFade.Start) {
        if mapNow == nil {
            mapPrevious = map
            mapPreviousSerial = serial
        } else if start == .current {
            mapPrevious = mapNow
            mapPreviousSerial = mapNowSerial
        }
        mapNow = map
        mapNowSerial = serial
    }

    func showWorld(territory: PlayTerritory?, serial: Int) {
        playTerritory = territory
        playTerritorySerial = serial
    }

    /// The textures of the map held by each buffer set, by identity (a test pins that no two sets share one).
    var mapTextureIdentities: [[ObjectIdentifier]] {
        slots.map { slot in
            [slot.world.current, slot.world.previous].flatMap { set in
                [set.rain, set.cloud, set.zeta, set.sea, set.winds].compactMap { $0.map { ObjectIdentifier($0 as AnyObject) } }
            }
        }
    }

    /// The map serials each buffer set holds, (current, previous).
    var mapSerialsBySlot: [(current: Int, previous: Int)] {
        slots.map { ($0.world.currentSerial, $0.world.previousSerial) }
    }

    private func goHome(aspect: Float) {
        guard let header else { return }
        settingHome = true
        camera = Camera.home(for: header, aspect: aspect)
        settingHome = false
        cameraIsHome = true
    }

    // MARK: MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        if mode == .world {
            onResize?(size)
            view.needsDisplay = true
            return
        }
        if cameraIsHome, size.height >= 1 { goHome(aspect: Float(size.width / size.height)) }
        view.needsDisplay = true
    }

    func draw(in view: MTKView) {
        let size = view.drawableSize
        guard size.width >= 1, size.height >= 1 else { return }  // nothing to draw into
        // The ticket comes before the drawable: asking the layer for a drawable blocks (up to a second) when the GPU is
        // behind, and the main thread never waits for the GPU. No ticket: draw again shortly.
        guard let ticket = gate.acquire(intoWindow: true, patience: .milliseconds(0)) else {
            retryLater(view)
            return
        }
        guard let descriptor = view.currentRenderPassDescriptor, let drawable = view.currentDrawable else {
            gate.release(ticket)
            retryLater(view, limit: Self.retryLimit)  // no drawable to draw into; a few more tries, then wait for a cue
            return
        }
        retries = 0
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].clearColor = Self.background
        descriptor.depthAttachment.loadAction = .clear
        descriptor.depthAttachment.clearDepth = 1
        do {
            try submit(ticket, descriptor: descriptor, size: size, drawable: drawable, waitUntilCompleted: false)
        } catch {
            let message = "\(error)"
            if message != lastFailure {
                lastFailure = message
                failureHandler?(message)
            }
        }
    }

    private var retryPending = false
    /// Consecutive redraws that found no drawable.
    private var retries = 0
    private static let retryLimit = 20

    /// Draws again in 100 ms (`limit` times in a row at most, when the cause is not the GPU being busy: a layer that
    /// has no drawable to give does not get one by being asked every 100 ms for ever).
    private func retryLater(_ view: MTKView, limit: Int? = nil) {
        guard !retryPending else { return }
        if let limit {
            retries += 1
            guard retries <= limit else { return }
        }
        retryPending = true
        Task { [weak self, weak view] in
            try? await Task.sleep(for: .milliseconds(100))
            self?.retryPending = false
            view?.needsDisplay = true
        }
    }

    // MARK: Offscreen

    /// A texture pair of `width` x `height` to draw into with `drawNow` or `renderOffscreen`.
    func makeOffscreenTarget(width: Int, height: Int) throws -> OffscreenTarget {
        guard width > 0, height > 0 else { throw Failure.allocation("an empty image") }
        let color = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: Self.colorFormat, width: width, height: height, mipmapped: false)
        color.usage = [.renderTarget]
        color.storageMode = .shared
        let depth = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: Self.depthFormat, width: width, height: height, mipmapped: false)
        depth.usage = [.renderTarget]
        depth.storageMode = .private
        guard let colorTexture = device.makeTexture(descriptor: color),
              let depthTexture = device.makeTexture(descriptor: depth) else {
            throw Failure.allocation("the offscreen textures")
        }
        return OffscreenTarget(color: colorTexture, depth: depthTexture, width: width, height: height)
    }

    private func descriptor(for target: OffscreenTarget) -> MTLRenderPassDescriptor {
        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = target.color
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store
        descriptor.colorAttachments[0].clearColor = Self.background
        descriptor.depthAttachment.texture = target.depth
        descriptor.depthAttachment.loadAction = .clear
        descriptor.depthAttachment.storeAction = .dontCare
        descriptor.depthAttachment.clearDepth = 1
        return descriptor
    }

    /// Draws the current scene into a `width` x `height` texture and reads it back. Blocks until the GPU is done.
    func renderOffscreen(width: Int, height: Int) throws -> OffscreenImage {
        let target = try makeOffscreenTarget(width: width, height: height)
        guard let ticket = gate.acquire(intoWindow: false, patience: .seconds(10)) else { throw Failure.busy }
        try submit(ticket, descriptor: descriptor(for: target), size: CGSize(width: width, height: height), drawable: nil,
                   waitUntilCompleted: true)
        var bytes = [UInt8](repeating: 0, count: 4 * width * height)
        bytes.withUnsafeMutableBytes {
            target.color.getBytes($0.baseAddress!, bytesPerRow: 4 * width,
                                  from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        return OffscreenImage(width: width, height: height, bgra: bytes)
    }

    /// Draws the current scene into `target` without waiting for the GPU, the way a draw into the window goes: false, and
    /// nothing drawn, when all three buffer sets are in flight (the window would try again shortly; the bench counts a
    /// dropped draw).
    @discardableResult
    func drawNow(into target: OffscreenTarget) throws -> Bool {
        guard let ticket = gate.acquire(intoWindow: false, patience: .milliseconds(0)) else { return false }
        try submit(ticket, descriptor: descriptor(for: target), size: CGSize(width: target.width, height: target.height),
                   drawable: nil, waitUntilCompleted: false)
        return true
    }

    // MARK: Drawing

    /// Encodes and commits one draw into `descriptor`, with the buffer set of `ticket`, which goes back to the gate when
    /// the GPU has finished (at once, if this throws).
    private func submit(_ ticket: DrawGate.Ticket, descriptor: MTLRenderPassDescriptor, size: CGSize,
                        drawable: (any MTLDrawable)?, waitUntilCompleted: Bool) throws {
        let started = CACurrentMediaTime()
        let slot = slots[ticket.slot]
        guard let commands = queue.makeCommandBuffer() else {
            gate.release(ticket)
            throw Failure.encoding
        }
        // The CPU writes the set before the encoder exists: the GPU has not been told about this draw yet.
        do {
            if mode == .world {
                try refreshWorld(slot)
            } else if let header, let frame {
                try refresh(slot, from: frame, header: header)
            }
        } catch {
            gate.release(ticket)
            throw error
        }
        guard let encoder = commands.makeRenderCommandEncoder(descriptor: descriptor) else {
            gate.release(ticket)
            throw Failure.encoding
        }
        encode(into: encoder, slot: slot, size: size)
        encoder.endEncoding()
        let cpuMs = (CACurrentMediaTime() - started) * 1000
        if let drawable {
            commands.present(drawable)
            if let presented = presentedHandler {
                drawable.addPresentedHandler { presented($0.presentedTime) }
            }
        }
        let giveBack = ticket.giveBack
        let timing = timingHandler
        commands.addCompletedHandler { done in
            giveBack()
            timing?(DrawTiming(cpuMs: cpuMs, gpuMs: (done.gpuEndTime - done.gpuStartTime) * 1000))
        }
        commands.commit()
        gate.commit(ticket)
        if waitUntilCompleted { commands.waitUntilCompleted() }
    }

    private func uniforms(header: StreamHeader, size: CGSize) -> SceneUniforms {
        let aspect = Float(size.width / size.height)
        return SceneUniforms(
            viewProjection: camera.viewProjectionMatrix(aspect: aspect),
            eye: SIMD4(camera.position, 1),
            right: SIMD4(camera.right, 0),
            up: SIMD4(camera.up, 0),
            world: SIMD4(Float(header.size), Float(header.height), Float(header.radius), 0),
            counts: SIMD4(UInt32(header.capacity), UInt32(frame?.nBonds ?? 0), UInt32(header.dims), 0),
            viewport: SIMD4(Float(size.width), Float(size.height), 0, 0))
    }

    /// The planes of a scene: in 3D the floor and the surface (with the frame option), and the field slice when the frame
    /// has one, at the middle of its layer.
    nonisolated static func planes(header: StreamHeader, frame: StreamFrame?, display: ViewerDisplay) -> [ScenePlane] {
        var planes: [ScenePlane] = []
        if header.dims == 3, display.frame {
            planes.append(ScenePlane(z: 0, color: SIMD4(0.2, 0.32, 0.5, 0.22), textured: false))
            planes.append(ScenePlane(z: Float(header.height), color: SIMD4(0.5, 0.65, 0.9, 0.1), textured: false))
        }
        if let layer = frame?.fieldLayer, frame?.field != nil {
            // A 3D slice is see-through; the 2D field is the background of the picture.
            let z = header.dims == 3 ? (Float(layer) + 0.5) * Float(header.fieldCell) : 0
            planes.append(ScenePlane(z: z, color: SIMD4(1, 1, 1, header.dims == 3 ? 0.6 : 1), textured: true))
        }
        return planes
    }

    private func encode(into encoder: any MTLRenderCommandEncoder, slot: Slot, size: CGSize) {
        guard let header else { return }
        if mode == .world {
            encodeWorld(into: encoder, slot: slot, header: header, size: size)
            return
        }
        var scene = uniforms(header: header, size: size)
        encoder.setCullMode(.none)
        encoder.setVertexBytes(&scene, length: MemoryLayout<SceneUniforms>.stride, index: 0)
        encoder.setFragmentBytes(&scene, length: MemoryLayout<SceneUniforms>.stride, index: 0)

        var planes = ScenePlane.backToFront(Self.planes(header: header, frame: frame, display: display),
                                            eyeZ: camera.position.z)
        func draw(_ plane: ScenePlane) {
            var constants = PlaneUniforms(color: plane.color, info: SIMD4(plane.z, 0, 0, 0))
            encoder.setVertexBytes(&constants, length: MemoryLayout<PlaneUniforms>.stride, index: 6)
            encoder.setFragmentBytes(&constants, length: MemoryLayout<PlaneUniforms>.stride, index: 6)
            if plane.textured {
                encoder.setRenderPipelineState(slicePipeline)
                encoder.setFragmentTexture(slot.field, index: 0)
            } else {
                encoder.setRenderPipelineState(planePipeline)
            }
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }

        // The 2D field is the background of everything else: drawn first, it neither hides nor z-fights with the agents
        // and bonds, which lie in its plane.
        if header.dims == 2, let background = planes.first(where: { $0.textured }), frame?.field != nil {
            encoder.setDepthStencilState(overlayDepth)
            draw(background)
            planes.removeAll { $0.textured }
        }

        encoder.setDepthStencilState(opaqueDepth)
        if let frame {
            encoder.setVertexBuffer(slot.alive, offset: 0, index: 1)
            encoder.setVertexBuffer(slot.pos, offset: 0, index: 2)
            encoder.setVertexBuffer(slot.orient, offset: 0, index: 3)
            encoder.setVertexBuffer(slot.rgb, offset: 0, index: 4)
            encoder.setVertexBuffer(slot.bonds, offset: 0, index: 5)
            if display.agents, frame.nAlive > 0 {
                encoder.setRenderPipelineState(spherePipeline)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: header.capacity)
            }
            if display.bonds, frame.nBonds > 0, slot.bonds != nil {
                encoder.setRenderPipelineState(bondPipeline)
                encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: 4, instanceCount: frame.nBonds)
            }
        }
        if display.frame {
            encoder.setRenderPipelineState(boxPipeline)
            encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: 24)
        }
        encoder.setDepthStencilState(overlayDepth)
        for plane in planes { draw(plane) }
    }

    // MARK: World pass

    private func worldUniforms(header: StreamHeader, pass: UInt32, size: CGSize) -> WorldUniforms {
        let L = Float(header.size)
        let hasMap = mapNow != nil
        let display = worldDraw.display
        var bits: UInt32 = 0
        if display.map, hasMap { bits |= WorldUniforms.mapBit | WorldUniforms.equatorBit }
        if display.territory != .off, playTerritory != nil { bits |= WorldUniforms.territoryBit }
        let territoryOn = bits & WorldUniforms.territoryBit != 0
        let tint = territoryOn && display.territory == .tint ? MapPalette.territoryTint : 0
        let limits = OrganismSizePolicy.limits(
            mode: display.sizeMode, cover: worldDraw.coverRadius, distance: camera.distance, size: L,
            agentRadius: Float(header.radius), viewHeight: Float(size.height), fovY: camera.fovY,
            pixelsPerPoint: worldDraw.pixelsPerPoint)
        return WorldUniforms(
            target: SIMD4(camera.target.x, camera.target.y, L, 1 / L),
            params: SIMD4(worldDraw.mapBlend, WorldLevels.mapOpacity(distance: camera.distance, size: L), tint,
                          WorldLevels.flatDiscPixels),
            refs: SIMD4(mapNow?.pRef ?? 0, mapNow?.dcRef ?? 0, mapNow?.zetaRef ?? 0, 1 / tan(camera.fovY / 2)),
            flags: SIMD4(pass, UInt32(playFrame?.n ?? 0), UInt32(playFrame?.nBonds ?? 0), bits),
            style: SIMD4(0.55, 1.5, limits.minimum, OutlineContrast.width),
            size: SIMD4(limits.cap.isFinite ? limits.cap : WorldUniforms.unbounded, OrganismSizePolicy.rampLow,
                        OrganismSizePolicy.rampHigh, OutlineContrast.depthCueSlope),
            look: SIMD4(display.exposure, MapPalette.territoryBorder, OutlineContrast.depthCueBase,
                        MapPalette.hemisphereWeight(distance: camera.distance / L)),
            patch: Self.patchUniform(worldDraw.copies, y: camera.target.y, size: L))
    }

    /// The `patch` uniform of a plan of copies: the cell of the view's centre and the alphas of the three cells around it.
    nonisolated static func patchUniform(_ copies: [SeamCopies.Copy]?, y: Float, size: Float) -> SIMD4<Float> {
        let k = SeamCopies.base(y: y, size: size)
        func alpha(_ offset: Int) -> Float { copies.map { $0.first { $0.k == k + offset }?.alpha ?? 0 } ?? 1 }
        return SIMD4(Float(k), alpha(-1), alpha(0), alpha(1))
    }

    private func encodeWorld(into encoder: any MTLRenderCommandEncoder, slot: Slot, header: StreamHeader, size: CGSize) {
        let aspect = Float(size.width / size.height)
        var scene = SceneUniforms(
            viewProjection: camera.viewProjectionMatrix(aspect: aspect),
            eye: SIMD4(camera.position, 1),
            right: SIMD4(camera.right, 0),
            up: SIMD4(camera.up, 0),
            world: SIMD4(Float(header.size), Float(header.height), Float(header.radius), 0),
            counts: SIMD4(UInt32(playFrame?.n ?? 0), UInt32(playFrame?.nBonds ?? 0), UInt32(header.dims), 0),
            viewport: SIMD4(Float(size.width), Float(size.height), 0, 0))
        encoder.setCullMode(.none)
        encoder.setVertexBytes(&scene, length: MemoryLayout<SceneUniforms>.stride, index: 0)
        encoder.setFragmentBytes(&scene, length: MemoryLayout<SceneUniforms>.stride, index: 0)
        let display = worldDraw.display
        let w = slot.world

        func setWorld(_ u: inout WorldUniforms) {
            encoder.setVertexBytes(&u, length: MemoryLayout<WorldUniforms>.stride, index: 7)
            encoder.setFragmentBytes(&u, length: MemoryLayout<WorldUniforms>.stride, index: 7)
        }

        // The map first, with no depth write: everything after it is drawn over it, in 2D too (where the floor is the plane
        // of the agents). A world with no weather and no territory to show has nothing to draw here.
        var u = worldUniforms(header: header, pass: 0, size: size)
        if u.flags.w & (WorldUniforms.mapBit | WorldUniforms.territoryBit) != 0 {
            encoder.setDepthStencilState(overlayDepth)
            encoder.setRenderPipelineState(mapPipeline)
            setWorld(&u)
            let hasMap = mapNow != nil && w.currentSerial >= 0
            for (set, base) in zip([w.current, w.previous], [0, 4]) {
                encoder.setFragmentTexture(hasMap ? set.rain : blankPlane, index: base)
                encoder.setFragmentTexture(hasMap ? set.cloud : blankPlane, index: base + 1)
                encoder.setFragmentTexture(hasMap ? set.zeta : blankPlane, index: base + 2)
                encoder.setFragmentTexture(hasMap ? set.sea : blankPlane, index: base + 3)
            }
            encoder.setFragmentTexture(playTerritory != nil ? w.territory : blankTerritory, index: 8)
            // The one-world patch and its copies across the wrap in y: three instances, those not wanted collapse in the vertex shader.
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: 3)
        }

        // The agents of the newest frame.
        encoder.setDepthStencilState(opaqueDepth)
        if let frame = playFrame, frame.n > 0, w.pos != nil {
            encoder.setVertexBuffer(w.pos, offset: 0, index: 2)
            encoder.setVertexBuffer(w.orient, offset: 0, index: 3)
            encoder.setVertexBuffer(w.rgb, offset: 0, index: 4)
            encoder.setVertexBuffer(w.bonds, offset: 0, index: 5)
            let layers = WorldLevels.agentLayers(at: worldDraw.level, display: display)
            if layers.contains(.points) {
                encoder.setDepthStencilState(overlayDepth)
                encoder.setRenderPipelineState(pointPipeline)
                setWorld(&u)
                encoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: frame.n)
                encoder.setDepthStencilState(opaqueDepth)
            }
            if layers.contains(.instances) {
                encoder.setRenderPipelineState(agentPipeline)
                setWorld(&u)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: frame.n)
                var flat = worldUniforms(header: header, pass: 1, size: size)
                encoder.setRenderPipelineState(flatPipeline)
                setWorld(&flat)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: frame.n)
            }
            if display.bonds, worldDraw.level != .planet, frame.nBonds > 0, w.bonds != nil {
                encoder.setRenderPipelineState(worldBondPipeline)
                setWorld(&u)
                encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: 4, instanceCount: frame.nBonds)
            }
        }
        if display.frame {
            encoder.setRenderPipelineState(boxPipeline)
            encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: 24)
        }
    }

    /// Writes the newest frame, map and territory into `slot` unless it already holds them. Under the draw's ticket, before
    /// the encoder exists: the GPU has not been told about this draw yet, and no committed draw reads this set.
    private func refreshWorld(_ slot: Slot) throws {
        let w = slot.world
        if let frame = playFrame, w.frameSerial != playFrameSerial {
            guard copy(frame.pos, into: &w.pos), copy(frame.orient, into: &w.orient), copy(frame.rgb, into: &w.rgb),
                  copy(frame.bonds, into: &w.bonds, slack: true) else {
                throw Failure.allocation("the agent buffers")
            }
            w.frameSerial = playFrameSerial
        }
        if let now = mapNow, let previous = mapPrevious {
            let plan = MapUploadPlan.make(have: (w.currentSerial, w.previousSerial),
                                          want: (mapNowSerial, mapPreviousSerial))
            if plan.swap {
                swap(&w.current, &w.previous)
                swap(&w.currentSerial, &w.previousSerial)
            }
            if plan.uploadCur {
                try upload(now, into: &w.current)
                w.currentSerial = mapNowSerial
            }
            if plan.uploadPrev {
                try upload(previous, into: &w.previous)
                w.previousSerial = mapPreviousSerial
            }
        }
        if let territory = playTerritory, w.territorySerial != playTerritorySerial {
            if w.territory?.width != territory.nx || w.territory?.height != territory.ny {
                let d = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .rgba8Unorm, width: territory.nx, height: territory.ny, mipmapped: false)
                d.usage = .shaderRead
                d.storageMode = .shared
                w.territory = device.makeTexture(descriptor: d)
            }
            guard let texture = w.territory else { throw Failure.allocation("the territory texture") }
            TerritoryImage.rgba(territory).withUnsafeBytes {
                texture.replace(region: MTLRegionMake2D(0, 0, territory.nx, territory.ny), mipmapLevel: 0,
                                withBytes: $0.baseAddress!, bytesPerRow: 4 * territory.nx)
            }
            w.territorySerial = playTerritorySerial
        }
    }

    /// Writes one delivery of the weather map into `textures`, creating them for its size on first use.
    private func upload(_ map: PlayWeatherMap, into textures: inout MapTextures) throws {
        func texture(_ current: (any MTLTexture)?, _ format: MTLPixelFormat, side: Int) throws -> any MTLTexture {
            if let current, current.width == side, current.height == side, current.pixelFormat == format { return current }
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: side, height: side, mipmapped: false)
            d.usage = .shaderRead
            d.storageMode = .shared
            guard let made = device.makeTexture(descriptor: d) else { throw Failure.allocation("a map texture") }
            return made
        }
        func write<T>(_ values: [T], to texture: any MTLTexture, side: Int, bytesPerPixel: Int) {
            values.withUnsafeBytes {
                texture.replace(region: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0, withBytes: $0.baseAddress!,
                                bytesPerRow: bytesPerPixel * side)
            }
        }
        let n = map.n, nw = map.nw
        textures.rain = try texture(textures.rain, .r8Unorm, side: n)
        textures.cloud = try texture(textures.cloud, .r8Unorm, side: n)
        textures.zeta = try texture(textures.zeta, .r8Snorm, side: n)
        textures.sea = try texture(textures.sea, .r8Unorm, side: n)
        textures.winds = try texture(textures.winds, .rgba8Snorm, side: nw)
        write(map.p, to: textures.rain!, side: n, bytesPerPixel: 1)
        write(map.dc, to: textures.cloud!, side: n, bytesPerPixel: 1)
        write(map.zeta, to: textures.zeta!, side: n, bytesPerPixel: 1)
        write(map.ts, to: textures.sea!, side: n, bytesPerPixel: 1)
        // The block means of u1, v1, u2, v2 (four planes of nw x nw) as the channels of one texture.
        let cells = nw * nw
        var interleaved = [Int8](repeating: 0, count: 4 * cells)
        for cell in 0..<cells {
            for plane in 0..<4 { interleaved[4 * cell + plane] = map.winds[plane * cells + cell] }
        }
        write(interleaved, to: textures.winds!, side: nw, bytesPerPixel: 4)
    }

    // MARK: Upload

    /// Copies the frame into `slot` unless the slot already holds it.
    private func refresh(_ slot: Slot, from frame: StreamFrame, header: StreamHeader) throws {
        guard slot.serial != frameSerial else { return }
        guard copy(frame.alive, into: &slot.alive), copy(frame.pos, into: &slot.pos),
              copy(frame.orient, into: &slot.orient), copy(frame.rgb, into: &slot.rgb),
              copy(frame.bonds, into: &slot.bonds, slack: true) else {
            throw Failure.allocation("the agent buffers")
        }
        if let field = frame.field {
            let cells = header.fieldGrid
            if slot.field?.width != cells {
                let d = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .bgra8Unorm, width: cells, height: cells, mipmapped: false)
                d.usage = .shaderRead
                d.storageMode = .shared
                slot.field = device.makeTexture(descriptor: d)
            }
            guard let texture = slot.field else { throw Failure.allocation("the field texture") }
            field.withUnsafeBytes {
                texture.replace(region: MTLRegionMake2D(0, 0, cells, cells), mipmapLevel: 0,
                                withBytes: $0.baseAddress!, bytesPerRow: 4 * cells)
            }
        }
        slot.serial = frameSerial
    }

    /// Copies `values` into a shared buffer, creating or growing it (`slack` leaves room, for a count that varies).
    private func copy<T>(_ values: [T], into buffer: inout (any MTLBuffer)?, slack: Bool = false) -> Bool {
        let bytes = values.count * MemoryLayout<T>.stride
        guard bytes > 0 else { return true }
        if (buffer?.length ?? 0) < bytes {
            buffer = device.makeBuffer(length: slack ? bytes + bytes / 2 : bytes, options: .storageModeShared)
        }
        guard let target = buffer else { return false }
        values.withUnsafeBytes { target.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes) }
        return true
    }
}

// MARK: - PNG

enum PNG {
    enum Failure: Error, CustomStringConvertible {
        case encode(String)
        var description: String {
            switch self {
            case .encode(let why): return "cannot write the PNG: \(why)"
            }
        }
    }

    /// Writes `image` to `url` as a PNG, creating the directory if needed.
    static func write(_ image: OffscreenImage, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let provider = CGDataProvider(data: Data(image.bgra) as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB),
              let picture = CGImage(
                  width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                  bytesPerRow: 4 * image.width, space: space,
                  bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue
                                           | CGImageAlphaInfo.noneSkipFirst.rawValue),
                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithURL(
                  url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw Failure.encode("cannot create the image")
        }
        CGImageDestinationAddImage(destination, picture, nil)
        guard CGImageDestinationFinalize(destination) else { throw Failure.encode("the encoder failed") }
    }
}

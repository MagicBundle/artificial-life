import AlifeCore
import AppKit
import Metal
import MetalKit
import QuartzCore
import simd

/// The constants of a draw of the sphere world (buffer 0): `SphereU` in `ShadersSphere`, field by field. 272 bytes (a test pins it).
struct SphereUniforms: Equatable {
    var viewProjection = matrix_identity_float4x4
    var eye = SIMD4<Float>(0, 0, 0, 1)
    var right = SIMD4<Float>(1, 0, 0, 0)
    var up = SIMD4<Float>(0, 1, 0, 0)
    var forward = SIMD4<Float>(0, 0, -1, 0)
    /// x: Rp, y: the lid's radius, z: the agent radius, w: tan(fovY / 2).
    var planet = SIMD4<Float>(1, 1, 1, 1)
    /// x, y: the drawable in pixels, z: its aspect, w: the cloud shell's alpha scale.
    var viewport = SIMD4<Float>(1, 1, 1, 1)
    /// x: the field Nside, y: the territory Nside, z: agents listed, w: segments.
    var nsides = SIMD4<Int32>(2, 2, 0, 0)
    /// x: exposure (the colours carry it: 1), y: territory tint alpha, z: territory border alpha, w: graticule alpha.
    var look = SIMD4<Float>(1, 0, 0, 0)
    /// x: the smallest agent radius in pixels, y: the largest, z: agent alpha, w: cell border alpha.
    var style = SIMD4<Float>(2.5, 16, 1, 0)
    /// x: bit 0 floor map, 1 cloud shell, 2 territory tint, 3 territory borders, 4 graticule, 5 height colour; y: the selected field pixel
    /// (`noPixel` for none).
    var flags = SIMD4<UInt32>(0, SphereUniforms.noPixel, 0, 0)
    /// xyz: the direction to the sun, w: the ambient light.
    var sun = SIMD4<Float>(0.4, 0.5, 0.8, 0.34)
    /// x: the shell's height.
    var colours = SIMD4<Float>(1, 0, 0, 0)

    static let noPixel = UInt32.max
    static let floorMap: UInt32 = 1, cloudShell: UInt32 = 2, territoryTint: UInt32 = 4, territoryBorders: UInt32 = 8
    static let graticule: UInt32 = 16, heightColour: UInt32 = 32, unlit: UInt32 = 64
}

/// A line of the overlay in the shader's layout (`Seg`): 64 bytes. Plain floats, because the shader's `packed_float3` is 12 bytes.
struct SphereSegment: Equatable {
    var ax: Float, ay: Float, az: Float, widthA: Float
    var bx: Float, by: Float, bz: Float, widthB: Float
    var colorA: SIMD4<Float>
    var colorB: SIMD4<Float>

    init(_ line: SphereLine) {
        (ax, ay, az, widthA) = (line.a.x, line.a.y, line.a.z, line.widthA)
        (bx, by, bz, widthB) = (line.b.x, line.b.y, line.b.z, line.widthB)
        colorA = line.colorA
        colorB = line.colorB
    }
}

/// What the renderer needs to be told about the camera and the layers, apart from the data buffers.
struct SphereDraw: Equatable {
    var camera: SphereCamera
    var display = WorldDisplay()
    /// The kin group of a territory highlight is in the colours; the renderer shows the selected field pixel as a gold cell.
    var selectedPixel: Int?
    /// A flat-shaded organism cap: true draws symbols whose size is held between the least and the cap in pixels, false at physical size.
    var cappedSize = true
    /// The drawable's pixels per point (a Retina display is 2).
    var pixelsPerPoint: Float = 2
    /// The cell borders show when the eye is closer than this height above the floor (0: never).
    var cellBorderHeight: Float = 60
    /// The floor's cell colours as they are, with no light, lines or glow (the tests read the shader's lookup back through it).
    var unlit = false
}

/// Draws the sphere world with Metal (DESIGN §46.8): the planet and its lid as ray-cast spheres whose colour is looked up by HEALPix pixel,
/// the lines of the overlay, and the organisms as instanced impostors, in one pass with 4x multisampling. It is the `MTKViewDelegate` of
/// `SphereMTKView` and it draws into a texture for the snapshot and the GPU tests (`renderOffscreen`). The buffers it draws from are
/// replaced, never written in place, so a draw in flight keeps the ones it was encoded with.
@MainActor
final class SphereRenderer: NSObject, MTKViewDelegate {
    enum Failure: Error, CustomStringConvertible {
        case noDevice
        case shaders(String)
        case missingFunction(String)
        case pipeline(String, String)
        case noQueue
        case allocation(String)
        case encoding

        var description: String {
            switch self {
            case .noDevice: return "this Mac has no Metal device"
            case .shaders(let log): return "the planet's shaders do not compile: \(log)"
            case .missingFunction(let name): return "the shader function \(name) is missing"
            case .pipeline(let name, let why): return "cannot build the \(name) pipeline: \(why)"
            case .noQueue: return "cannot create a Metal command queue"
            case .allocation(let what): return "cannot allocate \(what)"
            case .encoding: return "cannot encode the drawing commands"
            }
        }
    }

    static let colorFormat = MTLPixelFormat.bgra8Unorm
    static let depthFormat = MTLPixelFormat.depth32Float
    static let sampleCount = 4
    static let background = MTLClearColor(red: 0.016, green: 0.019, blue: 0.032, alpha: 1)

    let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let library: any MTLLibrary
    private let skyPipeline: any MTLRenderPipelineState
    private let floorPipeline: any MTLRenderPipelineState
    private let lidPipeline: any MTLRenderPipelineState
    private let segmentPipeline: any MTLRenderPipelineState
    private let agentPipeline: any MTLRenderPipelineState
    private let alwaysDepth: any MTLDepthStencilState
    private let writeDepth: any MTLDepthStencilState
    private let testDepth: any MTLDepthStencilState

    // MARK: Scene

    private(set) var sphere: SphereHeader?
    private var bodyRadius: Float = 1
    private var draw: SphereDraw?
    var camera: SphereCamera? {
        get { draw?.camera }
        set { if let newValue { draw?.camera = newValue } }
    }
    private var floorColours: any MTLBuffer
    private var cloudColours: any MTLBuffer
    private var territoryColours: any MTLBuffer
    private var agentPositions: (any MTLBuffer)?
    private var agentOrient: (any MTLBuffer)?
    private var agentColours: (any MTLBuffer)?
    private(set) var agentCount = 0
    private var overlay: (any MTLBuffer)?
    private var overlayCount = 0
    private var dynamic: (any MTLBuffer)?
    private var dynamicCount = 0
    /// The lines drawn after the lid, so that the cloud never veils them: the storms' rings and the marker of the selection.
    private var top: (any MTLBuffer)?
    private var topCount = 0
    private var territoryNside = 2
    private var fieldNside = 2
    /// The cloud shell's opacity (1: as the weather says).
    var cloudAlpha: Float = 1

    /// Told when a draw fails (once per distinct message), so the window can say so instead of staying blank.
    var failureHandler: (@MainActor (String) -> Void)?
    private var lastFailure: String?
    /// Told when the drawable's size changes.
    var onResize: (@MainActor (CGSize) -> Void)?
    private var lastSize = CGSize.zero

    init(device: any MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw Failure.noQueue }
        self.queue = queue
        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        do {
            library = try device.makeLibrary(source: ShadersSphere.source, options: options)
        } catch {
            throw Failure.shaders("\(error)")
        }
        let library = self.library
        func function(_ name: String) throws -> any MTLFunction {
            guard let f = library.makeFunction(name: name) else { throw Failure.missingFunction(name) }
            return f
        }
        func pipeline(_ name: String, vertex: String, fragment: String, blend: Bool) throws -> any MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.label = name
            d.vertexFunction = try function(vertex)
            d.fragmentFunction = try function(fragment)
            d.rasterSampleCount = Self.sampleCount
            d.colorAttachments[0].pixelFormat = Self.colorFormat
            d.depthAttachmentPixelFormat = Self.depthFormat
            if blend {  // premultiplied alpha
                let c = d.colorAttachments[0]!
                c.isBlendingEnabled = true
                c.rgbBlendOperation = .add
                c.alphaBlendOperation = .add
                c.sourceRGBBlendFactor = .one
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
        skyPipeline = try pipeline("sky", vertex: ShadersSphere.skyVertex, fragment: ShadersSphere.skyFragment, blend: false)
        floorPipeline = try pipeline("floor", vertex: ShadersSphere.skyVertex, fragment: ShadersSphere.floorFragment, blend: false)
        lidPipeline = try pipeline("lid", vertex: ShadersSphere.skyVertex, fragment: ShadersSphere.lidFragment, blend: true)
        segmentPipeline = try pipeline("segments", vertex: ShadersSphere.segmentVertex, fragment: ShadersSphere.segmentFragment, blend: true)
        agentPipeline = try pipeline("agents", vertex: ShadersSphere.agentVertex, fragment: ShadersSphere.agentFragment, blend: true)
        func depth(_ compare: MTLCompareFunction, write: Bool) throws -> any MTLDepthStencilState {
            let d = MTLDepthStencilDescriptor()
            d.depthCompareFunction = compare
            d.isDepthWriteEnabled = write
            guard let s = device.makeDepthStencilState(descriptor: d) else { throw Failure.allocation("a depth state") }
            return s
        }
        alwaysDepth = try depth(.always, write: false)
        writeDepth = try depth(.lessEqual, write: true)
        testDepth = try depth(.lessEqual, write: false)
        func blank(_ label: String) throws -> any MTLBuffer {
            guard let b = device.makeBuffer(length: 64, options: .storageModeShared) else { throw Failure.allocation(label) }
            b.label = label
            return b
        }
        floorColours = try blank("floor")
        cloudColours = try blank("cloud")
        territoryColours = try blank("territory")
        super.init()
    }

    // MARK: Telling the renderer

    /// A new world: the planet's constants, and the camera that looks at it.
    func configure(sphere: SphereHeader?, bodyRadius: Float, camera: SphereCamera?) {
        self.sphere = sphere
        self.bodyRadius = bodyRadius
        guard let sphere, let camera else {
            draw = nil
            agentCount = 0
            overlayCount = 0
            dynamicCount = 0
            topCount = 0
            return
        }
        fieldNside = sphere.nsideField
        territoryNside = sphere.nsideTerritory  // until a TERR says its own
        if draw == nil { draw = SphereDraw(camera: camera) } else { draw?.camera = camera }
    }

    func setDraw(_ new: SphereDraw) { draw = new }

    private func upload(_ colours: [SIMD4<Float>], label: String) -> (any MTLBuffer)? {
        guard !colours.isEmpty else { return nil }
        let buffer = colours.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
        buffer?.label = label
        return buffer
    }

    func setFloor(_ colours: [SIMD4<Float>]) { if let b = upload(colours, label: "floor") { floorColours = b } }
    func setCloud(_ colours: [SIMD4<Float>]) { if let b = upload(colours, label: "cloud") { cloudColours = b } }
    /// The territory map's colours: its Nside is the record's (`12 nside^2` cells), not the header's.
    func setTerritory(_ colours: [SIMD4<Float>]) {
        guard let b = upload(colours, label: "territory") else { return }
        territoryColours = b
        let k = Int(Double(colours.count / 12).squareRoot().rounded())
        if k >= 2, 12 * k * k == colours.count { territoryNside = k }
    }

    /// The organisms of a frame: positions, bearings and colours are uploaded as they are in the record.
    func show(frame: PlayFrame?) {
        guard let frame, frame.n > 0 else {
            agentCount = 0
            return
        }
        func buffer<T>(_ values: [T], _ label: String) -> (any MTLBuffer)? {
            values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 4), options: .storageModeShared) }
        }
        agentPositions = buffer(frame.pos, "positions")
        agentOrient = buffer(frame.orient, "orient")
        agentColours = buffer(frame.rgb, "colours")
        agentCount = frame.n
    }

    /// The organisms of a legacy stream frame (the Lab's viewer): the alive rows only.
    func show(streamFrame frame: StreamFrame?) {
        guard let frame else {
            agentCount = 0
            return
        }
        var pos: [Float] = [], orient: [UInt16] = [], rgb: [UInt32] = []
        for row in 0..<frame.rows where frame.isAlive(row) {
            pos += [frame.pos[3 * row], frame.pos[3 * row + 1], frame.pos[3 * row + 2]]
            orient += [frame.orient[2 * row], frame.orient[2 * row + 1]]
            rgb.append(frame.rgb[row])
        }
        guard !rgb.isEmpty else {
            agentCount = 0
            return
        }
        agentPositions = pos.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
        agentOrient = orient.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
        agentColours = rgb.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
        agentCount = rgb.count
    }

    private func segments(_ lines: [SphereLine]) -> (buffer: (any MTLBuffer)?, count: Int) {
        guard !lines.isEmpty else { return (nil, 0) }
        let list = lines.map(SphereSegment.init)
        let buffer = list.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
        return (buffer, buffer == nil ? 0 : list.count)
    }

    /// The lines that change slowly (storms, pools, winds), and the ones that change with each frame (bonds, the selection).
    func setOverlay(_ lines: [SphereLine]) { (overlay, overlayCount) = segments(lines) }
    func setDynamic(_ lines: [SphereLine]) { (dynamic, dynamicCount) = segments(lines) }
    func setTop(_ lines: [SphereLine]) { (top, topCount) = segments(lines) }

    // MARK: Uniforms

    func uniforms(for size: CGSize) -> SphereUniforms? {
        guard let draw, let sphere else { return nil }
        let aspect = Float(size.width / max(size.height, 1))
        let camera = draw.camera
        var u = SphereUniforms()
        u.viewProjection = camera.viewProjectionMatrix(aspect: aspect)
        u.eye = SIMD4(camera.eye, 1)
        u.right = SIMD4(camera.right, 0)
        u.up = SIMD4(camera.screenUp, 0)
        u.forward = SIMD4(camera.forward, 0)
        u.planet = SIMD4(Float(sphere.planetRadius), Float(sphere.lidRadius), bodyRadius, tan(camera.pose.fovY / 2))
        u.viewport = SIMD4(Float(size.width), Float(size.height), aspect, cloudAlpha)
        u.nsides = SIMD4(Int32(fieldNside), Int32(territoryNside), Int32(agentCount), Int32(overlayCount + dynamicCount + topCount))
        let display = draw.display
        let ppp = draw.pixelsPerPoint
        var flags: UInt32 = 0
        if display.map { flags |= SphereUniforms.floorMap | SphereUniforms.cloudShell }
        var tintAlpha: Float = 0, borderAlpha: Float = 0
        switch display.territory {
        case .off: break
        case .borders:
            flags |= SphereUniforms.territoryBorders
            borderAlpha = 0.55
        case .tint:
            flags |= SphereUniforms.territoryTint | SphereUniforms.territoryBorders
            tintAlpha = 0.28
            borderAlpha = 0.55
        }
        if display.frame { flags |= SphereUniforms.graticule }
        if display.heightColour { flags |= SphereUniforms.heightColour }
        if draw.unlit { flags |= SphereUniforms.unlit }
        // the floor map is always on: a world without a weather map has its own floor colours in the buffer
        flags |= SphereUniforms.floorMap
        u.look = SIMD4(1, tintAlpha, borderAlpha, 0.16)
        let height = max(camera.altitude, 0)
        let sizeCap: Float = draw.cappedSize ? 16 * ppp : 1e6
        u.style = SIMD4(2.4 * ppp, sizeCap, 1, draw.cellBorderHeight > 0 ? 0.28 * (1 - min(max(height / draw.cellBorderHeight, 0), 1)) : 0)
        u.flags = SIMD4(flags, draw.selectedPixel.map { UInt32($0) } ?? SphereUniforms.noPixel, 0, 0)
        // the light comes from over the camera's left shoulder
        let l = simd_normalize(-camera.forward * 0.8 - camera.right * 0.45 + camera.screenUp * 0.5)
        u.sun = SIMD4(l, 0.5)
        let planes = camera.clipPlanes
        u.colours = SIMD4(Float(sphere.shellHeight), planes.near, planes.far, 0)
        return u
    }

    // MARK: Encoding

    private func encode(into encoder: any MTLRenderCommandEncoder, size: CGSize) {
        guard var u = uniforms(for: size) else { return }
        let uniformSize = MemoryLayout<SphereUniforms>.stride
        encoder.setCullMode(.none)

        // the sky: the far plane, nothing written
        encoder.setRenderPipelineState(skyPipeline)
        encoder.setDepthStencilState(alwaysDepth)
        encoder.setFragmentBytes(&u, length: uniformSize, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        // the floor
        encoder.setRenderPipelineState(floorPipeline)
        encoder.setDepthStencilState(writeDepth)
        encoder.setFragmentBytes(&u, length: uniformSize, index: 0)
        encoder.setFragmentBuffer(floorColours, offset: 0, index: 1)
        encoder.setFragmentBuffer(territoryColours, offset: 0, index: 2)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        // the lines of the overlay and of the frame
        encoder.setRenderPipelineState(segmentPipeline)
        encoder.setDepthStencilState(testDepth)
        encoder.setVertexBytes(&u, length: uniformSize, index: 0)
        for (buffer, count) in [(overlay, overlayCount), (dynamic, dynamicCount)] where count > 0 {
            guard let buffer else { continue }
            encoder.setVertexBuffer(buffer, offset: 0, index: 1)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: count)
        }

        // the organisms
        if agentCount > 0, draw?.display.agents == true, let positions = agentPositions, let orient = agentOrient,
           let colours = agentColours {
            encoder.setRenderPipelineState(agentPipeline)
            encoder.setDepthStencilState(writeDepth)
            encoder.setVertexBytes(&u, length: uniformSize, index: 0)
            encoder.setFragmentBytes(&u, length: uniformSize, index: 0)
            encoder.setVertexBuffer(positions, offset: 0, index: 1)
            encoder.setVertexBuffer(orient, offset: 0, index: 2)
            encoder.setVertexBuffer(colours, offset: 0, index: 3)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: agentCount)
        }

        // the lid: the cloud shell and the glow, over everything (not in the unlit view of the tests)
        guard draw?.unlit != true else { return }
        encoder.setRenderPipelineState(lidPipeline)
        encoder.setDepthStencilState(testDepth)
        encoder.setFragmentBytes(&u, length: uniformSize, index: 0)
        encoder.setFragmentBuffer(cloudColours, offset: 0, index: 1)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        // the storms' rings and the marker of the selection, over the cloud
        if topCount > 0, let top {
            encoder.setRenderPipelineState(segmentPipeline)
            encoder.setDepthStencilState(testDepth)
            encoder.setVertexBytes(&u, length: uniformSize, index: 0)
            encoder.setVertexBuffer(top, offset: 0, index: 1)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: topCount)
        }
    }

    // MARK: MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        guard size != lastSize, size.width >= 1, size.height >= 1 else { return }
        lastSize = size
        onResize?(size)
    }

    func draw(in view: MTKView) {
        let size = view.drawableSize
        if size != lastSize, size.width >= 1, size.height >= 1 {
            lastSize = size
            onResize?(size)
        }
        guard size.width >= 1, size.height >= 1, draw != nil else { return }
        guard let descriptor = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
              let commands = queue.makeCommandBuffer(), let encoder = commands.makeRenderCommandEncoder(descriptor: descriptor) else {
            fail(Failure.encoding)
            return
        }
        encode(into: encoder, size: size)
        encoder.endEncoding()
        commands.present(drawable)
        commands.commit()
    }

    private func fail(_ error: Failure) {
        let message = "\(error)"
        guard message != lastFailure else { return }
        lastFailure = message
        failureHandler?(message)
    }

    // MARK: Offscreen

    /// Draws the scene into a `width` x `height` image (4x multisampled and resolved) and reads it back. Blocks until the GPU is done.
    func renderOffscreen(width: Int, height: Int) throws -> OffscreenImage {
        guard width > 0, height > 0 else { throw Failure.allocation("an empty image") }
        func texture(_ format: MTLPixelFormat, samples: Int, storage: MTLStorageMode) throws -> any MTLTexture {
            let d = MTLTextureDescriptor()
            d.textureType = samples > 1 ? .type2DMultisample : .type2D
            d.pixelFormat = format
            d.width = width
            d.height = height
            d.sampleCount = samples
            d.usage = [.renderTarget]
            d.storageMode = storage
            guard let t = device.makeTexture(descriptor: d) else { throw Failure.allocation("the offscreen textures") }
            return t
        }
        let multisampled = try texture(Self.colorFormat, samples: Self.sampleCount, storage: .private)
        let resolved = try texture(Self.colorFormat, samples: 1, storage: .shared)
        let depth = try texture(Self.depthFormat, samples: Self.sampleCount, storage: .private)
        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = multisampled
        descriptor.colorAttachments[0].resolveTexture = resolved
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .multisampleResolve
        descriptor.colorAttachments[0].clearColor = Self.background
        descriptor.depthAttachment.texture = depth
        descriptor.depthAttachment.loadAction = .clear
        descriptor.depthAttachment.storeAction = .dontCare
        descriptor.depthAttachment.clearDepth = 1
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeRenderCommandEncoder(descriptor: descriptor) else {
            throw Failure.encoding
        }
        encode(into: encoder, size: CGSize(width: width, height: height))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw Failure.pipeline("offscreen", "\(error)") }
        var bytes = [UInt8](repeating: 0, count: 4 * width * height)
        bytes.withUnsafeMutableBytes {
            resolved.getBytes($0.baseAddress!, bytesPerRow: 4 * width, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        return OffscreenImage(width: width, height: height, bgra: bytes)
    }

    // MARK: Kernels for the tests

    /// Runs the shader's HEALPix lookup over `directions` (unit vectors) at `nside`: the pixels the GPU computes.
    func healpixOnGPU(_ directions: [SIMD3<Float>], nside: Int) throws -> [Int32] {
        guard let function = library.makeFunction(name: ShadersSphere.healpixKernel) else {
            throw Failure.missingFunction(ShadersSphere.healpixKernel)
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        var packed: [Float] = []
        for d in directions { packed += [d.x, d.y, d.z] }
        guard let input = packed.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
              let output = device.makeBuffer(length: 4 * directions.count, options: .storageModeShared),
              let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            throw Failure.allocation("the kernel's buffers")
        }
        var n = Int32(nside)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&n, length: 4, index: 2)
        encoder.dispatchThreads(MTLSize(width: directions.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(64, directions.count), height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        let pointer = output.contents().bindMemory(to: Int32.self, capacity: directions.count)
        return Array(UnsafeBufferPointer(start: pointer, count: directions.count))
    }

    /// Runs the shader's heading of an agent over positions and bearings `(yaw, pitch)`.
    func headingsOnGPU(positions: [SIMD3<Float>], bearings: [SIMD2<Float>]) throws -> [SIMD3<Float>] {
        guard let function = library.makeFunction(name: ShadersSphere.headingKernel) else {
            throw Failure.missingFunction(ShadersSphere.headingKernel)
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        var packed: [Float] = []
        for p in positions { packed += [p.x, p.y, p.z] }
        var angles: [Float] = []
        for b in bearings { angles += [b.x, b.y] }
        guard let pos = packed.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
              let ang = angles.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
              let output = device.makeBuffer(length: 12 * positions.count, options: .storageModeShared),
              let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            throw Failure.allocation("the kernel's buffers")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(pos, offset: 0, index: 0)
        encoder.setBuffer(ang, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.dispatchThreads(MTLSize(width: positions.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(64, positions.count), height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        let pointer = output.contents().bindMemory(to: Float.self, capacity: 3 * positions.count)
        return (0..<positions.count).map { SIMD3(pointer[3 * $0], pointer[3 * $0 + 1], pointer[3 * $0 + 2]) }
    }
}

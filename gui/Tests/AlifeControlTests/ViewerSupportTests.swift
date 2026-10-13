import AlifeCore
import Foundation
import ImageIO
import Testing
import simd

@testable import AlifeControl

@Suite struct ViewerKeyTests {
    @Test func everyKeyOfTheViewerHasAnAction() {
        let table: [(String, ViewerAction)] = [
            (" ", .togglePause), (".", .step), ("+", .faster), ("=", .faster), ("-", .slower), ("_", .slower),
            ("c", .cycleColor), ("f", .toggleFields), ("4", .toggleFields), ("[", .fieldLayerDown),
            ("]", .fieldLayerUp), ("s", .snapshot), ("p", .snapshot), ("1", .toggleAgents), ("2", .toggleBonds),
            ("3", .toggleFrame), ("o", .toggleCameraMode), ("r", .resetCamera),
        ]
        for (key, action) in table { #expect(ViewerKeys.action(for: key) == action, "key '\(key)'") }
    }

    @Test func caseDoesNotMatterAndOtherKeysDoNothing() {
        #expect(ViewerKeys.action(for: "S") == .snapshot)
        #expect(ViewerKeys.action(for: "C") == .cycleColor)
        for key in ["x", "q", "w", "a", "d", "5", "", "ab"] { #expect(ViewerKeys.action(for: key) == nil) }
    }

    @Test func theSpeedDoublesAndHalvesWithinTheRange() {
        #expect(ViewerSpeed.faster(1) == 2)
        #expect(ViewerSpeed.faster(512) == 1024)
        #expect(ViewerSpeed.faster(1024) == 1024)
        #expect(ViewerSpeed.slower(8) == 4)
        #expect(ViewerSpeed.slower(1) == 1)
        #expect(ViewerSpeed.slower(3) == 1)
        #expect(ViewerSpeed.faster(0) == 2)  // a bad value is read as 1
    }
}

@Suite struct ViewerInputTests {
    @Test func aDragRightTurnsTheViewRightAndADragDownLooksDown() {
        let turn = ViewerInput.turn(dx: 100, dy: 0)
        #expect(turn.yaw < 0)
        #expect(turn.pitch == 0)
        #expect(ViewerInput.turn(dx: 0, dy: 100).pitch < 0)
        #expect(ViewerInput.turn(dx: -100, dy: 0).yaw > 0)
    }

    @Test func scrollingUpComesCloserAndAWheelCountsMoreThanATrackpad() {
        #expect(ViewerInput.dollyFactor(scroll: 5, precise: true) < 1)
        #expect(ViewerInput.dollyFactor(scroll: -5, precise: true) > 1)
        #expect(ViewerInput.dollyFactor(scroll: 1, precise: false) < ViewerInput.dollyFactor(scroll: 1, precise: true))
        #expect(ViewerInput.dollyFactor(scroll: 0, precise: true) == 1)
        // Bounded, so that one violent flick cannot send the camera to infinity.
        #expect(ViewerInput.dollyFactor(scroll: 1e6, precise: false) > 0.2)
        #expect(ViewerInput.dollyFactor(scroll: -1e6, precise: false) < 5)
    }

    @Test func aPinchOutComesCloser() {
        #expect(ViewerInput.dollyFactor(magnification: 0.5) < 1)
        #expect(ViewerInput.dollyFactor(magnification: -0.5) > 1)
        #expect(ViewerInput.dollyFactor(magnification: 0) == 1)
        #expect(ViewerInput.dollyFactor(magnification: -5).isFinite)
    }

    @Test func aFlyingStepIsAFractionOfTheWorld() {
        #expect(ViewerInput.flyStep(size: 120, fast: false) == 2)
        #expect(ViewerInput.flyStep(size: 120, fast: true) == 10)
    }
}

@Suite struct CameraEditTests {
    private func home() -> Camera { Camera.overview(size: 100, height: 40, aspect: 1.5) }

    private func close(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ tolerance: Float = 1e-3) -> Bool {
        simd_length(a - b) < tolerance
    }

    @Test func orbitingKeepsTheTargetAndFlyingKeepsTheEye() {
        var orbit = home()
        let target = orbit.target
        orbit.drag(dx: 120, dy: -40, mode: .orbit, pan: false, viewportHeight: 800)
        #expect(close(orbit.target, target))
        #expect(orbit.yaw != home().yaw)

        var fly = home()
        let eye = fly.position
        fly.drag(dx: 120, dy: -40, mode: .fly, pan: false, viewportHeight: 800)
        #expect(close(fly.position, eye))
        #expect(fly.yaw != home().yaw)
    }

    @Test func draggingGrabsTheSceneNearTheEyeFollowsThePointer() {
        // A point between the eye and the target is dead centre. Drag right: it moves to the right of the picture, as
        // the near side of a thing you hold and turn does.
        var camera = home()
        let near = camera.target - camera.forward * (camera.distance / 2)
        #expect(abs(camera.ndc(of: near, aspect: 1.5)!.x) < 1e-4)
        camera.drag(dx: 100, dy: 0, mode: .orbit, pan: false, viewportHeight: 800)
        #expect(camera.ndc(of: near, aspect: 1.5)!.x > 0.01)
        camera = home()
        camera.drag(dx: -100, dy: 0, mode: .orbit, pan: false, viewportHeight: 800)
        #expect(camera.ndc(of: near, aspect: 1.5)!.x < -0.01)
    }

    @Test func panningMovesTheTargetWithoutTurning() {
        var camera = home()
        let before = camera
        camera.drag(dx: 50, dy: 20, mode: .orbit, pan: true, viewportHeight: 800)
        #expect(camera.yaw == before.yaw && camera.pitch == before.pitch && camera.distance == before.distance)
        #expect(!close(camera.target, before.target))
    }

    @Test func zoomingScalesTheDistanceOrFliesAhead() {
        var orbit = home()
        let distance = orbit.distance
        orbit.zoom(by: 0.5, mode: .orbit, reach: 50)
        #expect(abs(orbit.distance - distance / 2) < 1e-3)

        var fly = home()
        let eye = fly.position
        fly.zoom(by: 0.5, mode: .fly, reach: 50)
        #expect(close(fly.position, eye + fly.forward * 25))
        #expect(fly.distance == distance)
    }

    @Test func theArrowKeysMeanWhatTheModeSays() {
        var fly = home()
        let eye = fly.position
        fly.press(.up, mode: .fly, step: 3)
        #expect(close(fly.position, eye + fly.forward * 3))
        fly.press(.right, mode: .fly, step: 3)
        #expect(close(fly.position, eye + fly.forward * 3 + fly.right * 3))
        fly.press(.pageUp, mode: .fly, step: 3)
        #expect(close(fly.position, eye + fly.forward * 3 + fly.right * 3 + fly.up * 3))

        var orbit = home()
        let target = orbit.target
        orbit.press(.left, mode: .orbit, step: 3)
        #expect(close(orbit.target, target))
        #expect(orbit.yaw != home().yaw)
        let distance = orbit.distance
        orbit.press(.up, mode: .orbit, step: 3)
        #expect(orbit.distance < distance)
        orbit.press(.down, mode: .orbit, step: 3)
        #expect(abs(orbit.distance - distance) < 1e-3)
    }
}

@Suite struct SceneTests {
    @Test func theUniformsHaveTheLayoutOfTheShaders() {
        // `Scene` and `Plane` in Shaders.swift: float4x4, then six 16-byte vectors; two 16-byte vectors.
        #expect(MemoryLayout<SceneUniforms>.size == 160)
        #expect(MemoryLayout<SceneUniforms>.stride == 160)
        #expect(MemoryLayout<SceneUniforms>.offset(of: \.viewProjection) == 0)
        #expect(MemoryLayout<SceneUniforms>.offset(of: \.eye) == 64)
        #expect(MemoryLayout<SceneUniforms>.offset(of: \.right) == 80)
        #expect(MemoryLayout<SceneUniforms>.offset(of: \.up) == 96)
        #expect(MemoryLayout<SceneUniforms>.offset(of: \.world) == 112)
        #expect(MemoryLayout<SceneUniforms>.offset(of: \.counts) == 128)
        #expect(MemoryLayout<SceneUniforms>.offset(of: \.viewport) == 144)
        #expect(MemoryLayout<PlaneUniforms>.size == 32)
        #expect(MemoryLayout<PlaneUniforms>.offset(of: \.info) == 16)
    }

    @Test func translucentPlanesAreDrawnFarthestFirst() {
        let planes = [
            ScenePlane(z: 0, color: .zero, textured: false),
            ScenePlane(z: 40, color: .zero, textured: false),
            ScenePlane(z: 14, color: .zero, textured: true),
        ]
        #expect(ScenePlane.backToFront(planes, eyeZ: 100).map(\.z) == [0, 14, 40])  // above: the floor is farthest
        #expect(ScenePlane.backToFront(planes, eyeZ: -10).map(\.z) == [40, 14, 0])  // below: the surface is
        #expect(ScenePlane.backToFront(planes, eyeZ: 20).map(\.z) == [0, 40, 14])  // inside: by distance
    }

    @Test func theShaderSourceNamesEveryEntryPointTheRendererBuilds() throws {
        let source = Shaders.source
        let found = source.matches(of: #/(?:vertex|fragment)\s+\w+\s+(\w+)\s*\(/#).map { String($0.1) }
        #expect(Set(found) == Set(Shaders.entryPoints))
        #expect(found.count == Shaders.entryPoints.count)
    }

    @Test func theShaderSourceIsBalancedAndHasNoBackslash() {
        // A cheap guard against a typo that only a compile on the GPU would report (the GPU test compiles it).
        #expect(!Shaders.source.contains("\\"))
        #expect(Shaders.source.hasPrefix("#include <metal_stdlib>"))
        // Comments can hold a half-open interval, so count in the code only.
        let code = Shaders.source.split(separator: "\n").map { line in
            line.firstRange(of: "//").map { String(line[..<$0.lowerBound]) } ?? String(line)
        }.joined(separator: "\n")
        for (open, close) in [("{", "}"), ("(", ")"), ("[", "]")] {
            #expect(code.filter { String($0) == open }.count == code.filter { String($0) == close }.count,
                    "unbalanced \(open)\(close)")
        }
    }

    @Test func theShaderStructsMirrorTheSwiftOnes() {
        // The field names and order of `Scene` as the shader declares them, against SceneUniforms' own.
        let scene = Shaders.source.matches(of: #/struct Scene \{([^}]*)\};/#).first.map { String($0.1) } ?? ""
        let names = scene.matches(of: #/(?:float4x4|float4|uint4)\s+(\w+);/#).map { String($0.1) }
        #expect(names == ["viewProjection", "eye", "right", "up", "world", "counts", "viewport"])
    }

    @Test func theScenesPlanesFollowTheWorldAndTheFrame() throws {
        let world3 = try StreamFixtureData(kind: "3d")
        let world2 = try StreamFixtureData(kind: "2d")
        let withField3 = try #require(world3.frames.first { $0.hasField })
        let withoutField3 = try #require(world3.frames.first { !$0.hasField })
        let layer = try #require(withField3.fieldLayer)
        let all = ViewerDisplay()
        var bare = ViewerDisplay()
        bare.frame = false

        // 3D: the floor and the surface, and a slice in the middle of its layer when the frame has one.
        let planes = Renderer.planes(header: world3.header, frame: withField3, display: all)
        #expect(planes.map(\.z) == [0, Float(world3.header.height), (Float(layer) + 0.5) * Float(world3.header.fieldCell)])
        #expect(planes.map(\.textured) == [false, false, true])
        #expect(planes[2].color.w < 1)  // see-through
        #expect(Renderer.planes(header: world3.header, frame: withoutField3, display: all).count == 2)
        #expect(Renderer.planes(header: world3.header, frame: nil, display: all).count == 2)
        #expect(Renderer.planes(header: world3.header, frame: withField3, display: bare).map(\.textured) == [true])
        #expect(Renderer.planes(header: world3.header, frame: withoutField3, display: bare).isEmpty)

        // 2D: no floor or surface; the field is the opaque background in the plane z = 0.
        let withField2 = try #require(world2.frames.first { $0.hasField })
        let flat = Renderer.planes(header: world2.header, frame: withField2, display: all)
        #expect(flat.count == 1 && flat[0].textured && flat[0].z == 0 && flat[0].color.w == 1)
        let noField2 = try #require(world2.frames.first { !$0.hasField })
        #expect(Renderer.planes(header: world2.header, frame: noField2, display: all).isEmpty)
    }
}

@Suite struct PNGTests {
    @Test func aWrittenImageReadsBack() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        // 3 x 2, BGRA: the top row red, green, blue; the bottom row black, white, grey.
        let bgra: [UInt8] = [
            0, 0, 255, 255, /**/ 0, 255, 0, 255, /**/ 255, 0, 0, 255,
            0, 0, 0, 255, /**/ 255, 255, 255, 255, /**/ 128, 128, 128, 255,
        ]
        let image = OffscreenImage(width: 3, height: 2, bgra: bgra)
        let url = scratch.url.appendingPathComponent("media/deep/view3d_7.png")
        try PNG.write(image, to: url)

        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let picture = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        #expect(picture.width == 3 && picture.height == 2)
        var rgba = [UInt8](repeating: 0, count: 3 * 2 * 4)
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        rgba.withUnsafeMutableBytes { bytes in
            let context = CGContext(
                data: bytes.baseAddress, width: 3, height: 2, bitsPerComponent: 8, bytesPerRow: 12, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            context.draw(picture, in: CGRect(x: 0, y: 0, width: 3, height: 2))
        }
        // Row 0 of the image is the first row in memory.
        #expect(Array(rgba[0..<3]) == [255, 0, 0])
        #expect(Array(rgba[4..<7]) == [0, 255, 0])
        #expect(Array(rgba[8..<11]) == [0, 0, 255])
        #expect(Array(rgba[16..<19]) == [255, 255, 255])
        #expect(image.pixel(2, 0) == (r: 0, g: 0, b: 255))
    }
}

// Generates an app icon as an .iconset directory: the AlifeControl icon (cells and bonds) or, with `storm`, the Storm
// Worlds icon (a storm cloud and a lightning bolt over a small world with a few cells).
//
//   swift gui/scripts/make-icon.swift <out.iconset dir> [lab|storm]
//   iconutil -c icns <out.iconset dir> -o AppIcon.icns
//
// Self-contained (CoreGraphics + ImageIO only), deterministic (hard-coded geometry), no text.
// The art is drawn in a 1024 x 1024 design space and scaled to each pixel size.
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct RGB {
    let r: CGFloat, g: CGFloat, b: CGFloat
    func cg(_ a: CGFloat = 1) -> CGColor { CGColor(srgbRed: r, green: g, blue: b, alpha: a) }
    func mixed(with o: RGB, _ t: CGFloat) -> RGB {
        RGB(r: r + (o.r - r) * t, g: g + (o.g - g) * t, b: b + (o.b - b) * t)
    }
}

struct Cell {
    let x: CGFloat, y: CGFloat, r: CGFloat
    let hue: RGB
}

let green = RGB(r: 0.30, g: 0.90, b: 0.45)
let cyan = RGB(r: 0.20, g: 0.85, b: 0.95)
let amber = RGB(r: 1.00, g: 0.72, b: 0.20)
let white = RGB(r: 1, g: 1, b: 1)
let space = CGColorSpace(name: CGColorSpace.sRGB)!

// y grows upward (CoreGraphics default).
let cells: [Cell] = [
    Cell(x: 410, y: 540, r: 128, hue: green),  // 0 large centre
    Cell(x: 640, y: 450, r: 100, hue: amber),  // 1
    Cell(x: 610, y: 670, r: 78, hue: cyan),  // 2
    Cell(x: 420, y: 330, r: 66, hue: cyan),  // 3
    Cell(x: 290, y: 710, r: 58, hue: amber),  // 4
    Cell(x: 770, y: 600, r: 52, hue: green),  // 5
    Cell(x: 730, y: 300, r: 50, hue: green),  // 6
    Cell(x: 235, y: 470, r: 46, hue: cyan),  // 7
]
let bonds: [(Int, Int)] = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 5), (1, 6), (0, 7)]

/// Superellipse ("squircle") path of half-extent `half` around (cx, cy).
func squircle(cx: CGFloat, cy: CGFloat, half: CGFloat, n: CGFloat = 5) -> CGPath {
    let path = CGMutablePath()
    let steps = 256
    for i in 0...steps {
        let t = 2 * CGFloat.pi * CGFloat(i) / CGFloat(steps)
        let c = cos(t), s = sin(t)
        let x = cx + half * (c < 0 ? -1 : 1) * pow(abs(c), 2 / n)
        let y = cy + half * (s < 0 ? -1 : 1) * pow(abs(s), 2 / n)
        if i == 0 { path.move(to: CGPoint(x: x, y: y)) } else { path.addLine(to: CGPoint(x: x, y: y)) }
    }
    path.closeSubpath()
    return path
}

func radial(_ ctx: CGContext, center: CGPoint, radius: CGFloat, inner: CGColor, outer: CGColor,
            innerRadius: CGFloat = 0, focus: CGPoint? = nil) {
    let g = CGGradient(colorsSpace: space, colors: [inner, outer] as CFArray, locations: [0, 1])!
    ctx.drawRadialGradient(g, startCenter: focus ?? center, startRadius: innerRadius, endCenter: center,
                           endRadius: radius, options: [])
}

func render(pixels: Int, variant: String) -> CGImage {
    let ctx = CGContext(data: nil, width: pixels, height: pixels, bitsPerComponent: 8, bytesPerRow: 0,
                        space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.interpolationQuality = .high
    ctx.setAllowsAntialiasing(true)
    let k = CGFloat(pixels) / 1024
    ctx.scaleBy(x: k, y: k)

    // Big Sur+ grid: 824 x 824 shape inside the 1024 canvas (100 px margin).
    let shape = squircle(cx: 512, cy: 512, half: 412)

    // Drop shadow under the shape.
    ctx.saveGState()
    ctx.setShadow(offset: CGSize(width: 0, height: -12), blur: 24, color: CGColor(srgbRed: 0, green: 0, blue: 0, alpha: 0.35))
    ctx.addPath(shape)
    ctx.setFillColor(RGB(r: 0.03, g: 0.12, b: 0.22).cg())
    ctx.fillPath()
    ctx.restoreGState()

    // Everything else is clipped to the shape.
    ctx.saveGState()
    ctx.addPath(shape)
    ctx.clip()

    if variant == "storm" {
        drawStorm(ctx, k: k)
    } else {
        drawCells(ctx, k: k)
    }
    ctx.restoreGState()

    // Thin top-edge highlight on the shape.
    ctx.saveGState()
    ctx.addPath(shape)
    ctx.setLineWidth(5)
    ctx.setStrokeColor(white.cg(0.18))
    ctx.strokePath()
    ctx.restoreGState()

    return ctx.makeImage()!
}

/// The AlifeControl art, drawn inside the clipped shape.
func drawCells(_ ctx: CGContext, k: CGFloat) {
    // Background: deep teal (top-left) to navy (bottom-right).
    let bg = CGGradient(colorsSpace: space,
                        colors: [RGB(r: 0.03, g: 0.36, b: 0.40).cg(), RGB(r: 0.02, g: 0.05, b: 0.18).cg()] as CFArray,
                        locations: [0, 1])!
    ctx.drawLinearGradient(bg, start: CGPoint(x: 140, y: 920), end: CGPoint(x: 880, y: 110), options: [])

    // Faint field: a soft halo plus a few concentric rings.
    radial(ctx, center: CGPoint(x: 500, y: 520), radius: 430, inner: cyan.cg(0.22), outer: cyan.cg(0))
    ctx.setLineWidth(5)
    for (i, rad) in [180.0, 290.0, 400.0].enumerated() {
        ctx.setStrokeColor(cyan.cg(0.13 - 0.03 * CGFloat(i)))
        ctx.strokeEllipse(in: CGRect(x: 500 - rad, y: 520 - rad, width: rad * 2, height: rad * 2))
    }

    // Bonds (kept at least ~1 device pixel wide so they survive at 16 px).
    let bondWidth = max(16, 1.1 / k)
    ctx.setLineCap(.round)
    ctx.setLineWidth(bondWidth)
    for (a, b) in bonds {
        let ca = cells[a], cb = cells[b]
        ctx.setStrokeColor(white.mixed(with: ca.hue, 0.4).cg(0.55))
        ctx.move(to: CGPoint(x: ca.x, y: ca.y))
        ctx.addLine(to: CGPoint(x: cb.x, y: cb.y))
        ctx.strokePath()
    }

    // Cells: glow, body with highlight, rim, nucleus.
    for c in cells {
        let p = CGPoint(x: c.x, y: c.y)
        radial(ctx, center: p, radius: c.r * 2.3, inner: c.hue.cg(0.50), outer: c.hue.cg(0), innerRadius: c.r * 0.6)

        ctx.saveGState()
        let body = CGRect(x: c.x - c.r, y: c.y - c.r, width: c.r * 2, height: c.r * 2)
        ctx.addEllipse(in: body)
        ctx.clip()
        radial(ctx, center: p, radius: c.r * 1.15,
               inner: white.mixed(with: c.hue, 0.35).cg(), outer: c.hue.mixed(with: RGB(r: 0, g: 0.1, b: 0.2), 0.25).cg(),
               focus: CGPoint(x: c.x - c.r * 0.35, y: c.y + c.r * 0.4))
        ctx.restoreGState()

        ctx.setLineWidth(max(6, 0.8 / k))
        ctx.setStrokeColor(white.cg(0.55))
        ctx.strokeEllipse(in: body.insetBy(dx: 2, dy: 2))

        let nr = c.r * 0.30
        ctx.setFillColor(c.hue.mixed(with: RGB(r: 0, g: 0.1, b: 0.2), 0.55).cg(0.75))
        ctx.fillEllipse(in: CGRect(x: c.x + c.r * 0.05 - nr, y: c.y - c.r * 0.05 - nr, width: nr * 2, height: nr * 2))
    }
}

/// The Storm Worlds art, drawn inside the clipped shape: a night sky, the rim of a small world with a few cells, a storm
/// cloud and a lightning bolt. Shapes stay bold so that it reads at 16 px.
func drawStorm(_ ctx: CGContext, k: CGFloat) {
    // Sky: indigo (top) to deep teal (bottom).
    let sky = CGGradient(colorsSpace: space,
                         colors: [RGB(r: 0.12, g: 0.10, b: 0.34).cg(), RGB(r: 0.02, g: 0.20, b: 0.28).cg()] as CFArray,
                         locations: [0, 1])!
    ctx.drawLinearGradient(sky, start: CGPoint(x: 512, y: 980), end: CGPoint(x: 512, y: 80), options: [])

    // The world: a large disc rising from the bottom edge, with a lit rim.
    let planet = CGPoint(x: 512, y: -430)
    let planetR: CGFloat = 800
    radial(ctx, center: planet, radius: planetR + 120, inner: cyan.cg(0.0), outer: cyan.cg(0.30), innerRadius: planetR - 40)
    ctx.saveGState()
    ctx.addEllipse(in: CGRect(x: planet.x - planetR, y: planet.y - planetR, width: planetR * 2, height: planetR * 2))
    ctx.clip()
    radial(ctx, center: planet, radius: planetR, inner: RGB(r: 0.02, g: 0.10, b: 0.16).cg(),
           outer: RGB(r: 0.05, g: 0.38, b: 0.40).cg(), innerRadius: planetR * 0.55)
    ctx.restoreGState()
    ctx.setLineWidth(max(8, 1.0 / k))
    ctx.setStrokeColor(cyan.cg(0.55))
    ctx.strokeEllipse(in: CGRect(x: planet.x - planetR, y: planet.y - planetR, width: planetR * 2, height: planetR * 2))

    // A few cells on the surface.
    let surface: [Cell] = [
        Cell(x: 250, y: 290, r: 52, hue: green), Cell(x: 395, y: 215, r: 38, hue: amber),
        Cell(x: 640, y: 225, r: 44, hue: cyan), Cell(x: 790, y: 300, r: 56, hue: green),
        Cell(x: 520, y: 150, r: 30, hue: cyan),
    ]
    for c in surface {
        let p = CGPoint(x: c.x, y: c.y)
        radial(ctx, center: p, radius: c.r * 2.2, inner: c.hue.cg(0.45), outer: c.hue.cg(0), innerRadius: c.r * 0.5)
        ctx.saveGState()
        ctx.addEllipse(in: CGRect(x: c.x - c.r, y: c.y - c.r, width: c.r * 2, height: c.r * 2))
        ctx.clip()
        radial(ctx, center: p, radius: c.r * 1.15, inner: white.mixed(with: c.hue, 0.35).cg(),
               outer: c.hue.mixed(with: RGB(r: 0, g: 0.1, b: 0.2), 0.25).cg(),
               focus: CGPoint(x: c.x - c.r * 0.35, y: c.y + c.r * 0.4))
        ctx.restoreGState()
    }

    // Rain, behind the bolt.
    ctx.setLineCap(.round)
    ctx.setLineWidth(max(12, 1.2 / k))
    ctx.setStrokeColor(cyan.mixed(with: white, 0.4).cg(0.40))
    let rain: [(CGFloat, CGFloat)] = [
        (215, 590), (300, 520), (385, 585), (230, 440), (330, 380), (700, 590), (790, 520), (870, 590), (745, 400), (840, 430),
    ]
    for (x, y) in rain {
        ctx.move(to: CGPoint(x: x, y: y))
        ctx.addLine(to: CGPoint(x: x - 34, y: y - 96))
        ctx.strokePath()
    }

    // The cloud: a dark underside, then the lit body shifted up.
    let puffs: [(CGFloat, CGFloat, CGFloat)] = [
        (300, 705, 115), (455, 765, 140), (640, 745, 128), (775, 690, 100), (560, 690, 132), (400, 665, 105), (700, 665, 100),
    ]
    ctx.saveGState()
    ctx.setShadow(offset: CGSize(width: 0, height: -10), blur: 30, color: CGColor(srgbRed: 0, green: 0, blue: 0, alpha: 0.45))
    ctx.setFillColor(RGB(r: 0.20, g: 0.23, b: 0.42).cg())
    for (x, y, r) in puffs { ctx.fillEllipse(in: CGRect(x: x - r, y: y - r, width: r * 2, height: r * 2)) }
    ctx.restoreGState()
    for (x, y, r) in puffs {
        ctx.saveGState()
        let body = CGRect(x: x - r * 0.93, y: y - r * 0.93 + 26, width: r * 1.86, height: r * 1.86)
        ctx.addEllipse(in: body)
        ctx.clip()
        radial(ctx, center: CGPoint(x: x, y: y + 26), radius: r * 1.1,
               inner: RGB(r: 0.90, g: 0.93, b: 1.0).cg(), outer: RGB(r: 0.45, g: 0.50, b: 0.72).cg(),
               focus: CGPoint(x: x - r * 0.25, y: y + r * 0.45))
        ctx.restoreGState()
    }

    // The bolt, from the cloud to the ground: glow, then a white-to-amber body.
    let bolt: [CGPoint] = [
        CGPoint(x: 600, y: 650), CGPoint(x: 420, y: 395), CGPoint(x: 528, y: 395), CGPoint(x: 385, y: 105),
        CGPoint(x: 715, y: 478), CGPoint(x: 590, y: 478), CGPoint(x: 725, y: 650),
    ]
    let path = CGMutablePath()
    path.addLines(between: bolt)
    path.closeSubpath()
    ctx.saveGState()
    ctx.setShadow(offset: .zero, blur: 60, color: amber.cg(0.95))
    ctx.addPath(path)
    ctx.setFillColor(amber.cg())
    ctx.fillPath()
    ctx.restoreGState()
    ctx.saveGState()
    ctx.addPath(path)
    ctx.clip()
    let fill = CGGradient(colorsSpace: space, colors: [RGB(r: 1, g: 0.98, b: 0.85).cg(), amber.cg()] as CFArray, locations: [0, 1])!
    ctx.drawLinearGradient(fill, start: CGPoint(x: 560, y: 650), end: CGPoint(x: 480, y: 105), options: [])
    ctx.restoreGState()
    ctx.addPath(path)
    ctx.setLineWidth(max(6, 0.8 / k))
    ctx.setLineJoin(.round)
    ctx.setStrokeColor(white.cg(0.7))
    ctx.strokePath()
}

func writePNG(_ image: CGImage, to url: URL) {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        FileHandle.standardError.write(Data("cannot create \(url.path)\n".utf8))
        exit(1)
    }
    CGImageDestinationAddImage(dest, image, nil)
    if !CGImageDestinationFinalize(dest) {
        FileHandle.standardError.write(Data("cannot write \(url.path)\n".utf8))
        exit(1)
    }
}

let variant = CommandLine.arguments.count == 3 ? CommandLine.arguments[2] : "lab"
guard CommandLine.arguments.count == 2 || CommandLine.arguments.count == 3, ["lab", "storm"].contains(variant) else {
    FileHandle.standardError.write(Data("usage: swift make-icon.swift <out.iconset dir> [lab|storm]\n".utf8))
    exit(2)
}
let outDir = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

// (file name, pixel size) as expected by iconutil.
let sizes: [(String, Int)] = [
    ("icon_16x16", 16), ("icon_16x16@2x", 32),
    ("icon_32x32", 32), ("icon_32x32@2x", 64),
    ("icon_128x128", 128), ("icon_128x128@2x", 256),
    ("icon_256x256", 256), ("icon_256x256@2x", 512),
    ("icon_512x512", 512), ("icon_512x512@2x", 1024),
]
var cache: [Int: CGImage] = [:]
for (name, px) in sizes {
    let image = cache[px] ?? render(pixels: px, variant: variant)
    cache[px] = image
    writePNG(image, to: outDir.appendingPathComponent("\(name).png"))
}

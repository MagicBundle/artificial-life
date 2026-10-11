// Generates the AlifeControl app icon as an .iconset directory.
//
//   swift gui/scripts/make-icon.swift <out.iconset dir>
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

func render(pixels: Int) -> CGImage {
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

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(Data("usage: swift make-icon.swift <out.iconset dir>\n".utf8))
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
    let image = cache[px] ?? render(pixels: px)
    cache[px] = image
    writePNG(image, to: outDir.appendingPathComponent("\(name).png"))
}

import AlifeCore
import simd

/// The globe's constants (`GlobeU` in ShadersGlobe.swift), buffer 8 of the world pass: 176 bytes of eleven 16-byte vectors. On the flat path it is
/// all zero (`shade.w` is the flag the shaders read).
struct GlobeUniforms: Equatable {
    /// x: s, y: L, z: H, w: R.
    var a = SIMD4<Float>.zero
    /// x: the plane's scale `M lambda(s)`, y: `zeta(s)`, z: `w2(s)`, w: `kz`.
    var b = SIMD4<Float>.zero
    /// x: the centre's column x, y: its signed row ybar, z: cos(heading), w: sin(heading).
    var c = SIMD4<Float>.zero
    /// xyz: the centre frame's east axis (world-fixed), w: the centre's latitude.
    var right = SIMD4<Float>.zero
    var up = SIMD4<Float>.zero
    var centre = SIMD4<Float>.zero
    /// x, y, z: the alpha of the copies of the map across the wrap in y (rows -1, 0, 1); w: 1 while a row may be drawn in one.
    var copies = SIMD4<Float>.zero
    /// x: the weight of space against the haze sky, y: the haze's strength, z: the stars' and the atmosphere's, w: the graticule's.
    var look = SIMD4<Float>.zero
    /// x: the weight of the lighting, y: of the rim, z: of the pole caps, w: 1 on the globe path.
    var shade = SIMD4<Float>.zero
    /// xyz: the key light, towards the light, in the camera's world; w: the weight of the haze at the limb.
    var light = SIMD4<Float>.zero
    /// x: the weight of the organism glow (`OrganismSizePolicy.glowAlpha` of the drawn radius at the centre of the view); y: the weight of
    /// the crowd's floor (a cell of 4 rows or more keeps 0.25 of light whatever x says), which belongs to the planet level.
    var glow = SIMD4<Float>.zero

    /// The key light in the camera's view space: above the upper left, in front (DESIGN §36.10; fixed in view space, so rotating the globe
    /// never darkens it and there is no night side).
    static let keyLight = SIMD3<Float>(-0.45, 0.55, 0.70)

    init() {}

    /// The constants of `vantage` with the copies of the map `copies` (a plan of `SeamCopies`; nil: every copy at the alpha of `s`) and the
    /// haze on or off.
    init(_ v: WorldVantage, copies plan: [SeamCopies.Copy]?, fog: Bool) {
        let s = v.s, L = v.size, f = v.frame
        let camera = v.camera
        a = SIMD4(s, L, v.height, GlobeMap.radius(L: L))
        b = SIMD4(Morph.M * Morph.lambda(s: s, latitude: f.latitude), Morph.zeta(s: s, L: L, height: v.height), Morph.w2(s: s),
                  GlobeMap.shellScale(L: L, height: v.height))
        c = SIMD4(f.x, f.ybar, cos(f.heading), sin(f.heading))
        right = SIMD4(f.right, f.latitude)
        up = SIMD4(f.up, 0)
        centre = SIMD4(f.centre, 0)
        let k = SeamCopies.base(y: v.pose.y, size: L)
        func alpha(_ row: Int) -> Float {
            plan.map { $0.first { $0.k == k + row }?.alpha ?? 0 } ?? (row == 0 ? 1 : GlobeLook.tilesAlpha(s: s))
        }
        copies = SIMD4(alpha(-1), alpha(0), alpha(1), GlobeLook.tilesAlpha(s: s) > 0 ? 1 : 0)
        look = SIMD4(GlobeLook.space(s: s), fog ? GlobeLook.fogStrength(s: s) : 0, GlobeLook.stars(s: s), GlobeLook.graticule(s: s))
        glow = SIMD4(0, Morph.smoothstep(0.3, 0.8, s), 0, 0)
        shade = SIMD4(GlobeLook.lit(s: s), GlobeLook.rim(s: s), GlobeLook.caps(s: s), 1)
        let key = Self.keyLight
        let towardEye = -camera.forward
        light = SIMD4(simd_normalize(key.x * camera.right + key.y * camera.up + key.z * towardEye), GlobeLook.limbFog(s: s))
    }

    /// The tiles of the map's mesh to draw (x: the shift in worlds east, y: the shift north, z: the alpha): the central one, its neighbours in
    /// x while `s` is under 0.3 and the view reaches them (`wide`), and the copies across the wrap in y by their alphas.
    func tiles(wide: Bool) -> [SIMD4<Float>] {
        let xAlpha = wide ? GlobeLook.tilesAlpha(s: a.x) : 0
        var out: [SIMD4<Float>] = []
        for iy in -1...1 {
            let ya = copies[iy + 1]
            for ix in -1...1 {
                let alpha = (ix == 0 ? 1 : xAlpha) * ya
                if alpha > 0 { out.append(SIMD4(Float(ix), Float(iy), alpha, 0)) }
            }
        }
        return out
    }
}

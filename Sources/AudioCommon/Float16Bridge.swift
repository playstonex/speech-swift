import Foundation

// MARK: - Float16 bit-pattern bridge
//
// `Float16` is only available on **arm64** (Apple Silicon). On **x86_64**
// (Intel) the Swift standard library marks `Float16` as unavailable —
// regardless of deployment target. Verified empirically:
//
//   swiftc -target arm64-apple-macos15.0   → Float16 compiles ✅
//   swiftc -target x86_64-apple-macos15.0  → error: 'Float16' is unavailable ❌
//
// The Petal app builds as a Universal macOS binary (arm64 + x86_64), so any
// `Float16` reference fails the x86_64 slice. iOS builds are unaffected
// (arm64 only).
//
// CoreML `MLMultiArray` buffers with `.float16` dataType store IEEE-754
// binary16 values as raw 16-bit words. We treat the buffer as `UInt16`
// (the bit pattern) and convert to/from `Float32` manually, keeping the
// code free of the `Float16` type entirely so it compiles for all arches.

/// Convert an IEEE-754 binary16 bit pattern to `Float`.
@inline(__always)
public func float16BitsToFloat(_ bits: UInt16) -> Float {
    let sign = UInt32(bits & 0x8000) << 16
    let exp  = UInt32((bits & 0x7C00) >> 10)
    let mant = UInt32(bits & 0x03FF)
    var result: UInt32

    if exp == 0 {
        if mant == 0 {
            result = sign
        } else {
            // Subnormal: normalize into the float32 range.
            var e: UInt32 = 127 - 15 + 1
            var m = mant
            while (m & 0x0400) == 0 { m <<= 1; e -= 1 }
            m &= 0x03FF
            result = sign | (e << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        // Inf / NaN.
        result = sign | 0x7F80_0000 | (mant << 13)
    } else {
        let newExp = UInt32(Int(exp) - 15 + 127)
        result = sign | (newExp << 23) | (mant << 13)
    }
    return Float(bitPattern: result)
}

/// Convert a `Float` to an IEEE-754 binary16 bit pattern (`UInt16`).
@inline(__always)
public func float32ToFloat16Bits(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let sign = UInt16((bits >> 16) & 0x8000)
    var exp = Int32((bits >> 23) & 0xFF) - 127 + 15
    var mant = bits & 0x7F_FFFF
    if exp <= 0 {
        if exp < -10 { return sign }
        mant |= 0x80_0000
        let shift = 14 - exp
        return sign | UInt16((mant >> shift) & 0x3FF)
    } else if exp >= 31 {
        if mant != 0 { return sign | 0x7E00 }
        return sign | 0x7C00
    } else {
        let expBits = UInt16(exp & 0x1F) << 10
        let mantBits = UInt16((mant >> 13) & 0x3FF)
        return sign | expBits | mantBits
    }
}

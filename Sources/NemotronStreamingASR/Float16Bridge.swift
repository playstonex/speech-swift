import Foundation

// MARK: - Float16 bit-pattern bridge
//
// `Float16` is only available on **arm64** (Apple Silicon). On **x86_64**
// (Intel) the Swift standard库 marks `Float16` as unavailable — regardless
// of deployment target. Confirmed empirically:
//
//   swiftc -target arm64-apple-macos15.0   → Float16 compiles ✅
//   swiftc -target x86_64-apple-macos15.0  → error: 'Float16' is unavailable in macOS ❌
//
// The Petal app's Scenario target builds as a Universal macOS binary
// (arm64 + x86_64), so any `Float16` reference here fails the x86_64 slice.
// iOS builds are unaffected (arm64 only).
//
// CoreML `MLMultiArray` buffers with `.float16` dataType store IEEE-754
// binary16 values as raw 16-bit words. We treat the buffer as `UInt16`
// (the bit pattern) and convert to `Float32` manually, mirroring the
// approach in `MagpieTTSCoreML/NpyReader.swift`. This keeps the code free
// of the `Float16` type entirely so it compiles for both architectures.

/// Convert an IEEE-754 binary16 bit pattern to `Float`.
@inline(__always)
func float16BitsToFloat(_ bits: UInt16) -> Float {
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

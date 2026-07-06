import Foundation

// MARK: - Float16 bit-pattern bridge
//
// `Float16` and its `BinaryFloatingPoint` conformance are only available on
// macOS 11.0+ / iOS 14.0+. This package must build against an effective
// deployment target below that (the Xcode-integrated build inherits a lower
// `MACOSX_DEPLOYMENT_TARGET` than the `15.0` declared in `Package.swift`),
// and under the Swift 6 language mode the availability gap is a hard error
// rather than a warning.
//
// CoreML `MLMultiArray` buffers with `.float16` dataType store IEEE-754
// binary16 values as raw 16-bit words. We treat the buffer as `UInt16`
// (the bit pattern) and convert to `Float32` manually, mirroring the
// approach in `MagpieTTSCoreML/NpyReader.swift`. This keeps the code free
// of the `Float16` type entirely so it compiles on macOS 10.15+.

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

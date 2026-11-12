import Foundation

/// Mix separated backing stems without adding the vocal estimate.
public enum StemMixer {
    public enum MixingError: Error, LocalizedError, Equatable {
        case missingStem(SeparationTarget)
        case invalidStereoShape(SeparationTarget)
        case mismatchedLength(SeparationTarget)
        case nonFiniteSamples(SeparationTarget)

        public var errorDescription: String? {
            switch self {
            case .missingStem(let target):
                return "The instrumental mix requires the \(target.rawValue) stem."
            case .invalidStereoShape(let target):
                return "The \(target.rawValue) stem must contain two nonempty, equally sized channels."
            case .mismatchedLength(let target):
                return "The \(target.rawValue) stem has a different frame count from the other backing stems."
            case .nonFiniteSamples(let target):
                return "The \(target.rawValue) stem contains non-finite samples."
            }
        }
    }

    /// Sum drums, bass and other at their existing relative levels.
    ///
    /// All three inputs must be finite stereo PCM with the same nonzero frame
    /// count. The vocal estimate is ignored. A single gain is applied to both
    /// channels only when the combined absolute peak exceeds 1; quieter mixes
    /// are left at unity gain. Individual stems are not modified.
    public static func instrumental(
        from stems: [SeparationTarget: [[Float]]]
    ) throws -> [[Float]] {
        let targets: [SeparationTarget] = [.drums, .bass, .other]
        var sources: [[[Float]]] = []
        var frameCount: Int?
        for target in targets {
            guard let audio = stems[target] else {
                throw MixingError.missingStem(target)
            }
            guard audio.count == 2, !audio[0].isEmpty,
                  audio[0].count == audio[1].count else {
                throw MixingError.invalidStereoShape(target)
            }
            if let frameCount, audio[0].count != frameCount {
                throw MixingError.mismatchedLength(target)
            }
            guard audio.allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
                throw MixingError.nonFiniteSamples(target)
            }
            frameCount = audio[0].count
            sources.append(audio)
        }

        let count = frameCount!
        // Accumulate in Double before scaling so finite Float inputs cannot
        // overflow during summation. Two passes avoid a second full mix buffer.
        var peak: Double = 0
        for channel in 0..<2 {
            for frame in 0..<count {
                let sample = sources.reduce(0.0) { $0 + Double($1[channel][frame]) }
                peak = max(peak, abs(sample))
            }
        }
        let gain = peak > 1 ? 1 / peak : 1
        var mix = [[Float]](repeating: [Float](repeating: 0, count: count), count: 2)
        for channel in 0..<2 {
            for frame in 0..<count {
                let sample = sources.reduce(0.0) { $0 + Double($1[channel][frame]) }
                mix[channel][frame] = Float(sample * gain)
            }
        }
        return mix
    }
}

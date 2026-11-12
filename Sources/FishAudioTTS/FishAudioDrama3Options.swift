import Foundation

public struct FishAudioDrama3Options: Sendable {
    public var maxFrames: Int
    public var minFrames: Int
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var seed: UInt64
    public var streamingChunkFrames: Int

    public init(maxFrames: Int = 2_048, minFrames: Int = 1, temperature: Float = 1,
                topK: Int = 30, topP: Float = 0.9, seed: UInt64 = 0, streamingChunkFrames: Int = 3) {
        self.maxFrames = maxFrames; self.minFrames = minFrames; self.temperature = temperature
        self.topK = topK; self.topP = topP; self.seed = seed
        self.streamingChunkFrames = streamingChunkFrames
    }

    public func validate() throws {
        guard (1...8_192).contains(maxFrames), (0...maxFrames).contains(minFrames),
              temperature.isFinite, temperature >= 0, topP.isFinite, topP > 0, topP <= 1,
              (1...4_097).contains(topK), (1...128).contains(streamingChunkFrames) else {
            throw FishAudioError.invalidInput("Invalid Drama 3 sampling or streaming options")
        }
    }
}

struct Drama3Sampler {
    var state: UInt64
    mutating func uniform() -> Double {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return Double((z ^ (z >> 31)) >> 11) / 9_007_199_254_740_992
    }

    mutating func sample(_ logits: [Float], temperature: Float, topK: Int, topP: Float,
                         suppressed: Int? = nil) throws -> Int {
        guard !logits.isEmpty, logits.allSatisfy(\.isFinite) else {
            throw FishAudioError.invalidInput("Drama 3 produced non-finite logits")
        }
        let sorted = logits.indices.filter { $0 != suppressed }.sorted {
            logits[$0] == logits[$1] ? $0 < $1 : logits[$0] > logits[$1]
        }
        guard let first = sorted.first else { throw FishAudioError.invalidInput("No valid Drama 3 tokens") }
        if temperature == 0 || topK == 1 { return first }
        // Upstream computes top-p over the full distribution, then drops the crossing token.
        let probabilities = sorted.map { exp(Double(($0 == first ? 0 : logits[$0] - logits[first])) / Double(max(temperature, 1e-5))) }
        let total = probabilities.reduce(0, +)
        var cumulative = 0.0, kept: [Int] = []
        for (rank, index) in sorted.enumerated() {
            cumulative += probabilities[rank] / total
            if rank > 0 && (rank >= topK || cumulative > Double(topP)) { break }
            kept.append(index)
        }
        let masses = kept.map { exp(Double(logits[$0] - logits[first]) / Double(max(temperature, 1e-5))) }
        let draw = uniform() * masses.reduce(0, +)
        var mass = 0.0
        for (index, probability) in zip(kept, masses) {
            mass += probability
            if draw < mass { return index }
        }
        return kept.last!
    }
}

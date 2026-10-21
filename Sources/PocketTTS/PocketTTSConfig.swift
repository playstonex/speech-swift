import Foundation

public enum PocketTTSError: Error, LocalizedError {
    case invalidInput(String)
    case invalidConfiguration(String)
    case processingFailed(String)
    case generationLimit
    public var errorDescription: String? {
        switch self {
        case .invalidInput(let reason), .invalidConfiguration(let reason), .processingFailed(let reason): return "Pocket TTS: \(reason)"
        case .generationLimit: return "Pocket TTS reached its generation/cache limit before completing speech"
        }
    }
}

enum PocketTtsConstants {
    static let kvCacheMaxLen = 512
    static let transformerDim = 1024
    static let samplesPerFrame = 1920
}

/// English, six-layer, fixed-Alba CoreML bundle contract (version 1).
public struct PocketTTSConfig: Codable, Sendable {
    public let modelType: String
    public let formatVersion: Int
    public let sampleRate: Int
    public let samplesPerFrame: Int
    public let language: String
    public let voice: String
    public let layers: Int
    public let cacheCapacity: Int
    public let voicePosition: Int
    public let flowSteps: Int
    public let mimiStaticInputs: [String]?
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", formatVersion = "format_version", sampleRate = "sample_rate"
        case samplesPerFrame = "samples_per_frame", language, voice, layers
        case cacheCapacity = "cache_capacity", voicePosition = "voice_position", flowSteps = "flow_steps"
        case mimiStaticInputs = "mimi_static_inputs"
    }
    public func validate() throws {
        let staticInputs = mimiStaticInputs ?? []
        let allowed = Set(["conv0_first", "conv_final_first"]
            + (0..<3).flatMap { i in ["res\(i)_conv0_first", "res\(i)_conv1_first", "res\(i)_conv1_prev"] })
        guard Set(staticInputs).count == staticInputs.count, Set(staticInputs).isSubset(of: allowed) else {
            throw PocketTTSError.invalidConfiguration("unsupported Mimi passthrough inputs")
        }
        guard modelType == "pocket_tts", formatVersion == 1, sampleRate == 24000,
              samplesPerFrame == 1920, language == "en", voice == "alba", layers == 6,
              cacheCapacity == 512, voicePosition > 0, voicePosition < 256, flowSteps == 8 else {
            throw PocketTTSError.invalidConfiguration("unsupported bundle contract")
        }
    }
}

public struct PocketTTSOptions: Sendable {
    public var seed: UInt64
    public var temperature: Float
    /// Per-text-chunk limit; reaching it without EOS is an error.
    public var maxFrames: Int
    public init(seed: UInt64 = 42, temperature: Float = 0.7, maxFrames: Int = 350) {
        self.seed = seed; self.temperature = temperature; self.maxFrames = maxFrames
    }
    public func validate() throws {
        guard temperature.isFinite, temperature >= 0, temperature <= 2,
              maxFrames > 0, maxFrames <= 500 else {
            throw PocketTTSError.invalidInput("temperature must be 0...2 and maxFrames 1...500")
        }
    }
}

/// Request-local deterministic noise. Zero is a valid seed.
struct PocketRNG {
    var state: UInt64
    mutating func uniform() -> Float {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        z ^= z >> 31
        return Float(Double(z >> 11) + 1) / Float(9_007_199_254_740_993.0)
    }
    mutating func gaussian() -> Float {
        let u = uniform(), v = uniform()
        return sqrt(-2 * log(u)) * cos(2 * .pi * v)
    }
}

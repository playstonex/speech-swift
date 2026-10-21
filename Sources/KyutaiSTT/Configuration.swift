import Foundation
import PersonaPlex

public enum KyutaiSTTVariant: String, CaseIterable, Sendable {
    case fp16, int4, int8

    public var modelId: String {
        switch self {
        case .fp16: return "aufklarer/Kyutai-STT-1B-MLX-FP16"
        case .int4: return "aufklarer/Kyutai-STT-1B-MLX-4bit"
        case .int8: return "aufklarer/Kyutai-STT-1B-MLX-8bit"
        }
    }
}

public enum KyutaiSTTError: Error, LocalizedError {
    case incompatibleCheckpoint(String)
    case invalidAudio(String)
    case sessionActive
    case sessionFinished

    public var errorDescription: String? {
        switch self {
        case .incompatibleCheckpoint(let reason): return "Incompatible Kyutai STT checkpoint: \(reason)"
        case .invalidAudio(let reason): return "Invalid Kyutai STT audio: \(reason)"
        case .sessionActive: return "Finish or cancel the active Kyutai STT session first."
        case .sessionFinished: return "This Kyutai STT session has finished. Create a new session."
        }
    }
}

/// Configuration for the original Kyutai STT checkpoints and native MLX exports.
public struct KyutaiSTTConfig: Decodable, Sendable {
    public struct Quantization: Decodable, Sendable {
        public let bits: Int
        public let groupSize: Int
        public let mode: String
        enum CodingKeys: String, CodingKey { case bits, groupSize = "group_size", mode }
    }

    public struct STT: Decodable, Sendable {
        public let audioDelaySeconds: Double
        public let audioSilencePrefixSeconds: Double
        enum CodingKeys: String, CodingKey {
            case audioDelaySeconds = "audio_delay_seconds"
            case audioSilencePrefixSeconds = "audio_silence_prefix_seconds"
        }
    }

    public let dim: Int
    public let numHeads: Int
    public let numLayers: Int
    public let textCard: Int
    public let card: Int
    public let numCodebooks: Int
    public let context: Int
    public let maxPeriod: Double
    public let stt: STT
    public let mimiName: String
    public let tokenizerName: String
    public let quantization: Quantization?
    public let weightLayout: String

    public static let sampleRate = 24_000
    public static let frameRate = 12.5
    public static let frameSize = 1_920
    public var prefixFrames: Int { Int(ceil(stt.audioSilencePrefixSeconds * Self.frameRate)) }
    /// Extra silence beyond the trained delay lets the final word complete.
    public var flushFrames: Int { Int(ceil(stt.audioDelaySeconds * Self.frameRate)) + 13 }
    public var intermediateSize: Int { dim * 11 / 4 }

    enum CodingKeys: String, CodingKey {
        case dim, card, context, quantization, delays, causal, norm, gating, conditioners
        case numHeads = "num_heads", numLayers = "num_layers", textCard = "text_card"
        case numCodebooks = "n_q", maxPeriod = "max_period", stt = "stt_config"
        case mimiName = "mimi_name", tokenizerName = "tokenizer_name"
        case weightLayout = "weight_layout", modelType = "model_type", depQ = "dep_q"
        case hiddenScale = "hidden_scale", positionalEmbedding = "positional_embedding"
        case layerScale = "layer_scale", crossAttention = "cross_attention"
        case extraHeads = "extra_heads_num_heads", textPaddingId = "existing_text_padding_id"
        case sampleRate = "sample_rate", frameRate = "frame_rate", formatVersion = "format_version"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dim = try c.decode(Int.self, forKey: .dim)
        numHeads = try c.decode(Int.self, forKey: .numHeads)
        numLayers = try c.decode(Int.self, forKey: .numLayers)
        textCard = try c.decode(Int.self, forKey: .textCard)
        card = try c.decode(Int.self, forKey: .card)
        numCodebooks = try c.decode(Int.self, forKey: .numCodebooks)
        context = try c.decode(Int.self, forKey: .context)
        maxPeriod = try c.decode(Double.self, forKey: .maxPeriod)
        stt = try c.decode(STT.self, forKey: .stt)
        mimiName = try c.decode(String.self, forKey: .mimiName)
        tokenizerName = try c.decode(String.self, forKey: .tokenizerName)
        quantization = try c.decodeIfPresent(Quantization.self, forKey: .quantization)
        weightLayout = try c.decodeIfPresent(String.self, forKey: .weightLayout) ?? "pytorch"

        func require(_ condition: Bool, _ reason: String) throws {
            if !condition { throw KyutaiSTTError.incompatibleCheckpoint(reason) }
        }
        try require(dim > 0 && numHeads > 0 && numLayers > 0 && textCard > 3 && context > 0,
                    "dimensions, vocabulary, and context must be positive")
        try require(dim % numHeads == 0 && (dim / numHeads) % 2 == 0 && dim % 8 == 0,
                    "invalid attention or feedforward dimensions")
        try require(maxPeriod.isFinite && maxPeriod > 0, "invalid RoPE period")
        try require(try c.decode(String.self, forKey: .modelType) == "stt", "model_type must be stt")
        try require(try c.decode(Int.self, forKey: .depQ) == 0, "STT must not generate audio codebooks")
        try require(numCodebooks == 32 && card == 2048, "expected the 32-codebook Mimi tokenizer")
        try require(try c.decode([Int].self, forKey: .delays) == Array(repeating: 0, count: 33),
                    "inter-codebook delays must all be zero")
        try require(try c.decode(Double.self, forKey: .hiddenScale) == 4.125, "unsupported hidden_scale")
        try require(try c.decode(String.self, forKey: .positionalEmbedding) == "rope", "expected interleaved RoPE")
        try require(try c.decode(String.self, forKey: .norm) == "rms_norm_f32", "unsupported normalization")
        try require(try c.decode(String.self, forKey: .gating) == "silu", "unsupported gating")
        try require(try c.decode(Bool.self, forKey: .causal), "expected causal attention")
        try require(!c.contains(.layerScale) || (try c.decodeNil(forKey: .layerScale)), "layer scale is unsupported")
        try require(!(try c.decodeIfPresent(Bool.self, forKey: .crossAttention) ?? false), "cross attention is unsupported")
        if c.contains(.conditioners) {
            let conditioners = try c.nestedContainer(keyedBy: DynamicKey.self, forKey: .conditioners)
            try require(conditioners.allKeys.isEmpty, "conditioners are unsupported")
        }
        try require(try c.decodeIfPresent(Int.self, forKey: .extraHeads) ?? 0 == 0, "extra heads are unsupported")
        try require(try c.decodeIfPresent(Int.self, forKey: .textPaddingId) ?? 3 == 3, "expected text padding id 3")
        try require(try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? Self.sampleRate == Self.sampleRate,
                    "expected 24 kHz audio")
        try require(try c.decodeIfPresent(Double.self, forKey: .frameRate) ?? Self.frameRate == Self.frameRate,
                    "expected 12.5 Hz Mimi frames")
        try require(["pytorch", "speech_swift_v1"].contains(weightLayout), "unsupported tensor layout")
        if weightLayout == "speech_swift_v1" {
            try require(try c.decode(Int.self, forKey: .formatVersion) == 1, "unsupported export version")
        }
        for seconds in [stt.audioDelaySeconds, stt.audioSilencePrefixSeconds] {
            try require(seconds.isFinite && (0...30).contains(seconds), "invalid STT timing")
        }
        for name in [mimiName, tokenizerName] {
            try require(!name.isEmpty && name != "." && name != ".." && !name.contains("/") && !name.contains("\\"),
                        "artifact names must be plain filenames")
        }
        if let q = quantization {
            try require([4, 8].contains(q.bits) && q.groupSize == 64 && q.mode == "affine",
                        "supported quantization: affine INT4 or INT8, group size 64")
            try require(dim % q.groupSize == 0 && intermediateSize % q.groupSize == 0,
                        "quantization groups do not fit model dimensions")
        }
    }

    var transformerConfig: TemporalTransformerConfig {
        TemporalTransformerConfig(
            dim: dim, numLayers: numLayers, numHeads: numHeads, hiddenScale: 4.125,
            card: card, textCard: textCard, context: context, maxPeriod: Int(maxPeriod),
            groupSize: quantization?.groupSize ?? 64, bits: quantization?.bits ?? 16)
    }
}

private struct DynamicKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

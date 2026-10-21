import Foundation

public enum VoxtralTTSError: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidInput(String)
    case invalidCheckpoint(String)
    case unknownVoice(String)
    case generationLimit(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message): return "Invalid Voxtral TTS configuration: \(message)"
        case .invalidInput(let message): return "Invalid Voxtral TTS input: \(message)"
        case .invalidCheckpoint(let message): return "Invalid Voxtral TTS checkpoint: \(message)"
        case .unknownVoice(let voice): return "Unknown Voxtral TTS voice: \(voice)"
        case .generationLimit(let frames): return "Voxtral TTS reached the \(frames)-frame limit before end of speech"
        }
    }
}

public struct VoxtralTTSQuantization: Decodable, Sendable {
    public let bits: Int
    public let groupSize: Int
    public let mode: String

    enum CodingKeys: String, CodingKey {
        case bits, mode
        case groupSize = "group_size"
    }
}

struct VoxtralTransformerConfig: Decodable, Sendable {
    let dim: Int
    let layers: Int
    let headDim: Int
    let hiddenDim: Int
    let heads: Int
    let kvHeads: Int
    let ropeTheta: Float
    let normEps: Float?

    enum CodingKeys: String, CodingKey {
        case dim
        case layers = "n_layers"
        case headDim = "head_dim"
        case hiddenDim = "hidden_dim"
        case heads = "n_heads"
        case kvHeads = "n_kv_heads"
        case ropeTheta = "rope_theta"
        case normEps = "norm_eps"
    }

    func validate() throws {
        guard (1...16_384).contains(dim), (1...128).contains(layers), (1...1024).contains(headDim), headDim % 2 == 0,
              (1...65_536).contains(hiddenDim), (1...256).contains(heads), kvHeads > 0, heads % kvHeads == 0,
              ropeTheta.isFinite, ropeTheta > 0 else {
            throw VoxtralTTSError.invalidConfiguration("invalid transformer dimensions")
        }
    }
}

struct VoxtralAudioEncoding: Decodable, Sendable {
    let sampleRate: Int
    let frameRate: Double
    let codebooks: Int

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sampling_rate"
        case frameRate = "frame_rate"
        case codebooks = "num_codebooks"
    }
}

struct VoxtralAudioModelConfig: Decodable, Sendable {
    let semanticSize: Int
    let acousticSize: Int
    let acousticCodebooks: Int
    let encoding: VoxtralAudioEncoding
    let audioToken: Int
    let beginAudioToken: Int
    let transformer: VoxtralTransformerConfig

    enum CodingKeys: String, CodingKey {
        case semanticSize = "semantic_codebook_size"
        case acousticSize = "acoustic_codebook_size"
        case acousticCodebooks = "n_acoustic_codebook"
        case encoding = "audio_encoding_args"
        case audioToken = "audio_token_id"
        case beginAudioToken = "begin_audio_token_id"
        case transformer = "acoustic_transformer_args"
    }

    var embeddingCount: Int {
        (semanticSize / 128 + 1) * 128 + ((acousticSize * acousticCodebooks + 127) / 128) * 128
    }
    var offsets: [Int32] {
        let firstAcoustic = semanticSize + 2
        let codebookWidth = acousticSize + 2
        var result: [Int32] = [0]
        for index in 0..<acousticCodebooks {
            result.append(Int32(firstAcoustic + index * codebookWidth))
        }
        return result
    }
}

struct VoxtralCodecConfig: Decodable, Sendable {
    let channels: Int
    let sampleRate: Int
    let patchSize: Int
    let patchKernel: Int
    let semanticSize: Int
    let semanticDim: Int
    let acousticSize: Int
    let acousticDim: Int
    let dim: Int
    let hiddenDim: Int
    let headDim: Int
    let heads: Int
    let kvHeads: Int
    let normEps: Float
    let qkNormEps: Float
    let layerLengths: String
    let convKernels: String
    let convStrides: String

    enum CodingKeys: String, CodingKey {
        case channels, dim
        case sampleRate = "sampling_rate"
        case patchSize = "pretransform_patch_size"
        case patchKernel = "patch_proj_kernel_size"
        case semanticSize = "semantic_codebook_size"
        case semanticDim = "semantic_dim"
        case acousticSize = "acoustic_codebook_size"
        case acousticDim = "acoustic_dim"
        case hiddenDim = "hidden_dim"
        case headDim = "head_dim"
        case heads = "n_heads"
        case kvHeads = "n_kv_heads"
        case normEps = "norm_eps"
        case qkNormEps = "qk_norm_eps"
        case layerLengths = "decoder_transformer_lengths_str"
        case convKernels = "decoder_convs_kernels_str"
        case convStrides = "decoder_convs_strides_str"
    }

    var lengths: [Int] { layerLengths.split(separator: ",").compactMap { Int($0) } }
    var kernels: [Int] { convKernels.split(separator: ",").compactMap { Int($0) } }
    var strides: [Int] { convStrides.split(separator: ",").compactMap { Int($0) } }
    var samplesPerFrame: Int { patchSize * strides.reduce(1, *) }
}

private struct VoxtralMultimodalConfig: Decodable {
    let audio: VoxtralAudioModelConfig
    let codec: VoxtralCodecConfig

    enum CodingKeys: String, CodingKey {
        case audio = "audio_model_args"
        case codec = "audio_tokenizer_args"
    }
}

public struct VoxtralTTSConfig: Decodable, Sendable {
    public let modelType: String
    public let quantization: VoxtralTTSQuantization?
    public let sampleRate: Int
    public let frameRate: Double
    public let maximumContext: Int
    let transformer: VoxtralTransformerConfig
    let audio: VoxtralAudioModelConfig
    let codec: VoxtralCodecConfig
    let vocabSize: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantization, multimodal
        case vocabSize = "vocab_size"
        case maximumContext = "max_seq_len"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        quantization = try c.decodeIfPresent(VoxtralTTSQuantization.self, forKey: .quantization)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        maximumContext = try c.decodeIfPresent(Int.self, forKey: .maximumContext) ?? 65_536
        transformer = try VoxtralTransformerConfig(from: decoder)
        let mm = try c.decode(VoxtralMultimodalConfig.self, forKey: .multimodal)
        audio = mm.audio
        codec = mm.codec
        sampleRate = audio.encoding.sampleRate
        frameRate = audio.encoding.frameRate
        try validate()
    }

    private func validate() throws {
        try transformer.validate()
        try audio.transformer.validate()
        guard modelType == "voxtral_tts", vocabSize > 0, maximumContext > 0,
              transformer.normEps?.isFinite == true, (transformer.normEps ?? 0) > 0,
              sampleRate == 24_000, frameRate == 12.5,
              (2...1_000_000).contains(audio.semanticSize), (2...1_000_000).contains(audio.acousticSize),
              (1...256).contains(audio.acousticCodebooks),
              audio.transformer.dim == transformer.dim, audio.transformer.dim % 2 == 0,
              audio.encoding.codebooks == audio.acousticCodebooks + 1,
              audio.audioToken >= 0, audio.audioToken < vocabSize,
              audio.beginAudioToken >= 0, audio.beginAudioToken < vocabSize,
              codec.channels == 1, codec.sampleRate == sampleRate,
              codec.semanticSize == audio.semanticSize, codec.acousticSize == audio.acousticSize,
              codec.acousticDim == audio.acousticCodebooks,
              (1...16_384).contains(codec.dim), (1...65_536).contains(codec.hiddenDim),
              (1...16_384).contains(codec.semanticDim), (1...256).contains(codec.heads),
              codec.kvHeads > 0, (1...1024).contains(codec.headDim),
              codec.heads == codec.kvHeads,
              codec.normEps.isFinite, codec.normEps > 0, codec.qkNormEps.isFinite, codec.qkNormEps > 0,
              codec.lengths.count == 4, codec.kernels.count == 4, codec.strides == [1, 2, 2, 2],
              codec.lengths.allSatisfy({ $0 > 0 }), codec.kernels.allSatisfy({ $0 >= 2 }),
              codec.patchSize == 240, codec.patchKernel > 0,
              audio.embeddingCount >= audio.semanticSize + 2 + audio.acousticCodebooks * (audio.acousticSize + 2)
        else { throw VoxtralTTSError.invalidConfiguration("unsupported model or audio dimensions") }
        if let q = quantization {
            guard [2, 3, 4, 5, 6, 8].contains(q.bits), [32, 64, 128].contains(q.groupSize), q.mode == "affine" else {
                throw VoxtralTTSError.invalidConfiguration("unsupported MLX quantization")
            }
        }
    }
}

public struct VoxtralTTSOptions: Sendable {
    public var maxFrames: Int
    public var denoisingSteps: Int
    public var guidance: Float
    public var seed: UInt64
    /// Codec frames per PCM chunk; each frame is 80 ms of audio.
    public var streamingChunkFrames: Int

    public init(maxFrames: Int = 500, denoisingSteps: Int = 8, guidance: Float = 1.2,
                seed: UInt64 = 0, streamingChunkFrames: Int = 3) {
        self.maxFrames = maxFrames
        self.denoisingSteps = denoisingSteps
        self.guidance = guidance
        self.seed = seed
        self.streamingChunkFrames = streamingChunkFrames
    }

    public func validate() throws {
        guard maxFrames > 0, maxFrames <= 4096, (2...64).contains(denoisingSteps),
              guidance.isFinite, (0...10).contains(guidance), (1...125).contains(streamingChunkFrames) else {
            throw VoxtralTTSError.invalidInput("maxFrames must be 1...4096, denoisingSteps 2...64, guidance 0...10, and streamingChunkFrames 1...125")
        }
    }
}

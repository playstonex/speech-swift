import Foundation

public enum BreezeTTSError: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidInput(String)
    case missingFile(String)
    case frameLimit(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message), .invalidInput(let message): return message
        case .missingFile(let name): return "Missing Breeze bundle file: \(name)"
        case .frameLimit(let count): return "Breeze reached \(count) frames before EOS; increase maxFrames."
        }
    }
}

struct BreezeQuantization: Codable, Sendable {
    let bits: Int
    let groupSize: Int
    enum CodingKeys: String, CodingKey { case bits; case groupSize = "group_size" }
}

struct BreezeRoPEScaling: Codable, Sendable {
    var ropeType: String = "default"
    var ropeTheta: Float?
    var factor: Float?
    var lowFreqFactor: Float?
    var highFreqFactor: Float?
    var originalMaxPositionEmbeddings: Int?
    enum CodingKeys: String, CodingKey {
        case ropeType = "rope_type", ropeTheta = "rope_theta", factor
        case lowFreqFactor = "low_freq_factor", highFreqFactor = "high_freq_factor"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }
}

public struct BreezeTTSOptions: Sendable, Equatable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var cfgScale: Float
    public var maxFrames: Int
    public var seed: UInt64
    public var firstChunkFrames: Int
    public var chunkFrames: Int
    public var decoderLeftContext: Int

    public init(temperature: Float = 0.9, topK: Int = 50, topP: Float = 1,
                cfgScale: Float = 4, maxFrames: Int = 750, seed: UInt64 = 42,
                firstChunkFrames: Int = 3, chunkFrames: Int = 25, decoderLeftContext: Int = 25) {
        self.temperature = temperature; self.topK = topK; self.topP = topP
        self.cfgScale = cfgScale; self.maxFrames = maxFrames; self.seed = seed
        self.firstChunkFrames = firstChunkFrames; self.chunkFrames = chunkFrames
        self.decoderLeftContext = decoderLeftContext
    }

    public func validate() throws {
        guard temperature.isFinite, temperature >= 0, topK >= 0,
              topP.isFinite, topP > 0, topP <= 1, cfgScale.isFinite, cfgScale >= 1,
              maxFrames > 0, firstChunkFrames > 0, chunkFrames > 0, decoderLeftContext >= 0 else {
            throw BreezeTTSError.invalidInput("Invalid Breeze sampling or streaming options")
        }
    }
}

extension BreezeTTSConfig {
    public var outputSampleRate: Int { sampleRate }

    public func validate() throws {
        guard modelType == "breeze", numCodebooks == 16, sampleRate == 24_000,
              codecVocabSize == 2048, audioVocabSize == 2051, codebookEOSTokenID == 0,
              tieCodebooksEmbeddings,
              audioEmbedSize > 0, backboneConfig.modelType == "qwen3",
              backboneConfig.hiddenSize == depthDecoderConfig.backboneHiddenSize,
              depthDecoderConfig.numCodebooks == numCodebooks,
              depthDecoderConfig.vocabSize == audioVocabSize,
              depthDecoderConfig.audioEmbedSize == audioEmbedSize,
              backboneConfig.numHiddenLayers > 0, depthDecoderConfig.numHiddenLayers > 0,
              textEncoderConfig.numHiddenLayers > 0,
              textEncoderConfig.layerTypes.count == textEncoderConfig.numHiddenLayers,
              textEncoderConfig.layerTypes.allSatisfy({ ["full_attention", "sliding_attention"].contains($0) }),
              textEncoderConfig.slidingWindow > 0 else {
            throw BreezeTTSError.invalidConfiguration("Unsupported Breeze architecture or codec configuration")
        }
        guard [backboneConfig.rmsNormEps, depthDecoderConfig.rmsNormEps, textEncoderConfig.rmsNormEps,
               backboneConfig.ropeTheta, depthDecoderConfig.ropeTheta,
               textEncoderConfig.queryPreAttentionScalar].allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw BreezeTTSError.invalidConfiguration("Invalid Breeze normalization or attention scale")
        }
        for scaling in [backboneConfig.ropeScaling, depthDecoderConfig.ropeScaling].compactMap({ $0 }) {
            guard ["default", "llama3"].contains(scaling.ropeType) else {
                throw BreezeTTSError.invalidConfiguration("Unsupported Breeze rotary scaling")
            }
            if scaling.ropeType == "llama3" {
                guard let factor = scaling.factor, factor.isFinite, factor >= 1,
                      let low = scaling.lowFreqFactor, let high = scaling.highFreqFactor,
                      low.isFinite, high.isFinite, low > 0, high > low,
                      let length = scaling.originalMaxPositionEmbeddings, length > 0 else {
                    throw BreezeTTSError.invalidConfiguration("Invalid Llama 3 rotary scaling")
                }
            }
        }
        for scaling in textEncoderConfig.ropeParameters.values {
            guard ["default", "linear"].contains(scaling.ropeType),
                  scaling.ropeTheta?.isFinite != false, (scaling.ropeTheta ?? 1) > 0,
                  scaling.factor?.isFinite != false, (scaling.factor ?? 1) > 0 else {
                throw BreezeTTSError.invalidConfiguration("Unsupported text encoder rotary scaling")
            }
        }
        for (q, kv, dim, intermediate, head) in [
            (backboneConfig.numAttentionHeads, backboneConfig.numKeyValueHeads, backboneConfig.hiddenSize,
             backboneConfig.intermediateSize, backboneConfig.headDim),
            (depthDecoderConfig.numAttentionHeads, depthDecoderConfig.numKeyValueHeads, depthDecoderConfig.hiddenSize,
             depthDecoderConfig.intermediateSize, depthDecoderConfig.headDim),
            (textEncoderConfig.numAttentionHeads, textEncoderConfig.numKeyValueHeads, textEncoderConfig.hiddenSize,
             textEncoderConfig.intermediateSize, textEncoderConfig.headDim)
        ] {
            guard q > 0, kv > 0, q.isMultiple(of: kv), dim > 0, intermediate > 0, head > 0,
                  head.isMultiple(of: 2) else {
                throw BreezeTTSError.invalidConfiguration("Invalid Breeze attention dimensions")
            }
        }
        if let quantization {
            guard [4, 8].contains(quantization.bits), quantization.groupSize == 64 else {
                throw BreezeTTSError.invalidConfiguration("Breeze supports affine 4/8-bit quantization with group size 64")
            }
        }
    }
}

// Adapted from MOSS-VoiceGenerator and MOSS-Audio-Tokenizer.
// Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
// Licensed under Apache 2.0; see LICENSE.
// Native Swift adaptation uses request-local state and bounded causal caches.

import Foundation

public enum MossVoiceGeneratorError: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidInput(String)
    case invalidOutput(String)
    case contextExceeded
    case generationLimit
    case unloaded

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let s): "Invalid MOSS-VoiceGenerator bundle: \(s)"
        case .invalidInput(let s): "Invalid MOSS-VoiceGenerator input: \(s)"
        case .invalidOutput(let s): "Invalid MOSS-VoiceGenerator output: \(s)"
        case .contextExceeded: "The prompt and requested audio exceed the model context."
        case .generationLimit: "MOSS-VoiceGenerator reached its limit before end-of-speech."
        case .unloaded: "MOSS-VoiceGenerator has been unloaded. Create a new model instance to generate speech."
        }
    }
}

public struct MossVoiceGeneratorConfiguration: Decodable, Sendable {
    public struct Language: Decodable, Sendable {
        public let hiddenSize, intermediateSize, numHiddenLayers: Int
        public let numAttentionHeads, numKeyValueHeads, headDim, vocabSize: Int
        public let maxPositionEmbeddings: Int
        public let rmsNormEps, ropeTheta: Float
        enum CodingKeys: String, CodingKey {
            case hiddenSize="hidden_size", intermediateSize="intermediate_size", numHiddenLayers="num_hidden_layers"
            case numAttentionHeads="num_attention_heads", numKeyValueHeads="num_key_value_heads", headDim="head_dim", vocabSize="vocab_size"
            case maxPositionEmbeddings="max_position_embeddings", rmsNormEps="rms_norm_eps", ropeTheta="rope_theta"
        }
    }
    public let modelType: String
    public let language: Language
    public let codebooks, audioVocabularySize, audioPadCode, sampleRate: Int
    public let audioStart, audioEnd, audioGeneration, audioDelay: Int
    public var imStart: Int { 151644 }
    public var imEnd: Int { 151645 }
    public var padToken: Int { 151643 }
    enum CodingKeys: String, CodingKey {
        case modelType="model_type", language="language_config", codebooks="n_vq"
        case audioVocabularySize="audio_vocab_size", audioPadCode="audio_pad_code", sampleRate="sampling_rate"
        case audioStart="audio_start_token_id", audioEnd="audio_end_token_id"
        case audioGeneration="audio_assistant_gen_slot_token_id", audioDelay="audio_assistant_delay_slot_token_id"
    }
    public func validate() throws {
        guard modelType=="moss_tts_delay", codebooks==16, audioVocabularySize==1024,
              audioPadCode==1024, sampleRate==24000,
              audioStart==151652,audioEnd==151653,audioGeneration==151656,audioDelay==151662,
              language.hiddenSize==2048,language.intermediateSize==6144,
              language.numHiddenLayers==28,language.numAttentionHeads==16,
              language.numKeyValueHeads==8,language.headDim==128,language.vocabSize==155648,
              language.maxPositionEmbeddings==40960,
              language.rmsNormEps.isFinite,language.rmsNormEps>0,
              language.ropeTheta.isFinite,language.ropeTheta>0 else {
            throw MossVoiceGeneratorError.invalidConfiguration("unsupported dimensions or checkpoint family")
        }
    }
}

public struct MossVoiceGeneratorOptions: Sendable {
    public var maxFrames: Int
    public var seed: UInt64
    public var textTemperature: Float
    public var audioTemperature: Float
    public var textTopP: Float
    public var audioTopP: Float
    public var topK: Int
    public var repetitionPenalty: Float
    public var chunkFrames: Int
    /// Use the upstream punctuation/metadata cleanup. Disable to preserve raw benchmark input.
    public var normalizeInputs: Bool

    public init(maxFrames: Int=750, seed: UInt64=42, textTemperature: Float=1.5,
                audioTemperature: Float=1.5, textTopP: Float=1, audioTopP: Float=0.6,
                topK: Int=50, repetitionPenalty: Float=1.1, chunkFrames: Int=3,
                normalizeInputs: Bool=false) {
        self.maxFrames=maxFrames;self.seed=seed;self.textTemperature=textTemperature
        self.audioTemperature=audioTemperature;self.textTopP=textTopP;self.audioTopP=audioTopP
        self.topK=topK;self.repetitionPenalty=repetitionPenalty;self.chunkFrames=chunkFrames
        self.normalizeInputs=normalizeInputs
    }
    public func validate() throws {
        guard (1...4096).contains(maxFrames),(1...32).contains(chunkFrames),
              textTemperature.isFinite,textTemperature>=0,audioTemperature.isFinite,audioTemperature>=0,
              textTopP.isFinite,textTopP>0,textTopP<=1,audioTopP.isFinite,audioTopP>0,audioTopP<=1,
              (0...155648).contains(topK),repetitionPenalty.isFinite,repetitionPenalty>0 else {
            throw MossVoiceGeneratorError.invalidInput("invalid sampling, frame or chunk settings")
        }
    }
}

public struct MossVoiceGeneratorResult: Sendable {
    public let samples: [Float]
    /// Undelayed audio codebooks in frame-major order.
    public let codes: [[Int32]]
    public let sampleRate: Int
    public let generationSeconds: Double
    public let firstTokenSeconds: Double
    public let firstAudioSeconds: Double
    public var duration: Double { Double(samples.count)/Double(sampleRate) }
}

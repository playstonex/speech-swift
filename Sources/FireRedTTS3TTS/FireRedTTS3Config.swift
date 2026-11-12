import Foundation

public enum FireRedTTS3Error: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidInput(String)
    case missingFile(String)
    case invalidWeights(String)
    case modelUnloaded
    case alreadyGenerating
    case planningLimit
    case audioLimit
    case invalidAudio

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let s): "Invalid FireRedTTS3 bundle: \(s)"
        case .invalidInput(let s): "Invalid FireRedTTS3 input: \(s)"
        case .missingFile(let s): "Missing FireRedTTS3 file: \(s)"
        case .invalidWeights(let s): "Invalid FireRedTTS3 tensor: \(s)"
        case .modelUnloaded: "FireRedTTS3 model has been unloaded"
        case .alreadyGenerating: "This FireRedTTS3 instance is already generating"
        case .planningLimit: "FireRedTTS3 voice planning reached its token limit"
        case .audioLimit: "FireRedTTS3 generation reached its audio limit before stopping"
        case .invalidAudio: "FireRedTTS3 generated empty or non-finite audio"
        }
    }
}

public struct FireRedTTS3QwenConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let vocabSize: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let slidingWindow: Int
}

public struct FireRedTTS3TransformerConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
}

public struct FireRedTTS3Config: Decodable, Sendable {
    public let modelType: String
    public let formatVersion: Int
    public let storageDtype: String
    public let sampleRate: Int
    public let latentDimension: Int
    public let patchSize: Int
    public let historyPatches: Int
    public let backbone: FireRedTTS3QwenConfig
    public let patchEncoder: FireRedTTS3TransformerConfig
    public let flow: FireRedTTS3TransformerConfig
    public let decoder: FireRedTTS3QwenConfig
    public let decoderUpsampleRate: Int
    public let audioPatchSize: Int
    public let latentScale: Float

    public static func load(from url: URL) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Self.self, from: Data(contentsOf: url))
        try config.validate()
        return config
    }

    public func validate() throws {
        guard modelType == "fireredtts3_instruct", formatVersion == 1,
              storageDtype == "float32", sampleRate == 24_000,
              latentDimension == 64, patchSize == 4, historyPatches == 2,
              decoderUpsampleRate == 2, audioPatchSize == 480,
              latentScale.isFinite, abs(latentScale - 0.4) < 1e-7 else {
            throw FireRedTTS3Error.invalidConfiguration("expected the Float32 Instruct voice-design export")
        }
        for q in [backbone, decoder] {
            guard q.hiddenSize > 0, q.hiddenSize <= 8192,
                  q.intermediateSize > 0, q.intermediateSize <= 32768,
                  q.numHiddenLayers > 0, q.numHiddenLayers <= 80,
                  (1...128).contains(q.numAttentionHeads), (1...128).contains(q.numKeyValueHeads),
                  q.numAttentionHeads % q.numKeyValueHeads == 0,
                  q.headDim > 0, q.headDim % 2 == 0,
                  q.headDim <= 256, (151678...1_000_000).contains(q.vocabSize),
                  q.ropeTheta.isFinite, q.ropeTheta > 1,
                  q.rmsNormEps.isFinite, q.rmsNormEps > 0,
                  (0...32768).contains(q.slidingWindow) else {
                throw FireRedTTS3Error.invalidConfiguration("invalid Qwen3 architecture")
            }
        }
        for t in [patchEncoder, flow] {
            guard t.hiddenSize > 0, t.hiddenSize <= 4096,
                  t.intermediateSize > 0, t.intermediateSize <= 32768,
                  t.numHiddenLayers > 0, t.numHiddenLayers <= 64,
                  t.numAttentionHeads > 0, t.hiddenSize % t.numAttentionHeads == 0,
                  (t.hiddenSize / t.numAttentionHeads) % 2 == 0 else {
                throw FireRedTTS3Error.invalidConfiguration("invalid patch/flow architecture")
            }
        }
    }
}

public struct FireRedTTS3Options: Sendable {
    public var seed: UInt64 = 2
    public var flowSteps: Int = 10
    public var guidanceScale: Float = 1.2
    public var stopThreshold: Float = 0.5
    public var minAudioPatches: Int = 6
    public var maxAudioPatches: Int = 400
    public var maxPlanningTokens: Int = 200
    public var textTemperature: Float = 0.7
    public var textTopK: Int = 20
    public var textTopP: Float = 0.8

    public init() {}

    public func validate() throws {
        guard (1...100).contains(flowSteps), guidanceScale.isFinite,
              (0...10).contains(guidanceScale), stopThreshold.isFinite,
              stopThreshold > 0, stopThreshold < 1,
              minAudioPatches >= 0, maxAudioPatches > minAudioPatches,
              maxAudioPatches <= 2000, (1...1000).contains(maxPlanningTokens),
              textTemperature.isFinite, textTemperature >= 0, textTemperature <= 5,
              textTopK >= 0, textTopP.isFinite, textTopP > 0, textTopP <= 1 else {
            throw FireRedTTS3Error.invalidInput("invalid generation limits or sampling settings")
        }
    }
}

enum FireRedTTS3Prompt {
    static let textEndID = 151677
    static let audioStartID = 151669

    static func build(text: String, instruction: String) throws -> String {
        for value in [text, instruction] {
            guard value.utf8.prefix(32769).count <= 32768,
                  !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  !value.contains("<|"), !value.contains("|>"),
                  !["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"].contains(where: value.contains) else {
                throw FireRedTTS3Error.invalidInput("text and instruction must be nonempty and contain no structural tokens")
            }
        }
        // Exact upstream voice-design conversation; spoken text is not normalized.
        return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\n\(instruction)\n\n根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频：\n\(text) /no_think<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n<|sot|>"
    }
}

import Foundation
import AudioCommon
import Qwen3ASR
import SpeechRuntime

/// Backend identities shared by the CLI and audio server.
public enum Qwen3ASRDescriptors {
    public static let small = mlx(modelID: Qwen3ASRModel.defaultModelId,
                                  id: "qwen3-asr-0.6b-mlx-int4")
    public static let large = mlx(modelID: Qwen3ASRModel.largeModelId,
                                  id: "qwen3-asr-1.7b-mlx-int8")
    public static let coreML = ModelDescriptor(
        id: "qwen3-asr-coreml", modelID: Qwen3ASRModel.coreMLModelId,
        backend: .coreML, inputSampleRate: 16000, streaming: .buffered)

    public static func mlx(modelID: String, id: String? = nil) -> ModelDescriptor {
        ModelDescriptor(id: id ?? modelID, modelID: modelID, backend: .mlx,
                        inputSampleRate: 16000, streaming: .buffered)
    }

    public static func resolveModelID(_ specifier: String) -> String {
        switch specifier.lowercased() {
        case "0.6b", "small": return small.modelID
        case "0.6b-8bit", "small-8bit": return "aufklarer/Qwen3-ASR-0.6B-MLX-8bit"
        case "1.7b", "large": return large.modelID
        case "1.7b-4bit", "large-4bit": return "aufklarer/Qwen3-ASR-1.7B-MLX-4bit"
        default: return specifier
        }
    }
}

/// Optional serialized ownership for Qwen3-ASR. Direct model APIs remain
/// available for callers that already manage their own execution and lifetime.
public final class Qwen3ASRRuntime: @unchecked Sendable {
    private enum Engine {
        case mlx(Qwen3ASRModel)
        case coreML(CoreMLASRModel)
    }

    public let descriptor: ModelDescriptor
    private let runtime: ModelRuntime<Engine>

    public init(
        descriptor: ModelDescriptor = Qwen3ASRDescriptors.small,
        offlineMode: Bool = false,
        progressHandler: (@Sendable (Double, String) -> Void)? = nil
    ) {
        self.descriptor = descriptor
        runtime = ModelRuntime(unloader: { engine in
            if case .mlx(let model) = engine { model.unload() }
        }) {
            switch descriptor.backend {
            case .mlx:
                return .mlx(try await Qwen3ASRModel.fromPretrained(
                    modelId: descriptor.modelID, offlineMode: offlineMode,
                    progressHandler: progressHandler))
            case .coreML:
                return .coreML(try await CoreMLASRModel.fromPretrained(
                    encoderModelId: descriptor.modelID, decoderModelId: descriptor.modelID,
                    offlineMode: offlineMode, progressHandler: progressHandler))
            case .hybrid:
                throw AudioModelError.invalidConfiguration(
                    model: descriptor.id, reason: "Use the native hybrid encoder API")
            }
        }
    }

    public func transcribe(
        audio: [Float], sampleRate: Int = 16000, language: String? = nil,
        maxTokens: Int = 448, context: String? = nil
    ) async throws -> String {
        try Task.checkCancellation()
        guard sampleRate > 0, maxTokens > 0, audio.allSatisfy(\.isFinite) else {
            throw AudioModelError.invalidConfiguration(
                model: descriptor.id, reason: "Audio must be finite and sample rate/token limit must be positive")
        }
        if descriptor.backend == .coreML, context?.isEmpty == false {
            throw AudioModelError.invalidConfiguration(
                model: descriptor.id, reason: "Context hints require the MLX backend")
        }
        if audio.isEmpty { return "" }
        return try await runtime.withModel { engine in
            switch engine {
            case .mlx(let model):
                return try model.transcribeThrowing(
                    audio: audio, sampleRate: sampleRate,
                    options: Qwen3DecodingOptions(maxTokens: maxTokens, language: language, context: context))
            case .coreML(let model):
                // CoreML scratch buffers are autoreleased. Bound their lifetime
                // even when a caller runs many requests within one async task.
                return try autoreleasepool {
                    try model.transcribe(audio: audio, sampleRate: sampleRate,
                                         language: language, maxTokens: maxTokens)
                }
            }
        }
    }

    public func unload() async throws { try await runtime.unload() }

    public func prepare(warmUp: Bool = false) async throws {
        try await runtime.withModel { engine in
            if warmUp, case .coreML(let model) = engine { try model.warmUp() }
        }
    }
}

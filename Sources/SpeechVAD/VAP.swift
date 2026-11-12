#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

/// Two-speaker forecasts in channel order `[user, agent]`.
public struct VAPForecast: Sendable {
    /// Normalized near-term floor probabilities from the first two projection bins.
    public let pNow: [Float]
    /// Normalized later floor probabilities from the last two projection bins.
    public let pFuture: [Float]
    /// Current speech activity probabilities for the two channels.
    public let voiceActivity: [Float]
}

struct VAPConfiguration: Decodable {
    let modelType: String
    let sampleRate: Int
    let channels: Int
    let channelOrder: [String]
    let windowSamples: Int
    let frameHz: Int
    let compiledModel: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", sampleRate = "sample_rate", channels
        case channelOrder = "channel_order", windowSamples = "window_samples"
        case frameHz = "frame_hz", compiledModel = "compiled_model"
    }
}

/// Voice Activity Projection with the otoSpeech checkpoint.
/// Supply separate user-microphone and agent-playback channels. Each request
/// recomputes their latest 20 seconds; shorter windows receive leading silence.
/// The caller decides when these forecasts should trigger a response or interruption.
public final class VAPModel {
    public static let defaultModelId = "aufklarer/VAP-OTO-CoreML"
    public static let sampleRate = 16_000
    public static let windowSamples = 320_000
    static let compiledModel = "vap.mlmodelc"
    private let model: MLModel

    private init(model: MLModel) { self.model = model }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        downloadOptions: DownloadOptions = .init(),
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> VAPModel {
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId, to: directory,
            additionalFiles: ["\(compiledModel)/**", "config.json", "LICENSE", "NOTICE"],
            offlineMode: offlineMode, downloadOptions: downloadOptions,
            progressHandler: { progressHandler?($0, "Downloading VAP model...") })
        return try load(from: directory)
    }

    /// Load a compiled bundle without network access.
    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .all
    ) throws -> VAPModel {
        _ = try decodeConfiguration(Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = try CoreMLLoader.load(
            url: directory.appendingPathComponent(compiledModel),
            computeUnits: computeUnits, name: "vap-oto")
        let inputs = model.modelDescription.inputDescriptionsByName
        let outputs = model.modelDescription.outputDescriptionsByName
        guard Set(inputs.keys) == ["audio"],
            let input = inputs["audio"]?.multiArrayConstraint,
            input.shape.map(\.intValue) == [1, 2, windowSamples],
            input.dataType == .float32,
            Set(outputs.keys) == ["p_now", "p_future", "vad"]
        else { throw invalid("incompatible compiled VAP graph") }
        for description in outputs.values {
            guard let constraint = description.multiArrayConstraint,
                constraint.shape.map(\.intValue) == [1, 1_000, 2],
                constraint.dataType == .float32
            else { throw invalid("incompatible compiled VAP output") }
        }
        return VAPModel(model: model)
    }

    static func decodeConfiguration(_ data: Data) throws -> VAPConfiguration {
        let config = try JSONDecoder().decode(VAPConfiguration.self, from: data)
        guard config.modelType == "vap-oto-coreml", config.sampleRate == sampleRate,
            config.channels == 2, config.channelOrder == ["user", "agent"],
            config.windowSamples == windowSamples, config.frameHz == 50,
            config.compiledModel == compiledModel
        else { throw invalid("incompatible VAP model configuration") }
        return config
    }

    static func invalid(_ reason: String) -> AudioModelError {
        .invalidConfiguration(model: "VAP otoSpeech", reason: reason)
    }

    /// Returns the most recent forecast. The channels must be synchronized and
    /// contain the same number of samples. An inactive agent uses a zero channel.
    public func forecast(
        userAudio: [Float], agentAudio: [Float], sampleRate: Int = VAPModel.sampleRate
    ) throws -> VAPForecast {
        guard sampleRate > 0, userAudio.count == agentAudio.count, !userAudio.isEmpty else {
            throw Self.invalid("synchronized, nonempty channels and a positive sample rate are required")
        }
        let user: [Float], agent: [Float]
        if sampleRate == Self.sampleRate {
            user = userAudio
            agent = agentAudio
        } else {
            guard userAudio.allSatisfy(\.isFinite), agentAudio.allSatisfy(\.isFinite) else {
                throw Self.invalid("audio contains a non-finite sample")
            }
            user = AudioFileLoader.resample(userAudio, from: sampleRate, to: Self.sampleRate)
            agent = AudioFileLoader.resample(agentAudio, from: sampleRate, to: Self.sampleRate)
        }
        let window = try Self.preparedWindow(user: user, agent: agent)
        return try autoreleasepool {
            let input = try MLMultiArray(shape: [1, 2, Self.windowSamples as NSNumber], dataType: .float32)
            window.withUnsafeBufferPointer {
                input.dataPointer.assumingMemoryBound(to: Float.self).update(from: $0.baseAddress!, count: $0.count)
            }
            let provider = try MLDictionaryFeatureProvider(dictionary: ["audio": MLFeatureValue(multiArray: input)])
            let output = try model.prediction(from: provider)
            func lastFrame(_ name: String) throws -> [Float] {
                guard let values = output.featureValue(for: name)?.multiArrayValue,
                    values.shape.map(\.intValue) == [1, 1_000, 2]
                else { throw Self.invalid("VAP omitted output: \(name)") }
                let result = [values[1_998].floatValue, values[1_999].floatValue]
                guard result.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
                    throw Self.invalid("VAP returned invalid probabilities")
                }
                return result
            }
            return try VAPForecast(pNow: lastFrame("p_now"), pFuture: lastFrame("p_future"),
                                   voiceActivity: lastFrame("vad"))
        }
    }

    static func preparedWindow(user: [Float], agent: [Float]) throws -> [Float] {
        guard user.count == agent.count, !user.isEmpty,
            user.allSatisfy(\.isFinite), agent.allSatisfy(\.isFinite)
        else { throw invalid("VAP requires two synchronized finite audio channels") }
        let count = min(user.count, windowSamples)
        let offset = windowSamples - count
        var window = [Float](repeating: 0, count: 2 * windowSamples)
        window.replaceSubrange(offset..<windowSamples, with: user.suffix(count))
        window.replaceSubrange((windowSamples + offset)..<(2 * windowSamples), with: agent.suffix(count))
        return window
    }
}
#endif

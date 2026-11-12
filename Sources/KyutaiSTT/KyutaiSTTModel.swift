import AudioCommon
import Foundation
import MLX
import MLXNN
import PersonaPlex
import MLXCommon

/// Native delayed-stream speech recognition. Use each model from one task at a time.
/// Original upstream bundles and exported FP16/INT4/INT8 bundles use the same API.
public final class KyutaiSTTModel: SpeechRecognitionModel {
    var mlxCacheLimitLease = MLXCacheLimitCoordinator.shared.acquireDefault()

    public static let defaultModelId = KyutaiSTTVariant.int8.modelId
    public static let upstreamModelId = "kyutai/stt-1b-en_fr"
    public static let englishModelId = "kyutai/stt-2.6b-en"
    public let config: KyutaiSTTConfig
    public var inputSampleRate: Int { KyutaiSTTConfig.sampleRate }

    let decoder: KyutaiSTTDecoder
    let mimi: KyutaiSTTCodec
    let vocabulary: SentencePieceModel
    private var activeSession: UUID?

    private init(config: KyutaiSTTConfig, vocabulary: SentencePieceModel) {
        self.config = config
        self.vocabulary = vocabulary
        self.decoder = KyutaiSTTDecoder(config: config)
        self.mimi = KyutaiSTTCodec(numCodebooks: config.numCodebooks)
    }

    public static func fromDirectory(_ directory: URL) throws -> KyutaiSTTModel {
        let loadingCacheLimit = MLXCacheLimitCoordinator.shared.acquireDefault()
        defer { loadingCacheLimit?.release() }

        let config = try JSONDecoder().decode(
            KyutaiSTTConfig.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let vocabulary = try SentencePieceModel(contentsOf: directory.appendingPathComponent(config.tokenizerName))
        guard vocabulary.count == config.textCard else {
            throw KyutaiSTTError.incompatibleCheckpoint("tokenizer vocabulary does not match text_card")
        }
        let model = KyutaiSTTModel(config: config, vocabulary: vocabulary)
        let weights = try KyutaiSTTDecoder.sanitize(
            MLX.loadArrays(url: directory.appendingPathComponent("model.safetensors")))
        try model.decoder.update(parameters: .unflattened(weights), verify: .all)
        eval(model.decoder)

        let rawMimi = try MLX.loadArrays(url: directory.appendingPathComponent(config.mimiName))
        let mimiWeights = try KyutaiSTTCodec.sanitize(rawMimi)
        try model.mimi.update(parameters: .unflattened(mimiWeights), verify: .all)
        eval(model.mimi)
        return model
    }

    public static func load(
        _ modelPath: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        downloadOptions: DownloadOptions = .init(),
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> KyutaiSTTModel {
        let loadingCacheLimit = MLXCacheLimitCoordinator.shared.acquireDefault()
        defer { loadingCacheLimit?.release() }

        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDirectory), isDirectory.boolValue {
            return try fromDirectory(URL(fileURLWithPath: modelPath, isDirectory: true))
        }
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelPath)
        // Read the configuration first: original and exported bundles name Mimi differently.
        try await HuggingFaceDownloader.downloadFiles(
            modelId: modelPath, to: directory, files: ["config.json"],
            offlineMode: offlineMode,
            downloadOptions: downloadOptions)
        let config = try JSONDecoder().decode(
            KyutaiSTTConfig.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelPath, to: directory,
            additionalFiles: ["model.safetensors", config.mimiName, config.tokenizerName],
            offlineMode: offlineMode, downloadOptions: downloadOptions, progressHandler: progressHandler)
        return try fromDirectory(directory)
    }

    /// Create one incremental 24 kHz PCM session. Finish or cancel it before reusing the model.
    public func makeSession() throws -> KyutaiSTTSession {
        guard activeSession == nil else { throw KyutaiSTTError.sessionActive }
        let id = UUID()
        activeSession = id
        mimi.encoder.resetState()
        mimi.downsample.resetState()
        return KyutaiSTTSession(model: self, id: id)
    }

    func encodeFrame(_ pcm: [Float], cache: [KyutaiSTTCache]) -> MLXArray {
        let encoded = mimi.encoder.step(MLXArray(pcm).reshaped(1, 1, pcm.count))
        // Match the original codec's two-frame attention step and ring capacity.
        let transformed = mimi.encoder_transformer(encoded, cache: cache)[0]
        let latent = mimi.downsample.step(transformed)
        return mimi.quantizer.encode(latent)
    }

    func releaseSession(_ id: UUID) {
        if activeSession == id { activeSession = nil }
    }

    /// Validates audio, resamples to 24 kHz, and flushes the delayed final text.
    public func transcribe(audio: [Float], sampleRate: Int) throws -> String {
        guard sampleRate > 0, audio.allSatisfy(\.isFinite) else {
            throw KyutaiSTTError.invalidAudio("sample rate must be positive and samples must be finite")
        }
        guard !audio.isEmpty else { return "" }
        let pcm = sampleRate == inputSampleRate
            ? audio : AudioFileLoader.resample(audio, from: sampleRate, to: inputSampleRate)
        let session = try makeSession()
        defer { session.cancel() }
        _ = try session.append(pcm)
        _ = try session.finish()
        return session.text
    }

    /// Protocol convenience. Language is detected from speech; this model has no language prompt.
    /// Invalid input returns an empty transcript. Use the throwing overload to inspect errors.
    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        (try? transcribe(audio: audio, sampleRate: sampleRate)) ?? ""
    }
}

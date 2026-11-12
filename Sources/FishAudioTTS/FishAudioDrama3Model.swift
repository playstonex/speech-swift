import AudioCommon
import Foundation
import MLX
import MLXNN
import SpeechRuntime
import MLXCommon

/// Experimental native Drama 3 inference using the converted MLX checkpoint format.
public final class FishAudioDrama3Model: SpeechGenerationModel, @unchecked Sendable {
    var mlxCacheLimitLease = MLXCacheLimitCoordinator.shared.acquireDefault()

    public static let defaultModelId = "mlx-community/fish-audio-drama-3-4bit"
    public let config: FishAudioDrama3Config
    public let sampleRate = FishAudioCodecDefaults.sampleRate
    public let bundleDirectory: URL
    public let memoryOptions: FishAudioMemoryOptions
    private let tokenizer: FishAudioTokenizer
    let network: Drama3Network
    let codec: FishAudioCodec
    let inference = InferenceGate()

    private init(config: FishAudioDrama3Config, directory: URL, tokenizer: FishAudioTokenizer,
                 network: Drama3Network, codec: FishAudioCodec, memoryOptions: FishAudioMemoryOptions) {
        self.config = config; bundleDirectory = directory; self.tokenizer = tokenizer
        self.network = network; self.codec = codec
        self.memoryOptions = memoryOptions
    }

    public static func fromPretrained(modelId: String = defaultModelId, cacheDir: URL? = nil,
                                      offlineMode: Bool = false,
                                      memoryOptions: FishAudioMemoryOptions = .balanced,
                                      downloadOptions: DownloadOptions = .init(),
                                      progressHandler: ((Double, String) -> Void)? = nil) async throws -> FishAudioDrama3Model {
        let loadingCacheLimit = MLXCacheLimitCoordinator.shared.acquireDefault()
        defer { loadingCacheLimit?.release() }

        try memoryOptions.validate()
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadFiles(modelId: modelId, to: directory,
            files: ["config.json", "tokenizer.json", "tokenizer_config.json", "model*.safetensors",
                    "codec.safetensors", "LICENSE", "NOTICE"], offlineMode: offlineMode, downloadOptions: downloadOptions) {
            progressHandler?($0 * 0.85, "Downloading Fish Audio Drama 3")
        }
        return try await fromBundle(directory, memoryOptions: memoryOptions) { progress, message in
            progressHandler?(0.85 + progress * 0.15, message)
        }
    }

    public static func fromBundle(_ directory: URL,
                                  memoryOptions: FishAudioMemoryOptions = .balanced,
                                  progressHandler: ((Double, String) -> Void)? = nil) async throws -> FishAudioDrama3Model {
        let loadingCacheLimit = MLXCacheLimitCoordinator.shared.acquireDefault()
        defer { loadingCacheLimit?.release() }

        try memoryOptions.validate()
        progressHandler?(0, "Loading Drama 3 configuration")
        let config = try FishAudioDrama3Config.load(from: directory.appendingPathComponent("config.json"))
        guard config.tower.codebookCount == FishAudioCodecDefaults.totalCodebooks,
              config.tower.vocabularySize == FishAudioCodecDefaults.semanticCodebookSize else {
            throw FishAudioError.invalidConfig("Drama 3 requires the 10-codebook Fish DAC codec")
        }
        let tokenizer = try await FishAudioTokenizer.load(from: directory)
        _ = try FishAudioDrama3Prompt.build(text: "Hello.", tokenizer: tokenizer, config: config)
        progressHandler?(0.1, "Loading Drama 3 model weights")
        let network = try loadNetwork(config: config, from: directory)
        memoryOptions.trimCache()
        progressHandler?(0.8, "Loading Fish DAC codec")
        let codec = try FishAudioCodec.loadForDrama3(from: directory, memoryOptions: memoryOptions)
        memoryOptions.trimCache()
        try Task.checkCancellation()
        progressHandler?(1, "Fish Audio Drama 3 ready")
        return FishAudioDrama3Model(config: config, directory: directory, tokenizer: tokenizer, network: network,
                                   codec: codec, memoryOptions: memoryOptions)
    }

    static func loadNetwork(config: FishAudioDrama3Config, from directory: URL) throws -> Drama3Network {
        let files = try FishAudioWeightLoader.modelSafetensorFiles(in: directory)
        var weights: [String: MLXArray] = [:]
        for file in files {
            try Task.checkCancellation()
            for (key, value) in try MLX.loadArrays(url: file) {
                guard weights.updateValue(value, forKey: key) == nil else {
                    throw FishAudioError.invalidWeights("duplicate parameter \(key)")
                }
            }
        }
        // Every quantized path must be explicit: this format mixes expert and dense bit widths.
        let quantizedPaths = Set(weights.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(7)) })
        guard quantizedPaths == Set(config.quantization.keys) else {
            throw FishAudioError.invalidWeights("quantization paths do not match the checkpoint")
        }
        let network = Drama3Network(config)
        do { try network.update(parameters: ModuleParameters.unflattened(weights), verify: .all) }
        catch { throw FishAudioError.invalidWeights(String(describing: error)) }
        network.train(false)
        eval(network)
        return network
    }

    public func generate(text: String, language: String?) async throws -> [Float] {
        try await generate(text: text, options: .init())
    }

    public func generate(text: String, references: [FishAudioReferencePrompt] = [],
                         history: [FishAudioReferencePrompt] = [], options: FishAudioDrama3Options = .init()) async throws -> [Float] {
        let input = try validatedInput(text: text, references: references, history: history, options: options)
        let permit = try await inference.acquire()
        defer { memoryOptions.trimCache(); permit.release() }
        let codes = try network.generate(input, options: options, memoryOptions: memoryOptions)
        memoryOptions.trimCache()
        try Task.checkCancellation()
        let samples = try codec.decode(codes)
        guard !samples.isEmpty, samples.allSatisfy(\.isFinite) else {
            throw FishAudioError.invalidInput("Drama 3 decoded an invalid waveform")
        }
        return samples
    }

    public func generate(text: String, referenceAudio: [Float], referenceSampleRate: Int, referenceText: String,
                         options: FishAudioDrama3Options = .init()) async throws -> [Float] {
        try FishAudioDrama3Prompt.validateText(text)
        try FishAudioDrama3Prompt.validateText(referenceText)
        try options.validate()
        let reference = try await encodeReferencePrompt(audio: referenceAudio, sampleRate: referenceSampleRate, text: referenceText)
        return try await generate(text: text, references: [reference], options: options)
    }

    public func generateCodebooks(text: String, references: [FishAudioReferencePrompt] = [],
                                  history: [FishAudioReferencePrompt] = [], options: FishAudioDrama3Options = .init()) async throws -> FishAudioGeneratedCodebooks {
        let input = try validatedInput(text: text, references: references, history: history, options: options)
        let permit = try await inference.acquire()
        defer { memoryOptions.trimCache(); permit.release() }
        return try network.generate(input, options: options, memoryOptions: memoryOptions)
    }

    public func encodeReferencePrompt(audio: [Float], sampleRate: Int, text: String) async throws -> FishAudioReferencePrompt {
        try FishAudioDrama3Prompt.validateText(text)
        guard sampleRate > 0, !audio.isEmpty, audio.allSatisfy(\.isFinite), audio.count <= sampleRate * 30 else {
            throw FishAudioError.invalidInput("Drama 3 reference must contain 0–30 seconds of finite mono audio")
        }
        let permit = try await inference.acquire()
        defer { memoryOptions.trimCache(); permit.release() }
        try Task.checkCancellation()
        let codes = try codec.encode(audio: audio, sampleRate: sampleRate)
        return try FishAudioReferencePrompt(text: text, codes: codes.codes)
    }

    func validatedInput(text: String, references: [FishAudioReferencePrompt], history: [FishAudioReferencePrompt],
                                options: FishAudioDrama3Options) throws -> FishAudioModelInput {
        try options.validate()
        return try FishAudioDrama3Prompt.build(text: text, references: references, history: history, tokenizer: tokenizer, config: config)
    }
}

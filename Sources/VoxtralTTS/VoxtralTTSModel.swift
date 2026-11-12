import AudioCommon
import Foundation
import MLX
import MLXNN
import SpeechRuntime
import MLXCommon

final class VoxtralAudioEmbeddings: Module {
    @ModuleInfo var embeddings: Embedding
    init(_ config: VoxtralTTSConfig) {
        _embeddings.wrappedValue = Embedding(embeddingCount: config.audio.embeddingCount, dimensions: config.transformer.dim)
        super.init()
    }
}

final class VoxtralTTSNetwork: Module {
    @ModuleInfo(key: "language_model") var languageModel: VoxtralTTSLlamaModel
    @ModuleInfo(key: "acoustic_transformer") var acoustic: VoxtralAcousticTransformer
    @ModuleInfo(key: "audio_tokenizer") var decoder: VoxtralAudioDecoder
    @ModuleInfo(key: "audio_codebook_embeddings") var audioEmbeddings: VoxtralAudioEmbeddings

    init(_ config: VoxtralTTSConfig) {
        _languageModel.wrappedValue = VoxtralTTSLlamaModel(config)
        _acoustic.wrappedValue = VoxtralAcousticTransformer(config)
        _decoder.wrappedValue = VoxtralAudioDecoder(config.codec)
        _audioEmbeddings.wrappedValue = VoxtralAudioEmbeddings(config)
        super.init()
    }
}

/// Voxtral 4B TTS with the checkpoint's preset voices. Each request owns its KV
/// cache, audio frames and random keys. Calls on one model are admitted serially.
public final class VoxtralTTSModel: SpeechGenerationModel, @unchecked Sendable {
    var mlxCacheLimitLease = MLXCacheLimitCoordinator.shared.acquireDefault()

    public static let defaultModelId = "aufklarer/Voxtral-4B-TTS-2603-MLX-4bit"
    public static let defaultVoice = "casual_male"
    public static let voices = [
        "casual_female", "casual_male", "cheerful_female", "neutral_female", "neutral_male",
        "pt_male", "pt_female", "nl_male", "nl_female", "it_male", "it_female", "fr_male", "fr_female",
        "es_male", "es_female", "de_male", "de_female", "ar_male", "hi_male", "hi_female",
    ]
    static var bundleFiles: [String] {
        ["config.json", "tekken.json", "model.safetensors"]
            + voices.map { "voice_embedding/\($0).safetensors" }
    }
    public let config: VoxtralTTSConfig
    public var sampleRate: Int { config.sampleRate }
    public var availableVoices: [String] { tokenizer.voices }
    private let tokenizer: VoxtralTTSTokenizer
    let network: VoxtralTTSNetwork
    private let directory: URL
    let inference = InferenceGate()

    private init(config: VoxtralTTSConfig, tokenizer: VoxtralTTSTokenizer,
                 network: VoxtralTTSNetwork, directory: URL) {
        self.config = config
        self.tokenizer = tokenizer
        self.network = network
        self.directory = directory
    }

    public static func fromBundle(_ directory: URL) throws -> VoxtralTTSModel {
        let loadingCacheLimit = MLXCacheLimitCoordinator.shared.acquireDefault()
        defer { loadingCacheLimit?.release() }

        let config = try JSONDecoder().decode(VoxtralTTSConfig.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let tokenizer = try VoxtralTTSTokenizer(tekkenURL: directory.appendingPathComponent("tekken.json"))
        guard Set(tokenizer.voices) == Set(voices) else {
            throw VoxtralTTSError.invalidCheckpoint("expected all 20 preset voices")
        }
        for voice in voices {
            guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("voice_embedding/\(voice).safetensors").path) else {
                throw VoxtralTTSError.invalidCheckpoint("missing preset voice \(voice)")
            }
        }
        let original = try MLX.loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        var weights: [String: MLXArray] = [:]
        for (key, value) in original {
            let mapped = key.replacingOccurrences(of: "language_model.model.model.", with: "language_model.")
            guard weights.updateValue(value, forKey: mapped) == nil else {
                throw VoxtralTTSError.invalidCheckpoint("duplicate parameter \(mapped)")
            }
        }
        let network = VoxtralTTSNetwork(config)
        if let quantization = config.quantization {
            MLXNN.quantize(model: network) { path, _ in
                weights["\(path).scales"] == nil ? nil : (quantization.groupSize, quantization.bits, .affine)
            }
        }
        do {
            try network.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        } catch {
            throw VoxtralTTSError.invalidCheckpoint(error.localizedDescription)
        }
        network.train(false)
        eval(network)
        return VoxtralTTSModel(config: config, tokenizer: tokenizer, network: network, directory: directory)
    }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        downloadOptions: DownloadOptions = .init(),
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> VoxtralTTSModel {
        let loadingCacheLimit = MLXCacheLimitCoordinator.shared.acquireDefault()
        defer { loadingCacheLimit?.release() }

        try Task.checkCancellation()
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        // Preset embeddings are required assets outside the backbone's weight index.
        try await HuggingFaceDownloader.downloadFiles(modelId: modelId, to: directory,
            files: bundleFiles,
            offlineMode: offlineMode, downloadOptions: downloadOptions, progressHandler: { progressHandler?($0, "Downloading Voxtral TTS") })
        try Task.checkCancellation()
        return try fromBundle(directory)
    }

    public func generate(text: String, language: String?) async throws -> [Float] {
        let voice = try Self.voice(for: language)
        return try await generate(text: text, voice: voice)
    }

    /// Generate a complete utterance. Reaching the frame limit throws instead
    /// of returning audio with a silently truncated final word.
    public func generate(text: String, voice: String = defaultVoice,
                         options: VoxtralTTSOptions = .init()) async throws -> [Float] {
        let prompt = try validatedPrompt(text: text, voice: voice, options: options)
        let permit = try await inference.acquire()
        defer { permit.release() }
        try Task.checkCancellation()
        var generator = try frameGenerator(prompt: prompt, voice: voice, options: options)
        var frames: [MLXArray] = []
        while let codes = try generator.next() { frames.append(codes) }
        try Task.checkCancellation()
        return try decodedSamples(network.decoder.decode(concatenated(frames, axis: 1)), frames: frames.count)
    }

    /// Stream PCM while frames are generated. The callback is awaited before
    /// continuing, providing backpressure and retaining admission through delivery.
    /// A failure after earlier chunks throws; it never emits a successful final chunk.
    public func generateStreaming(
        text: String, voice: String = defaultVoice, options: VoxtralTTSOptions = .init(),
        onChunk: (AudioChunk) async throws -> Void
    ) async throws {
        let state = VoxtralTTSStreamState(model: self, voice: { voice }, text: text, options: options)
        do {
            while let chunk = try await state.next() { try await onChunk(chunk) }
        } catch {
            await state.close()
            throw error
        }
    }

    public func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error> {
        makeStream(text: text, voice: { try Self.voice(for: language) }, options: .init())
    }

    /// Chunks contain only new audio. Decoding advances when the iterator asks
    /// for the next chunk; there is no background producer or PCM backlog.
    public func generateStream(text: String, voice: String = defaultVoice,
                               options: VoxtralTTSOptions = .init()) -> AsyncThrowingStream<AudioChunk, Error> {
        makeStream(text: text, voice: { voice }, options: options)
    }

    private func makeStream(text: String, voice: @escaping @Sendable () throws -> String,
                            options: VoxtralTTSOptions) -> AsyncThrowingStream<AudioChunk, Error> {
        let state = VoxtralTTSStreamState(model: self, voice: voice, text: text, options: options)
        return AsyncThrowingStream(unfolding: {
            let chunk = try await state.next()
            if chunk?.isFinal == true { await state.close() }
            return chunk
        })
    }

    fileprivate func validatedPrompt(text: String, voice: String, options: VoxtralTTSOptions) throws -> [Int] {
        try Task.checkCancellation()
        try options.validate()
        let normalized = try VoxtralTTSTokenizer.normalizedText(text)
        guard Self.voices.contains(voice) else { throw VoxtralTTSError.unknownVoice(voice) }
        let prompt = try tokenizer.speechPrompt(text: normalized, voice: voice)
        guard prompt.count + options.maxFrames <= config.maximumContext,
              prompt.allSatisfy({ $0 >= 0 && $0 < config.vocabSize }) else {
            throw VoxtralTTSError.invalidInput("speech prompt exceeds the model context or vocabulary")
        }
        return prompt
    }

    fileprivate func frameGenerator(prompt: [Int], voice: String, options: VoxtralTTSOptions) throws -> VoxtralFrameGenerator {
        try VoxtralFrameGenerator(network: network, config: config, prompt: prompt,
            voiceURL: directory.appendingPathComponent("voice_embedding/\(voice).safetensors"),
            voiceFrames: tokenizer.voiceTokenCounts[voice]!, options: options)
    }

    fileprivate func decodedSamples(_ audio: MLXArray, frames: Int) throws -> [Float] {
        let audio = audio.reshaped(-1).asType(.float32)
        eval(audio)
        try Task.checkCancellation()
        let samples = audio.asArray(Float.self)
        guard samples.count == frames * config.codec.samplesPerFrame,
              samples.allSatisfy(\.isFinite) else {
            throw VoxtralTTSError.invalidCheckpoint("decoder produced invalid PCM samples")
        }
        return samples
    }

    public static func voice(for language: String?) throws -> String {
        guard let language, !language.isEmpty else { return defaultVoice }
        let aliases = ["english": "en", "french": "fr", "german": "de", "spanish": "es",
                       "italian": "it", "portuguese": "pt", "dutch": "nl", "arabic": "ar", "hindi": "hi"]
        let raw = language.lowercased().split(separator: "-").first.map(String.init) ?? "en"
        let code = aliases[raw] ?? raw
        if code == "en" { return defaultVoice }
        let voice = "\(code)_male"
        guard voices.contains(voice) else { throw VoxtralTTSError.invalidInput("unsupported language: \(language)") }
        return voice
    }
}

/// One pull-driven utterance. Dropping the stream releases its permit and all
/// generation state; no unstructured task can keep decoding after its consumer.
private actor VoxtralTTSStreamState {
    let model: VoxtralTTSModel
    let voice: @Sendable () throws -> String
    let text: String
    let options: VoxtralTTSOptions
    var permit: InferencePermit?
    var generator: VoxtralFrameGenerator?
    var decoder: VoxtralAudioDecoder.State?
    var emittedFrames = 0
    var start: Double?
    var finished = false
    var advancing = false

    init(model: VoxtralTTSModel, voice: @escaping @Sendable () throws -> String,
         text: String, options: VoxtralTTSOptions) {
        self.model = model
        self.voice = voice
        self.text = text
        self.options = options
    }

    deinit { permit?.release() }

    func next() async throws -> AudioChunk? {
        guard !advancing else { throw VoxtralTTSError.invalidInput("a speech stream supports one consumer") }
        advancing = true
        defer { advancing = false }
        if finished { close(); return nil }
        do {
            try Task.checkCancellation()
            if permit == nil {
                start = ProcessInfo.processInfo.systemUptime
                let voice = try voice()
                let prompt = try model.validatedPrompt(text: text, voice: voice, options: options)
                permit = try await model.inference.acquire()
                try Task.checkCancellation()
                generator = try model.frameGenerator(prompt: prompt, voice: voice, options: options)
                decoder = VoxtralAudioDecoder.State(model.config.codec)
            }
            var pending: [MLXArray] = []
            while pending.count < options.streamingChunkFrames {
                guard let codes = try generator!.next() else { finished = true; break }
                pending.append(codes)
            }
            let samples = pending.isEmpty ? [] : try model.decodedSamples(
                model.network.decoder.decode(concatenated(pending, axis: 1), state: &decoder!), frames: pending.count)
            try Task.checkCancellation()
            let chunk = AudioChunk(samples: samples, sampleRate: model.sampleRate,
                frameIndex: emittedFrames, isFinal: finished,
                elapsedTime: ProcessInfo.processInfo.systemUptime - start!)
            emittedFrames += pending.count
            // Exact chunk multiples have an empty final marker. Convolution
            // overlap after the last frame is trimmed, matching full decoding.
            // Callback delivery keeps admission through the final write. The
            // iterator adapter closes when returning its final chunk instead.
            return chunk
        } catch {
            close()
            throw error
        }
    }

    func close() {
        finished = true
        generator = nil
        decoder = nil
        permit?.release()
        permit = nil
    }
}

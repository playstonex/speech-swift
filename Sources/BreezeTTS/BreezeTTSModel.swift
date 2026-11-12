import AudioCommon
import Foundation
import MLX
import MLXCommon
import MLXNN
import Qwen3TTS

final class BreezeNetwork: Module {
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "backbone_model") var backbone: BreezeBackbone
    @ModuleInfo(key: "depth_decoder") var depth: BreezeDepthDecoder
    @ModuleInfo(key: "text_encoder") var textEncoder: BreezeTTSTextEncoder
    @ModuleInfo(key: "text_encoder_proj") var textProjection: Linear

    init(config: BreezeTTSConfig) {
        _lmHead.wrappedValue = breezeLinear(config.backboneConfig.hiddenSize, config.audioVocabSize + 1,
                                            quantization: config.quantization)
        _backbone.wrappedValue = BreezeBackbone(config: config)
        _depth.wrappedValue = BreezeDepthDecoder(config: config.depthDecoderConfig)
        _textEncoder.wrappedValue = BreezeTTSTextEncoder(config: config.textEncoderConfig)
        _textProjection.wrappedValue = breezeLinear(config.textEncoderConfig.hiddenSize,
            config.backboneConfig.hiddenSize, quantization: config.quantization)
    }
}

/// Native Breeze TTS 2 inference. Use one generation at a time per instance.
/// Weights and generated audio are subject to the upstream non-commercial license.
public final class BreezeTTSModel: SpeechGenerationModel, @unchecked Sendable {
    var mlxCacheLimitLease = MLXCacheLimitCoordinator.shared.acquireDefault()

    public static let defaultModelId = "aufklarer/Breeze-TTS-2-MLX-8bit"
    public let config: BreezeTTSConfig
    public var sampleRate: Int { config.sampleRate }
    public let bundleDirectory: URL
    let network: BreezeNetwork
    private let decoder: SpeechTokenizerDecoder
    private var encoder: SpeechTokenizerEncoder?
    private let tokenizer: BreezeTextTokenizer
    private let generationLock = NSLock()
    private var generating = false

    private static let requiredFiles = [
        "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "audio_tokenizer/config.json", "audio_tokenizer/model.safetensors"
    ]

    private init(config: BreezeTTSConfig, directory: URL, network: BreezeNetwork,
                 decoder: SpeechTokenizerDecoder, tokenizer: BreezeTextTokenizer) {
        self.config = config; bundleDirectory = directory; self.network = network
        self.decoder = decoder; self.tokenizer = tokenizer
    }

    public static func fromPretrained(modelId: String = defaultModelId, cacheDir: URL? = nil,
        offlineMode: Bool = false, downloadOptions: DownloadOptions = .init(), progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> BreezeTTSModel {
        let loadingCacheLimit = MLXCacheLimitCoordinator.shared.acquireDefault()
        defer { loadingCacheLimit?.release() }

        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadFiles(modelId: modelId, to: directory,
            files: requiredFiles, offlineMode: offlineMode, downloadOptions: downloadOptions) { progress in
            progressHandler?(progress * 0.9, "Downloading Breeze TTS 2")
        }
        return try await fromBundle(directory, progressHandler: progressHandler)
    }

    public static func fromBundle(_ directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> BreezeTTSModel {
        let loadingCacheLimit = MLXCacheLimitCoordinator.shared.acquireDefault()
        defer { loadingCacheLimit?.release() }

        for file in requiredFiles where !FileManager.default.fileExists(atPath: directory.appendingPathComponent(file).path) {
            throw BreezeTTSError.missingFile(file)
        }
        let config = try JSONDecoder().decode(BreezeTTSConfig.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let tokenizer = try await BreezeTextTokenizer.fromBundle(directory)
        progressHandler?(0.92, "Loading Breeze weights")
        let network = withThrowawayRandomState { BreezeNetwork(config: config) }
        let weights = sanitize(weights: try CommonWeightLoader.loadAllSafetensors(from: directory))
        try network.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(network.parameters())
        let codecConfig = try codecConfiguration(Data(contentsOf: directory.appendingPathComponent("audio_tokenizer/config.json")))
        let decoder = BreezeTTSCodec.makeDecoder(config: codecConfig)
        try TTSWeightLoader.loadSpeechTokenizerDecoderWeights(into: decoder,
            from: directory.appendingPathComponent("audio_tokenizer"))
        eval(decoder.parameters())
        progressHandler?(1, "Breeze TTS 2 ready")
        return BreezeTTSModel(config: config, directory: directory, network: network, decoder: decoder, tokenizer: tokenizer)
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights.filter { key, _ in
            !key.hasPrefix("codec_model.") && !key.hasPrefix("embed_text_tokens.")
                && !key.hasSuffix(".initialized") && !key.contains("rotary_emb.inv_freq")
        }
        if let tied = result["depth_decoder.model.embed_tokens.weight"] {
            result["backbone_model.embed_tokens.embed_audio_tokens.weight"] = tied
        }
        return result
    }

    static func codecConfiguration(_ data: Data) throws -> SpeechTokenizerDecoderConfig {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              object["model_type"] as? String == "qwen3_tts_tokenizer_12hz",
              object["input_sample_rate"] as? Int == 24_000,
              object["output_sample_rate"] as? Int == 24_000,
              object["decode_upsample_rate"] as? Int == 1920,
              let decoder = object["decoder_config"] as? [String: Any],
              decoder["hidden_size"] as? Int == 512,
              decoder["num_attention_heads"] as? Int == 16,
              decoder["num_hidden_layers"] as? Int == 8,
              let eps = decoder["rms_norm_eps"] as? Double, eps.isFinite, eps > 0 else {
            throw BreezeTTSError.invalidConfiguration("Expected the bundled 24 kHz Qwen3 12 Hz audio tokenizer")
        }
        var config = SpeechTokenizerDecoderConfig()
        // The published codec metadata lists a 512-wide codebook; its actual
        // tensors use the standard 256-wide Qwen3 codebooks handled by this decoder.
        config.rmsNormEps = Float(eps)
        return config
    }

    static func promptText(text: String, instruction: String?) -> String {
        guard let instruction, !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "[S0]\(text)"
        }
        return "[S0]<ins_bos>\(instruction)<ins_eos>\(text)"
    }

    public func generate(text: String, language: String? = nil) async throws -> [Float] {
        try await generate(text: text, instruction: nil, language: language)
    }

    public func generateStream(text: String, language: String? = nil) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.generateStreaming(text: text, language: language) { continuation.yield($0) }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public func generate(text: String, instruction: String?, referenceAudio: [Float]? = nil,
        referenceText: String? = nil, language: String? = nil, options: BreezeTTSOptions = .init()
    ) async throws -> [Float] {
        var samples: [Float] = []
        try await generateStreaming(text: text, instruction: instruction, referenceAudio: referenceAudio,
            referenceText: referenceText, language: language, options: options) { samples += $0.samples }
        return samples
    }

    /// Reference audio must be mono Float32 PCM at 24 kHz with its complete transcript.
    /// Audio chunks are emitted after codec decoding; frame-limit exhaustion throws.
    public func generateStreaming(text: String, instruction: String? = nil, referenceAudio: [Float]? = nil,
        referenceText: String? = nil, language: String? = nil, options: BreezeTTSOptions = .init(),
        onChunk: (AudioChunk) async throws -> Void
    ) async throws {
        try options.validate()
        try Self.validateInput(text: text, referenceAudio: referenceAudio, referenceText: referenceText, language: language)
        let acquired = generationLock.withLock { () -> Bool in
            guard !generating else { return false }
            generating = true; return true
        }
        guard acquired else { throw BreezeTTSError.invalidInput("Breeze is already generating on this instance") }
        defer { generationLock.withLock { generating = false } }
        try Task.checkCancellation()
        // A local random state prevents a request from altering other models' sampling seeds.
        let random = MLXRandom.RandomState(seed: options.seed)
        let reference = try referenceEmbeddings(audio: referenceAudio, text: referenceText)
        let conditional = try promptEmbeddings(text: text, instruction: instruction, reference: reference)
        let useGuidance = instruction?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false && options.cfgScale > 1
        let unconditional = useGuidance ? try promptEmbeddings(text: text, instruction: nil, reference: reference) : nil
        eval(conditional)
        if let unconditional { eval(unconditional) }
        let cache = network.backbone.makeCache()
        var hidden = network.backbone(conditional, cache: cache)[0..., -1, 0...]
        let negativeCache = unconditional.map { _ in network.backbone.makeCache() }
        var negative: MLXArray?
        if let unconditional, let negativeCache {
            negative = network.backbone(unconditional, cache: negativeCache)[0..., -1, 0...]
        }
        eval(hidden)
        if let negative { eval(negative) }
        var frames: [[Int32]] = []
        var emitted = 0
        var threshold = options.firstChunkFrames
        var reachedEOS = false

        for _ in 0..<options.maxFrames {
            try Task.checkCancellation()
            let logits = guided(network.lmHead(hidden), negative.map(network.lmHead.callAsFunction), scale: options.cfgScale)
            let first = sample(logits, allowsEOS: true, options: options, random: random)
            if first == config.audioVocabSize { reachedEOS = true; break }
            var frameTokens = [MLXArray(Int32(first))]
            let depthCache = network.depth.model.layers.map { _ in BreezeKVCache() }
            let branches = negative.map { concatenated([hidden, $0], axis: 0) } ?? hidden
            var depthHidden = network.depth.model.prefill(first: first, backbone: branches, cache: depthCache)
            for codebook in 1..<config.numCodebooks {
                if codebook > 1 {
                    depthHidden = network.depth.model.step(token: frameTokens.last!, codebook: codebook - 1,
                        batchSize: branches.dim(0), cache: depthCache)
                }
                let depthLogits = network.depth.logits(depthHidden, codebook: codebook)
                let combined = useGuidance
                    ? guided(depthLogits[0..<1], depthLogits[1..<2], scale: options.cfgScale) : depthLogits
                let token = breezeSampleCodebook(combined, codecVocabSize: config.codecVocabSize,
                    audioVocabSize: config.audioVocabSize, allowsEOS: false, options: options, random: random)
                frameTokens.append(token)
                // Submit depth work without a host read for each codebook.
                let state = [token] + depthCache.flatMap(\.arrays)
                asyncEval(state)
            }
            let frameCodes = stacked(frameTokens)
            let frame = frameCodes.asArray(Int32.self)
            frames.append(frame)
            if frames.count - emitted >= threshold {
                let samples = try decode(frames: frames, from: emitted, context: options.decoderLeftContext)
                try await onChunk(AudioChunk(samples: samples, sampleRate: sampleRate, frameIndex: emitted, isFinal: false))
                emitted = frames.count; threshold = options.chunkFrames
            }
            let embeddings = network.backbone.embedTokens(frameCodes.reshaped([1, 1, config.numCodebooks]))
            hidden = network.backbone(embeddings, cache: cache)[0..., -1, 0...]
            if let negativeCache { negative = network.backbone(embeddings, cache: negativeCache)[0..., -1, 0...] }
            // Evaluate the hidden states and caches together to bound the lazy graph.
            var backboneState: [MLXArray] = [hidden]
            if let negative { backboneState.append(negative) }
            backboneState.append(contentsOf: cache.flatMap(\.arrays))
            if let negativeCache {
                backboneState.append(contentsOf: negativeCache.flatMap(\.arrays))
            }
            eval(backboneState)
        }
        guard reachedEOS else { throw BreezeTTSError.frameLimit(options.maxFrames) }
        guard !frames.isEmpty else { throw BreezeTTSError.invalidInput("Breeze produced no audio frames") }
        try Task.checkCancellation()
        let tail = emitted < frames.count ? try decode(frames: frames, from: emitted, context: options.decoderLeftContext) : []
        try await onChunk(AudioChunk(samples: tail, sampleRate: sampleRate, frameIndex: emitted, isFinal: true))
    }

    static func validateInput(text: String, referenceAudio: [Float]?, referenceText: String?, language: String?) throws {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw BreezeTTSError.invalidInput("Breeze requires nonempty synthesis text")
        }
        if let language, !["en", "english", "zh", "chinese", "zh-cn", "mandarin"].contains(language.lowercased()) {
            throw BreezeTTSError.invalidInput("Breeze supports English and Chinese; language is inferred from the text")
        }
        guard (referenceAudio == nil) == (referenceText == nil) else {
            throw BreezeTTSError.invalidInput("Reference audio and its transcript must be provided together")
        }
        if let referenceAudio, let referenceText {
            guard !referenceAudio.isEmpty, referenceAudio.count <= 30 * 24_000,
                  referenceAudio.allSatisfy(\.isFinite),
                  !referenceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw BreezeTTSError.invalidInput("Use a nonempty reference clip up to 30 seconds and its exact transcript")
            }
        }
    }

    private func promptEmbeddings(text: String, instruction: String?, reference: MLXArray?) throws -> MLXArray {
        let target = try textEmbeddings(Self.promptText(text: text, instruction: instruction))
        let result = reference.map { concatenated([$0, target], axis: 1) } ?? target
        guard result.dim(1) <= 2048 else { throw BreezeTTSError.invalidInput("Breeze prompt exceeds 2048 positions") }
        return result
    }

    private func textEmbeddings(_ text: String) throws -> MLXArray {
        let ids = try tokenIDs(text)
        guard !ids.isEmpty, ids.count <= 2048, ids.allSatisfy({ $0 >= 0 && $0 < config.textEncoderConfig.vocabSize }) else {
            throw BreezeTTSError.invalidInput("Breeze text segment is empty, too long, or has invalid token IDs")
        }
        return network.textProjection(network.textEncoder(MLXArray(ids.map(Int32.init)).reshaped([1, -1])))
    }

    func tokenIDs(_ text: String) throws -> [Int] { try tokenizer.encode(text) }

    private func referenceEmbeddings(audio: [Float]?, text: String?) throws -> MLXArray? {
        guard let audio, let text else { return nil }
        if encoder == nil {
            let value = withThrowawayRandomState { SpeechTokenizerEncoder(config: SpeechTokenizerDecoderConfig()) }
            try TTSWeightLoader.loadSpeechTokenizerEncoderWeights(into: value,
                from: bundleDirectory.appendingPathComponent("audio_tokenizer"))
            eval(value.parameters()); encoder = value
        }
        let codes = encoder!.encode(samples: audio).transposed(0, 2, 1)
        let eos = MLXArray(Array(repeating: Int32(config.codebookEOSTokenID), count: config.numCodebooks))
            .reshaped([1, 1, config.numCodebooks])
        return concatenated([try textEmbeddings("[S0]\(text)"), network.backbone.embedTokens(codes),
                             network.backbone.embedTokens(eos)], axis: 1)
    }

    private func guided(_ positive: MLXArray, _ negative: MLXArray?, scale: Float) -> MLXArray {
        negative.map { $0 + scale * (positive - $0) } ?? positive
    }

    private func sample(_ logits: MLXArray, allowsEOS: Bool, options: BreezeTTSOptions,
                        random: MLXRandom.RandomState) -> Int {
        breezeSampleCodebook(logits, codecVocabSize: config.codecVocabSize,
            audioVocabSize: config.audioVocabSize, allowsEOS: allowsEOS,
            options: options, random: random).item(Int.self)
    }

    private func decode(frames: [[Int32]], from start: Int, context: Int) throws -> [Float] {
        let contextStart = max(0, start - context)
        let codes = MLXArray(frames[contextStart...].flatMap { $0 })
            .reshaped([1, frames.count - contextStart, config.numCodebooks]).transposed(0, 2, 1)
        let waveform = decoder(codes).reshaped([-1]).asType(.float32)
        eval(waveform)
        let trim = min((start - contextStart) * 1920, waveform.size)
        let expected = (frames.count - start) * 1920
        guard waveform.size - trim == expected else {
            throw BreezeTTSError.invalidConfiguration("Breeze codec did not produce 1,920 samples per frame")
        }
        return waveform[trim...].asArray(Float.self)
    }
}

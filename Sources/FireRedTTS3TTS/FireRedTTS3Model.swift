import AudioCommon
import Foundation
import MLX
import MLXCommon
import MLXNN
import Tokenizers

/// Audio and the instruction attributes planned by the Instruct checkpoint.
public struct FireRedTTS3Output: Sendable {
    public let samples: [Float]
    public let voiceDescription: String
    public let planningSeconds: Double
    public let acousticSeconds: Double
    public let decodingSeconds: Double
    public let audioPatches: Int
}

/// Native reference-free FireRedTTS3-Instruct voice design, in English and Chinese.
/// Output is buffered 24 kHz mono PCM. Voice cloning and editing are not exposed.
public final class FireRedTTS3Model: SpeechGenerationModel, ModelMemoryManageable, @unchecked Sendable {
    public let config: FireRedTTS3Config
    public let bundleDirectory: URL
    public var sampleRate: Int { config.sampleRate }
    private let tokenizer: Tokenizer
    private let stateLock = NSLock()
    private var network: FireRedTTS3Network?
    private var generating = false
    private var cacheLease = MLXCacheLimitCoordinator.shared.acquireDefault()

    private static let requiredFiles = [
        "config.json", "model.safetensors", "decoder.safetensors", "tokenizer.json", "tokenizer_config.json"
    ]

    private init(config: FireRedTTS3Config, directory: URL, network: FireRedTTS3Network, tokenizer: Tokenizer) {
        self.config = config; bundleDirectory = directory; self.network = network; self.tokenizer = tokenizer
    }

    /// Load an exported bundle without consulting a remote model registry.
    public static func fromBundle(_ directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> FireRedTTS3Model {
        try Task.checkCancellation()
        for file in requiredFiles where !FileManager.default.fileExists(atPath: directory.appendingPathComponent(file).path) {
            throw FireRedTTS3Error.missingFile(file)
        }
        let config = try FireRedTTS3Config.load(from: directory.appendingPathComponent("config.json"))
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let specialIDs = ["<|im_start|>":151644, "<|im_end|>":151645,
            "<think>":151667, "</think>":151668, "<|sosp|>":FireRedTTS3Prompt.audioStartID,
            "<|sot|>":151676, "<|eot|>":FireRedTTS3Prompt.textEndID]
        guard specialIDs.allSatisfy({ tokenizer.convertTokenToId($0.key) == $0.value }) else {
            throw FireRedTTS3Error.invalidConfiguration("tokenizer special IDs do not match the Instruct checkpoint")
        }
        let lease = MLXCacheLimitCoordinator.shared.acquireDefault()
        defer { lease?.release() }
        progressHandler?(0.1, "Loading FireRedTTS3 Instruct weights")
        var weights = try CommonWeightLoader.loadSafetensors(url: directory.appendingPathComponent("model.safetensors"))
        for (key, value) in try CommonWeightLoader.loadSafetensors(url: directory.appendingPathComponent("decoder.safetensors")) {
            guard weights[key] == nil else { throw FireRedTTS3Error.invalidWeights("duplicate \(key)") }
            weights[key] = value
        }
        let network = try FireRedTTS3Network(config: config, weights: weights)
        eval(Array(weights.values))
        try Task.checkCancellation()
        progressHandler?(1, "FireRedTTS3 Instruct ready")
        return FireRedTTS3Model(config: config, directory: directory, network: network, tokenizer: tokenizer)
    }

    /// A model identifier must be supplied for a compatible exported bundle.
    public static func fromPretrained(modelId: String, cacheDir: URL? = nil,
        offlineMode: Bool = false, downloadOptions: DownloadOptions = .init(),
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> FireRedTTS3Model {
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadFiles(modelId: modelId, to: directory, files: requiredFiles,
            offlineMode: offlineMode, downloadOptions: downloadOptions) { progress in
            progressHandler?(progress * 0.9, "Downloading FireRedTTS3 Instruct")
        }
        return try await fromBundle(directory) { progress, stage in
            progressHandler?(0.9 + progress * 0.1, stage)
        }
    }

    public var isLoaded: Bool { stateLock.withLock { network != nil } }
    public var memoryFootprint: Int { stateLock.withLock { network?.weightBytes ?? 0 } }

    func tokenIDs(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }

    /// An admitted generation retains its own weights and cache lease until it finishes.
    /// Later requests fail immediately after unload, including while that request finishes.
    public func unload() {
        stateLock.withLock {
            network = nil
            cacheLease?.release(); cacheLease = nil
        }
    }

    public func generate(text: String, language: String? = nil) async throws -> [Float] {
        try await generate(text: text, instruction: "A natural, clear speaking voice.", language: language)
    }

    public func generate(text: String, instruction: String, language: String? = nil,
        options: FireRedTTS3Options = .init()
    ) async throws -> [Float] {
        try await generateWithPlan(text: text, instruction: instruction, language: language, options: options).samples
    }

    public func generateStream(text: String, language: String? = nil) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let samples = try await self.generate(text: text, language: language)
                    try Task.checkCancellation()
                    continuation.yield(AudioChunk(samples: samples, sampleRate: self.sampleRate, frameIndex: 0, isFinal: true))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public func generateWithPlan(text: String, instruction: String, language: String? = nil,
        options: FireRedTTS3Options = .init()
    ) async throws -> FireRedTTS3Output {
        try options.validate()
        try Task.checkCancellation()
        if let language, !["en", "english", "zh", "chinese", "zh-cn", "mandarin"].contains(language.lowercased()) {
            throw FireRedTTS3Error.invalidInput("voice design supports English and Chinese; language is inferred from the script")
        }
        let prompt = try FireRedTTS3Prompt.build(text: text, instruction: instruction)
        let network: FireRedTTS3Network = try stateLock.withLock {
            guard let network = self.network else { throw FireRedTTS3Error.modelUnloaded }
            guard !generating else { throw FireRedTTS3Error.alreadyGenerating }
            generating = true
            return network
        }
        defer { stateLock.withLock { generating = false } }
        let ids = tokenIDs(prompt)
        guard !ids.isEmpty, ids.count <= 4096,
              ids.allSatisfy({ $0 >= 0 && $0 < config.backbone.vocabSize }) else {
            throw FireRedTTS3Error.invalidInput("prompt exceeds 4096 positions or contains invalid token IDs")
        }
        let lease = MLXCacheLimitCoordinator.shared.acquireDefault()
        defer { lease?.release() }
        try Task.checkCancellation()
        let random = MLXRandom.RandomState(seed: options.seed)
        var cache = Array<(MLXArray, MLXArray)?>(repeating: nil, count: config.backbone.numHiddenLayers)
        var offset = 0
        var input = network.embedding(ids)
        var plan = [Int]()
        let started = ProcessInfo.processInfo.systemUptime
        var planned = false
        for _ in 0..<options.maxPlanningTokens {
            try Task.checkCancellation()
            let hidden = try network.backbone(input, cache: &cache, offset: offset)
            offset += input.dim(1)
            let token = try Self.sample(network.textLogits(hidden[0..., -1, 0...]), options: options, random: random)
            input = network.embedding([token])
            Self.materialize(cache)
            if token == FireRedTTS3Prompt.textEndID { planned = true; break }
            plan.append(token)
        }
        guard planned else { throw FireRedTTS3Error.planningLimit }
        // The upstream checkpoint consumes EOT and SOS once here, then SOS a
        // second time in the first audio iteration. Keep that trained protocol.
        var hidden = try network.backbone(input, cache: &cache, offset: offset)
        offset += 1; eval(hidden); Self.materialize(cache)
        input = network.embedding([FireRedTTS3Prompt.audioStartID])
        hidden = try network.backbone(input, cache: &cache, offset: offset)
        offset += 1; eval(hidden); Self.materialize(cache)
        let planningSeconds = ProcessInfo.processInfo.systemUptime - started
        let acousticStart = ProcessInfo.processInfo.systemUptime
        let historyLength = config.historyPatches * config.patchSize
        var latentHistory = MLXArray.zeros([1, historyLength, config.latentDimension])
        // Zero *backbone* history goes through dit_head, including its bias.
        var conditions = MLXArray.zeros([1, config.historyPatches, config.backbone.hiddenSize])
        var patches = [MLXArray]()
        var stopped = false
        let times = (0...options.flowSteps).map { 1 - cos(Float($0) / Float(options.flowSteps) * .pi / 2) }
        for step in 0..<options.maxAudioPatches {
            try Task.checkCancellation()
            hidden = try network.backbone(input, cache: &cache, offset: offset)
            offset += input.dim(1)
            let last = hidden[0..., (hidden.dim(1)-1)..<hidden.dim(1), 0...]
            let score = sigmoid(network.linear(last, "stop_head")).item(Float.self)
            guard score.isFinite else { throw FireRedTTS3Error.invalidAudio }
            Self.materialize(cache)
            if score >= options.stopThreshold && step >= options.minAudioPatches { stopped = true; break }
            let appendedConditions = concatenated([conditions, last], axis: 1)
            conditions = appendedConditions[0..., (appendedConditions.dim(1)-config.historyPatches-1)..<appendedConditions.dim(1), 0...]
            let projected = network.linear(conditions, "dit_head")
            let repeated = MLX.broadcast(projected.expandedDimensions(axis: 2),
                to: [1, config.historyPatches + 1, config.patchSize, config.flow.hiddenSize])
                .reshaped([1, historyLength + config.patchSize, config.flow.hiddenSize])
            let noise = MLXRandom.normal([1, config.patchSize, config.latentDimension], dtype: .float32, key: random.next())
            var current = concatenated([latentHistory, noise], axis: 1)
            for i in 0..<options.flowSteps {
                try Task.checkCancellation()
                let positive = concatenated([current, repeated], axis: -1)
                let guided = options.guidanceScale > 0
                let flowInput = guided
                    ? concatenated([positive, concatenated([current, MLXArray.zeros(like: repeated)], axis: -1)], axis: 0)
                    : positive
                let velocity = network.flow(flowInput,
                    time: MLXArray(Array(repeating: times[i], count: guided ? 2 : 1)))
                let v = guided
                    ? (1 + options.guidanceScale) * velocity[0..<1] - options.guidanceScale * velocity[1..<2]
                    : velocity
                let tail = current[0..., historyLength..<(historyLength+config.patchSize), 0...]
                    + (times[i+1] - times[i]) * v[0..., historyLength..<(historyLength+config.patchSize), 0...]
                current = concatenated([latentHistory, tail], axis: 1)
                eval(current)
            }
            let patch = current[0..., historyLength..<(historyLength+config.patchSize), 0...]
            input = network.encodePatch(patch)
            latentHistory = current[0..., config.patchSize..<(config.patchSize+historyLength), 0...]
            eval(input, latentHistory, conditions)
            patches.append(patch)
        }
        guard stopped else { throw FireRedTTS3Error.audioLimit }
        guard !patches.isEmpty else { throw FireRedTTS3Error.invalidAudio }
        let acousticSeconds = ProcessInfo.processInfo.systemUptime - acousticStart
        try Task.checkCancellation()
        let decodingStart = ProcessInfo.processInfo.systemUptime
        let audio = try network.decode(concatenated(patches, axis: 1)).reshaped([-1])
        eval(audio)
        try Task.checkCancellation()
        let samples = audio.asArray(Float.self)
        guard samples.count == patches.count * config.patchSize * config.decoderUpsampleRate * config.audioPatchSize,
              !samples.isEmpty, samples.allSatisfy(\.isFinite) else { throw FireRedTTS3Error.invalidAudio }
        return FireRedTTS3Output(samples: samples, voiceDescription: tokenizer.decode(tokens: plan),
            planningSeconds: planningSeconds, acousticSeconds: acousticSeconds,
            decodingSeconds: ProcessInfo.processInfo.systemUptime - decodingStart, audioPatches: patches.count)
    }

    private static func materialize(_ cache: [(MLXArray, MLXArray)?]) {
        for value in cache { if let value { eval(value.0, value.1) } }
    }

    static func sample(_ logits: MLXArray, options: FireRedTTS3Options, random: MLXRandom.RandomState) throws -> Int {
        var values = logits.reshaped([-1])
        guard values.max().item(Float.self).isFinite else { throw FireRedTTS3Error.invalidAudio }
        if options.textTemperature == 0 { return argMax(values).item(Int.self) }
        values = values / options.textTemperature
        guard values.max().item(Float.self).isFinite else { throw FireRedTTS3Error.invalidAudio }
        if options.textTopK > 0 && options.textTopK < values.size {
            let cutoff = sorted(values)[values.size - options.textTopK]
            values = MLX.where(values .< cutoff, MLXArray(-Float.infinity), values)
        }
        if options.textTopP < 1 {
            let order = argSort(-values)
            let ordered = values[order]
            let probabilities = softmax(ordered)
            let filtered = MLX.where((cumsum(probabilities) - probabilities) .< options.textTopP,
                ordered, MLXArray(-Float.infinity))
            return order[MLXRandom.categorical(filtered, key: random.next()).item(Int.self)].item(Int.self)
        }
        return MLXRandom.categorical(values, key: random.next()).item(Int.self)
    }
}

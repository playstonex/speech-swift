import AudioCommon
@preconcurrency import CoreML
import Foundation
import SpeechRuntime

/// Native streaming Pocket TTS. All graphs default to CPU-only execution.
/// Calls share immutable weights; each request owns its noise and recurrent state.
public final class PocketTTSModel: SpeechGenerationModel, @unchecked Sendable {
    public static let defaultModelId = "aufklarer/Pocket-TTS-100M-CoreML-INT8"
    public static let defaultVoice = "alba"
    public let config: PocketTTSConfig
    public var sampleRate: Int { config.sampleRate }
    public var availableVoices: [String] { [Self.defaultVoice] }
    let inference = InferenceGate()
    let tokenizer: PocketTTSTokenizer
    let prefill: MLModel, step: MLModel, flow: MLModel, mimi: MLModel
    let prefillKeys: PocketTtsLayerKeys, stepKeys: PocketTtsLayerKeys, mimiKeys: PocketTtsMimiKeys
    let directory: URL
    let embeddings: [Float], bos: [Float]
    let voiceCache: [[Float]]

    private init(directory: URL, computeUnits: MLComputeUnits) throws {
        self.directory = directory
        config = try JSONDecoder().decode(PocketTTSConfig.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try config.validate()
        let constants = directory.appendingPathComponent("constants_bin")
        tokenizer = try PocketTTSTokenizer(data: Data(contentsOf: constants.appendingPathComponent("tokenizer.model")))
        embeddings = try Self.readFloats(constants.appendingPathComponent("text_embed_table.bin"), count: 4001*1024)
        bos = try Self.readFloats(constants.appendingPathComponent("bos_emb.bin"), count: 32)
        guard tokenizer.pieces.count == 4000 else { throw PocketTTSError.invalidConfiguration("expected English vocabulary") }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        func load(_ name: String) throws -> MLModel {
            try MLModel(contentsOf: directory.appendingPathComponent(name+".mlmodelc"), configuration: configuration)
        }
        prefill = try load("cond_prefill")
        step = try load("flowlm_stepv2")
        flow = try load("flow_decoder_fused")
        // Mimi's recurrent convolution state is validated on CPU only.
        let cpu = MLModelConfiguration(); cpu.computeUnits = .cpuOnly
        mimi = try MLModel(contentsOf: directory.appendingPathComponent("mimi_decoder.mlmodelc"), configuration: cpu)
        prefillKeys = try .discover(from: prefill, kind: .condStep, expectedLayers: 6, modelName: "cond_prefill")
        stepKeys = try .discover(from: step, kind: .flowlmStep, expectedLayers: 6, modelName: "flowlm_stepv2")
        mimiKeys = try .discover(from: mimi, staticInputs: Set(config.mimiStaticInputs ?? []))
        // Fail at load time for incomplete or malformed voice state.
        var voiceCache: [[Float]] = []
        for i in 0..<6 {
            voiceCache.append(try Self.readFloats(constants.appendingPathComponent("alba_voice_cache/cache\(i).bin"), count: 2*512*1024))
            let pos = try Self.readFloats(constants.appendingPathComponent("alba_voice_cache/position\(i).bin"), count: 1)
            guard pos == [Float(config.voicePosition)] else { throw PocketTTSError.invalidConfiguration("inconsistent voice position") }
        }
        self.voiceCache = voiceCache
    }

    public static func fromBundle(_ directory: URL, computeUnits: MLComputeUnits = .cpuOnly) throws -> PocketTTSModel {
        try PocketTTSModel(directory: directory, computeUnits: computeUnits)
    }
    public static func fromPretrained(modelId: String = defaultModelId, cacheDir: URL? = nil,
                                      offlineMode: Bool = false, computeUnits: MLComputeUnits = .cpuOnly,
                                      downloadOptions: DownloadOptions = .init(),
                                      progressHandler: ((Double, String) -> Void)? = nil) async throws -> PocketTTSModel {
        try Task.checkCancellation()
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(modelId: modelId, to: directory,
            additionalFiles: ["cond_prefill.mlmodelc/**", "flowlm_stepv2.mlmodelc/**", "flow_decoder_fused.mlmodelc/**",
                              "mimi_decoder.mlmodelc/**", "constants_bin/**", "NOTICE", "LICENSE", "README.md", "manifest.json"],
            offlineMode: offlineMode, downloadOptions: downloadOptions, progressHandler: { progressHandler?($0, "Downloading Pocket TTS") })
        try Task.checkCancellation()
        return try fromBundle(directory, computeUnits: computeUnits)
    }
    public static func validateInput(text: String, language: String? = nil, voice: String = "alba") throws {
        try validate(language: language, voice: voice)
        _ = try PocketTTSTokenizer.normalized(text)
    }
    public static func validate(language: String?, voice: String = "alba") throws {
        guard voice == "alba" else { throw PocketTTSError.invalidInput("only the Alba voice is supported") }
        guard language == nil || ["en", "english"].contains(language!.lowercased()) else {
            throw PocketTTSError.invalidInput("this bundle supports English only")
        }
    }
    public func generate(text: String, language: String?) async throws -> [Float] {
        try await generate(text: text, language: language, options: .init())
    }
    public func generate(text: String, language: String? = nil, options: PocketTTSOptions) async throws -> [Float] {
        var audio: [Float] = []
        try await generateStreaming(text: text, language: language, options: options) { audio.append(contentsOf: $0.samples) }
        return audio
    }
    /// Awaited delivery provides backpressure. Earlier PCM remains partial if a later step fails.
    public func generateStreaming(text: String, language: String? = nil, options: PocketTTSOptions = .init(),
                                  onChunk: (AudioChunk) async throws -> Void) async throws {
        let session = PocketSession(model: self, text: text, language: language, options: options)
        do {
            while let chunk = try await session.next() { try await onChunk(chunk) }
        } catch {
            await session.close()
            throw error
        }
    }
    public func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error> {
        generateStream(text: text, language: language, options: .init())
    }
    public func generateStream(text: String, language: String? = nil, options: PocketTTSOptions) -> AsyncThrowingStream<AudioChunk, Error> {
        let session = PocketSession(model: self, text: text, language: language, options: options)
        return AsyncThrowingStream(unfolding: { try await session.next() })
    }
    static func readFloats(_ url: URL, count: Int) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count == count*4 else { throw PocketTTSError.invalidConfiguration("invalid tensor size: \(url.lastPathComponent)") }
        let values = data.withUnsafeBytes { bytes in
            (0..<count).map { Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0*4, as: UInt32.self))) }
        }
        guard values.allSatisfy(\.isFinite) else { throw PocketTTSError.invalidConfiguration("nonfinite constant: \(url.lastPathComponent)") }
        return values
    }
}

// All CoreML arrays and prediction state are isolated to one session actor.
private actor PocketSession {
    let model: PocketTTSModel, text: String, language: String?, options: PocketTTSOptions
    var permit: InferencePermit?
    var chunks: [String] = []
    var initialized = false, finished = false
    var chunkIndex = 0, stepIndex = 0, outputIndex = 0
    var eosStep: Int?, tailFrames = 3
    var cache: [String: MLMultiArray] = [:], mimiState: [String: MLMultiArray] = [:]
    var sequence: [Float] = Array(repeating: .nan, count: 32)
    var rng: PocketRNG
    init(model: PocketTTSModel, text: String, language: String?, options: PocketTTSOptions) {
        self.model = model; self.text = text; self.language = language; self.options = options
        rng = PocketRNG(state: options.seed)
    }
    func close() {
        finished = true; cache.removeAll(); mimiState.removeAll()
        permit?.release(); permit = nil
    }
    func next() async throws -> AudioChunk? {
        guard !finished else { return nil }
        do {
            try Task.checkCancellation()
            if !initialized {
                try PocketTTSModel.validate(language: language)
                try options.validate()
                chunks = try model.tokenizer.chunks(text)
                permit = try await model.inference.acquire()
                try Task.checkCancellation()
                initialized = true
                // Conditioning allocates full KV tensors too. Drain these temporaries
                // per request, as well as the frame-level prediction temporaries.
                try autoreleasepool {
                    try initializeMimi()
                    try prepareChunk()
                }
            }
            // Predictions are synchronous and scoped, so autoreleased providers do not
            // accumulate across frames. Cancellation is checked between predictions.
            if let samples = try autoreleasepool(invoking: { try frame() }) {
                let frameIndex = outputIndex; outputIndex += 1
                return AudioChunk(samples: samples, sampleRate: 24000, frameIndex: frameIndex, isFinal: false)
            }
            close()
            return AudioChunk(samples: [], sampleRate: 24000, frameIndex: outputIndex, isFinal: true)
        } catch { close(); throw error }
    }
    private func initializeMimi() throws {
        for (name, shape) in model.mimiKeys.stateShapes {
            let count = shape.reduce(1, *)
            mimiState[name] = try array([Float](repeating: 0, count: count), shape)
        }
    }
    private func prepareChunk() throws {
        for i in 0..<6 {
            cache["cache\(i)"] = try array(model.voiceCache[i], [2,1,512,16,64])
            cache["position\(i)"] = try array([Float(model.config.voicePosition)], [1])
        }
        var text = chunks[chunkIndex]
        if let first = text.first { text = first.uppercased() + text.dropFirst() }
        if let last = text.last, !".!?".contains(last) { text += "." }
        let short = text.split(separator: " ").count < 5
        if short { text = String(repeating: " ", count: 8) + text }
        tailFrames = short ? 5 : 3
        let ids = try model.tokenizer.encode(text)
        guard ids.count <= 256, model.config.voicePosition+ids.count < 512 else {
            throw PocketTTSError.invalidConfiguration("conditioning exceeds cache capacity")
        }
        var embedding = [Float](repeating: 0, count: 256*1024)
        for (index, id) in ids.enumerated() {
            guard id < 4001 else { throw PocketTTSError.invalidConfiguration("token out of range") }
            embedding.replaceSubrange(index*1024..<(index+1)*1024, with: model.embeddings[id*1024..<(id+1)*1024])
        }
        var inputs = cache
        inputs["conditioning"] = try array(embedding,[1,256,1024])
        inputs["valid_len"] = try array([Float(ids.count)],[1])
        let result = try predict(model.prefill, inputs)
        try updateCache(result, keys: model.prefillKeys)
        stepIndex = 0; eosStep = nil; sequence = Array(repeating: .nan, count: 32)
    }
    private func frame() throws -> [Float]? {
        while chunkIndex < chunks.count {
            try Task.checkCancellation()
            if let eosStep, stepIndex >= eosStep+tailFrames {
                chunkIndex += 1
                if chunkIndex == chunks.count { return nil }
                try prepareChunk()
            }
            guard stepIndex < options.maxFrames, let position = cache["position0"], position[0].intValue < 512 else {
                throw PocketTTSError.generationLimit
            }
            var inputs = cache
            inputs["sequence"] = try array(sequence,[1,1,32])
            inputs["bos_emb"] = try array(model.bos,[32])
            let result = try predict(model.step,inputs)
            try updateCache(result,keys: model.stepKeys)
            let eos = try output(result,model.stepKeys.eosLogit!)[0].floatValue
            guard eos.isFinite else { throw PocketTTSError.processingFailed("nonfinite EOS") }
            if eos > -4 && eosStep == nil { eosStep = stepIndex }
            if let eosStep, stepIndex >= eosStep+tailFrames { continue }
            let hidden = try output(result,model.stepKeys.transformerOut!)
            let noise = (0..<32).map { _ in rng.gaussian()*sqrt(options.temperature) }
            let flowResult = try predict(model.flow,["transformer_out": try array(floats(hidden),[1,1024]),
                                                    "latent_init": try array(noise,[1,32])])
            sequence = try floats(output(flowResult,"latent_final"))
            guard sequence.count == 32, sequence.allSatisfy(\.isFinite) else { throw PocketTTSError.processingFailed("invalid flow latent") }
            try Task.checkCancellation()
            var mimiInputs = mimiState
            mimiInputs["latent"] = try array(sequence,[1,32])
            let decoded = try predict(model.mimi,mimiInputs)
            let samples = try floats(output(decoded,model.mimiKeys.audioOutput))
            guard samples.count == 1920, samples.allSatisfy(\.isFinite) else { throw PocketTTSError.processingFailed("invalid PCM frame") }
            for (input, outputName) in model.mimiKeys.stateMapping { mimiState[input] = try output(decoded,outputName) }
            stepIndex += 1
            return samples
        }
        return nil
    }
    private func updateCache(_ result: MLFeatureProvider, keys: PocketTtsLayerKeys) throws {
        for i in 0..<6 {
            cache["cache\(i)"] = try output(result,keys.cacheKeys[i])
            cache["position\(i)"] = try output(result,keys.positionKeys[i])
        }
    }
    private func predict(_ model: MLModel, _ inputs: [String: MLMultiArray]) throws -> MLFeatureProvider {
        try Task.checkCancellation()
        return try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
    }
    private func output(_ result: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
        guard let value = result.featureValue(for: name)?.multiArrayValue else { throw PocketTTSError.processingFailed("missing output \(name)") }
        return value
    }
    private func array(_ values: [Float], _ shape: [Int]) throws -> MLMultiArray {
        guard shape.reduce(1, *) == values.count else { throw PocketTTSError.processingFailed("tensor shape mismatch") }
        let array = try MLMultiArray(shape: shape.map(NSNumber.init),dataType: .float32)
        if !values.isEmpty { values.withUnsafeBufferPointer { array.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: values.count*4) } }
        return array
    }
    private func floats(_ array: MLMultiArray) -> [Float] { (0..<array.count).map { array[$0].floatValue } }
}

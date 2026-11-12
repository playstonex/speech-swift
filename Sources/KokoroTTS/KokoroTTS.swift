import CoreML
import Foundation
import AudioCommon

/// Kokoro-82M text-to-speech — CoreML-based, runs on Neural Engine.
///
/// Lightweight (82M params) non-autoregressive TTS model.
/// Supports 10 languages with 54 preset voices. Designed for iOS/iPad deployment.
///
/// German is served by separate single-voice bundles (the kikiri-tts
/// fine-tunes): load one with `fromPretrained(modelId: "aufklarer/Kokoro-82M-German-Martin-CoreML")`
/// (or `…-Victoria-CoreML`) and synthesize with `language: "de"`.
///
/// Uses an end-to-end CoreML model (`kokoro_5s.mlmodelc`) that runs the full
/// pipeline (BERT → duration → alignment → prosody → decoder) in one call.
///
/// ```swift
/// let tts = try await KokoroTTSModel.fromPretrained()
/// let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
/// ```
public final class KokoroTTSModel {

    /// Default HuggingFace model ID.
    public static let defaultModelId = "aufklarer/Kokoro-82M-CoreML"
    /// German voice bundles: one Kokoro checkpoint per voice, each with its
    /// single `dm_martin` / `df_victoria` preset and `dict_de.json`.
    public static let germanMartinModelId = "aufklarer/Kokoro-82M-German-Martin-CoreML"
    public static let germanVictoriaModelId = "aufklarer/Kokoro-82M-German-Victoria-CoreML"
    /// Default voice preset.
    public static let defaultVoice = "af_heart"
    /// Output sample rate.
    public static let outputSampleRate = 24000

    let config: KokoroConfig
    var network: KokoroNetwork?
    let phonemizer: KokoroPhonemizer
    var voiceEmbeddings: [String: [Float]]

    var _isLoaded: Bool { network != nil }

    init(config: KokoroConfig, network: KokoroNetwork, phonemizer: KokoroPhonemizer, voiceEmbeddings: [String: [Float]]) {
        self.config = config
        self.network = network
        self.phonemizer = phonemizer
        self.voiceEmbeddings = voiceEmbeddings
    }

    // MARK: - Synthesis

    /// Input slots of the end-to-end graph (`input_ids` is `[1, 128]`).
    static let inputSlots = 128
    /// Token budget per chunk, as in speech-core's 5.0 s Kokoro profile. The
    /// packed-LSTM models speak at upstream Kokoro's rate, about 10.5 phoneme
    /// tokens per second, so a chunk this size stays inside the 5 s graph at
    /// speed 1; the earlier 72 overflowed and split-and-retried.
    static let chunkTokenBudget = 56
    /// No chunk grows past this many tokens, the hard cap of the same
    /// speech-core profile. A merged chunk near the 128-token input would
    /// overrun the 5 s graph at this speech rate and render twice.
    static let chunkTokenHardCap = 64
    /// A last chunk shorter than this merges into the previous one when the
    /// hard cap allows it; very short utterances render unreliably.
    static let minTailTokens = 12
    /// Pieces of an overflow retry never go below this many tokens.
    static let minRetryTokens = 6
    /// How many times one chunk may be split again after an overflow.
    static let maxRetryDepth = 4
    /// The last 200 ms of the bounded graph's output window are not trusted,
    /// as in speech-core: speech reaching into them means the chunk was too
    /// long and is split and rendered again.
    static let outputTailGuardSamples = 4800

    /// Synthesize speech from text.
    ///
    /// The text is split into chunks of up to 56 phoneme tokens, about what
    /// the 5 s graph holds at the models' speech rate, preferring sentence
    /// ends, then clauses, then words. Each chunk renders in one call and the
    /// audio is concatenated, so short text is a single call. A chunk whose
    /// speech would still overrun the graph's output window is split again
    /// and rendered as two pieces.
    public func synthesize(
        text: String,
        voice: String = "af_heart",
        language: String = "en",
        speed: Float = 1.0
    ) throws -> [Float] {
        var audio: [Float] = []
        try synthesizeChunks(text: text, voice: voice, language: language, speed: speed) { samples, _ in
            audio.append(contentsOf: samples)
        }
        return audio
    }

    /// Renders `text` chunk by chunk and hands each piece's audio to
    /// `onChunk` as soon as it is ready. Exactly one call, the last, has
    /// `isFinal` set; text with nothing to speak yields one empty final piece.
    func synthesizeChunks(
        text: String,
        voice: String,
        language: String,
        speed: Float,
        onChunk: ([Float], Bool) throws -> Void
    ) throws {
        guard _isLoaded, let network else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-synthesize", reason: "Model not loaded")
        }
        guard let styleVector = voiceEmbeddings[voice] else {
            let available = Array(voiceEmbeddings.keys).sorted().prefix(5)
            throw AudioModelError.voiceNotFound(
                voice: voice,
                searchPath: "Available: \(available.joined(separator: ", "))...")
        }

        // Packing measures many overlapping candidates and rendering needs the
        // ids of the pieces it kept; phonemize each piece of text once.
        var tokenCache: [String: [Int]] = [:]
        let tokenIds: (String) -> [Int] = { piece in
            if let ids = tokenCache[piece] { return ids }
            let ids = self.phonemizer.tokenize(piece, maxLength: .max, language: language)
            tokenCache[piece] = ids
            return ids
        }
        // Chunk up front, as speech-core does. Text within the budget is one
        // chunk and renders in one call; rendering longer text whole first
        // would only overrun the 5 s window and cost a discarded pass.
        let chunks = Self.chunkPlan(
            text, countTokens: { tokenIds($0).count },
            capacity: min(config.maxPhonemeLength, Self.inputSlots))

        guard !chunks.isEmpty else {
            try onChunk([], true)
            return
        }
        var emitted = false
        for (index, chunk) in chunks.enumerated() {
            // An overflow retry can turn one chunk into several pieces; they
            // come back together, so the final piece is known when emitted.
            let pieces = try renderChunk(
                chunk, network: network, style: styleVector, speed: speed,
                tokenIds: tokenIds, depth: 0)
            for (pieceIndex, var piece) in pieces.enumerated() {
                // A short fade-in keeps the seam between pieces click-free.
                if emitted { fadeIn(&piece) }
                let isFinal = index == chunks.count - 1 && pieceIndex == pieces.count - 1
                try onChunk(piece, isFinal)
                emitted = true
            }
        }
    }

    /// The chunks `text` renders as: packed to `chunkTokenBudget`, with a
    /// short last chunk merged only within `chunkTokenHardCap`.
    static func chunkPlan(
        _ text: String,
        countTokens: (String) -> Int,
        capacity: Int = inputSlots
    ) -> [String] {
        KokoroTextChunker.chunk(
            text, countTokens: countTokens, maxTokens: chunkTokenBudget,
            hardCapTokens: min(chunkTokenHardCap, capacity), minTailTokens: minTailTokens)
    }

    /// Renders one chunk, splitting it again while its speech would overrun
    /// the graph's output window.
    private func renderChunk(
        _ text: String,
        network: KokoroNetwork,
        style: [Float],
        speed: Float,
        tokenIds: (String) -> [Int],
        depth: Int
    ) throws -> [[Float]] {
        let ids = tokenIds(text)
        let capacity = min(config.maxPhonemeLength, Self.inputSlots)
        var unsplit: [Float]?
        if ids.count <= capacity {
            let pass = try runModel(tokenIds: ids, network: network, style: style, speed: speed)
            if !pass.overran { return [pass.audio] }
            unsplit = pass.audio
        }
        if depth < Self.maxRetryDepth, ids.count > Self.minRetryTokens * 2 {
            let pieces = KokoroTextChunker.splitForRetry(
                text, countTokens: { tokenIds($0).count },
                minTokens: Self.minRetryTokens, maxTokens: ids.count - 1)
            if pieces.count == 2 {
                AudioLog.inference.info("Kokoro: splitting a \(ids.count)-token chunk that overran the output window")
                return try pieces.flatMap {
                    try renderChunk(
                        $0, network: network, style: style, speed: speed,
                        tokenIds: tokenIds, depth: depth + 1)
                }
            }
        }
        // No safe split exists: keep what one bounded pass produces.
        if let unsplit { return [unsplit] }
        let clipped = Array(ids.prefix(capacity - 1)) + [phonemizer.eosId]
        return [try runModel(tokenIds: clipped, network: network, style: style, speed: speed).audio]
    }

    /// One pass of the end-to-end graph. `overran` is set when the predicted
    /// speech reaches the untrusted tail of the output window.
    private func runModel(
        tokenIds: [Int],
        network: KokoroNetwork,
        style styleVector: [Float],
        speed: Float
    ) throws -> (audio: [Float], overran: Bool) {
        let padTo = Self.inputSlots
        let tokenCount = min(tokenIds.count, padTo)
        let paddedIds = phonemizer.pad(Array(tokenIds.prefix(padTo)), to: padTo)

        let inputIds = try createInt32Array(shape: [1, padTo], values: paddedIds.map { Int32($0) })
        let maskArray = try createInt32Array(shape: [1, padTo], values: (0..<padTo).map { Int32($0 < tokenCount ? 1 : 0) })
        let refS = try createFloatArray(shape: [1, config.styleDim], values: styleVector)
        let speedArray = try createFloatArray(shape: [1], values: [speed])

        let t0 = CFAbsoluteTimeGetCurrent()
        let result = try network.predictE2E(inputIds: inputIds, attentionMask: maskArray, refS: refS, speed: speedArray)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        let outputCapacity = result.audio.count
        let overran = outputCapacity > Self.outputTailGuardSamples
            && result.audioLengthSamples > outputCapacity - Self.outputTailGuardSamples
        let validSamples = min(result.audioLengthSamples, outputCapacity)
        guard validSamples > 0 else { return ([], overran) }

        var audio = [Float](repeating: 0, count: validSamples)
        if result.audio.dataType == .float16 {
            let ptr = result.audio.dataPointer.bindMemory(to: Float16.self, capacity: validSamples)
            for i in 0..<validSamples { audio[i] = Float(ptr[i]) }
        } else {
            let ptr = result.audio.dataPointer.bindMemory(to: Float.self, capacity: validSamples)
            for i in 0..<validSamples { audio[i] = ptr[i] }
        }

        // Kokoro's E2E model often emits 100–300 ms of trailing artifacts
        // past the real speech (random low-energy noise + an occasional
        // loud-spike click, observed on iPhone playback). Find where real
        // speech actually ended by walking backwards through 10 ms windows
        // and locating the last window above a silence-energy threshold —
        // then zero everything past that, plus a short ramp-down on the
        // last few ms of real speech to avoid a click at the new boundary.
        // Use a 50 ms window with a higher silence floor and require the
        // window to be SUSTAINED above threshold — Kokoro's trailing
        // artifacts are often a single 10–20 ms spike with low surrounding
        // energy, so a tight short window otherwise mistakes them for
        // speech. Real speech tails consistently above ~0.03 RMS over a
        // 50 ms span; isolated artifact spikes don't. Soft word endings
        // (German "Tisch", "Hamburg") fall under 0.030, so the floor is 0.015;
        // lower values only keep more tail.
        let win = max(1, Int(0.050 * Double(config.sampleRate)))
        let silenceRMS: Float = 0.015
        var speechEnd = validSamples
        var i = validSamples - win
        while i > 0 {
            var sumSq: Float = 0
            for j in 0..<win { let v = audio[i + j]; sumSq += v * v }
            let rms = sqrt(sumSq / Float(win))
            if rms > silenceRMS { speechEnd = i + win; break }
            i -= win / 2  // 50 % overlap so we don't miss a window boundary
        }
        // Zero the trailing artifact region.
        if speechEnd < validSamples {
            for k in speechEnd..<validSamples { audio[k] = 0 }
        }
        // Linear fade-out on the last ~10 ms of the kept signal so the
        // boundary between speech and the silenced region is also smooth.
        let fadeSamples = min(speechEnd, Int(0.010 * Double(config.sampleRate)))
        if fadeSamples >= 2 {
            let start = speechEnd - fadeSamples
            let denom = Float(fadeSamples - 1)
            for k in 0..<fadeSamples {
                let gain = Float(fadeSamples - 1 - k) / denom
                audio[start + k] *= gain
            }
        }
        // The vocoder is not bounded to +-1: the packed-LSTM models occasionally
        // overshoot by a few percent. Clamp so callers get valid float PCM.
        for k in 0..<speechEnd { audio[k] = min(1, max(-1, audio[k])) }

        let duration = Double(validSamples) / Double(config.sampleRate)
        let elapsedMs = elapsed * 1000
        AudioLog.inference.info("Kokoro E2E: \(tokenCount) tokens → \(validSamples) samples (\(String(format: "%.1f", duration))s) in \(String(format: "%.0f", elapsedMs))ms")
        return (audio, overran)
    }

    /// 5 ms linear fade-in at the start of a piece that follows another.
    private func fadeIn(_ audio: inout [Float]) {
        let count = min(audio.count, config.sampleRate / 200)
        guard count >= 2 else { return }
        for i in 0..<count { audio[i] *= Float(i) / Float(count) }
    }

    /// List available voice presets.
    public var availableVoices: [String] {
        Array(voiceEmbeddings.keys).sorted()
    }

    /// Registers custom `word -> IPA` pronunciations for proper nouns missing from
    /// the shipped dictionaries. Resolved ahead of the bundled dictionaries, suffix
    /// stemming, and the neural G2P fallback.
    ///
    /// Keys are matched lowercased and must be single words — a key containing a
    /// space, hyphen, or apostrophe is never looked up, because text is split on
    /// punctuation before resolution. The IPA must use symbols from the model
    /// vocabulary; anything else is dropped at tokenization.
    ///
    /// Entries apply to English only. Passing `language:` zh, ja, it, fr, es, pt,
    /// or hi to `synthesize` selects a phonemizer that never sees them.
    ///
    /// The lookup is unlocked, so register entries before `synthesize` starts
    /// running concurrently on this instance.
    public func addPronunciations(_ entries: [String: String]) {
        phonemizer.addPronunciations(entries)
    }

    // MARK: - Helpers

    private func createInt32Array(shape: [Int], values: [Int32]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { $0 as NSNumber }, dataType: .int32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<values.count { ptr[i] = values[i] }
        return arr
    }

    private func createFloatArray(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { $0 as NSNumber }, dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<values.count { ptr[i] = values[i] }
        return arr
    }

    // MARK: - Warmup

    /// Warm up CoreML model by running a dummy inference.
    ///
    /// A model's first prediction is several times slower than the ones after
    /// it. `fromPretrained` calls this before returning unless `warmUp` is
    /// false; a model loaded that way can be warmed up here later.
    public func warmUp() throws {
        let voice = voiceEmbeddings[Self.defaultVoice] != nil ? Self.defaultVoice : (availableVoices.first ?? Self.defaultVoice)
        _ = try? synthesize(text: "hello", voice: voice)
    }

    // MARK: - Model Loading

    /// Load a pretrained Kokoro model from HuggingFace.
    ///
    /// - Parameter computeUnits: Which hardware the main CoreML model runs on.
    ///   Defaults to `.all` (Neural Engine preferred). Pass `.cpuAndGPU` to bypass
    ///   the Neural Engine — useful as a fallback on platforms where the ANE
    ///   compiler produces incorrect output for this model.
    /// - Parameter warmUp: Run one short synthesis before returning, so the
    ///   first call is as fast as later ones. Pass `false` to load without it,
    ///   for example when the model is only being downloaded.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        voice: String = defaultVoice,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        computeUnits: MLComputeUnits = .all,
        downloadOptions: DownloadOptions = .init(),
        warmUp: Bool = true,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> KokoroTTSModel {
        AudioLog.modelLoading.info("Loading Kokoro model: \(modelId)")

        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Download E2E model + G2P + voice
        progressHandler?(0.0, "Downloading model...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: [
                "kokoro_5s.mlmodelc/**",
                "G2PEncoder.mlmodelc/**",
                "G2PDecoder.mlmodelc/**",
                "vocab_index.json",
                "g2p_vocab.json",
                "us_gold.json",
                "us_silver.json",
                "dict_de.json",
                "voices/*.json",
            ],
            offlineMode: offlineMode,
            downloadOptions: downloadOptions
        ) { fraction in
            progressHandler?(fraction * 0.7, "Downloading model...")
        }

        // Load vocabulary
        progressHandler?(0.72, "Loading vocabulary...")
        let vocabURL = cacheDir.appendingPathComponent("vocab_index.json")
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            throw AudioModelError.modelLoadFailed(modelId: modelId, reason: "vocab_index.json not found")
        }
        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocabURL)
        try phonemizer.loadDictionaries(from: cacheDir)
        try phonemizer.loadGermanDictionary(from: cacheDir)

        // Load G2P models
        progressHandler?(0.76, "Loading G2P models...")
        let g2pEncoderURL = cacheDir.appendingPathComponent("G2PEncoder.mlmodelc", isDirectory: true)
        let g2pDecoderURL = cacheDir.appendingPathComponent("G2PDecoder.mlmodelc", isDirectory: true)
        let g2pVocabURL = cacheDir.appendingPathComponent("g2p_vocab.json")
        if FileManager.default.fileExists(atPath: g2pEncoderURL.path) &&
           FileManager.default.fileExists(atPath: g2pDecoderURL.path) {
            try phonemizer.loadG2PModels(
                encoderURL: g2pEncoderURL, decoderURL: g2pDecoderURL, vocabURL: g2pVocabURL)
            AudioLog.modelLoading.debug("Loaded CoreML G2P encoder + decoder")
        }

        // Load voice embeddings
        progressHandler?(0.78, "Loading voice embeddings...")
        var voiceEmbeddings = [String: [Float]]()
        let voicesDir = cacheDir.appendingPathComponent("voices")
        if FileManager.default.fileExists(atPath: voicesDir.path) {
            let files = try FileManager.default.contentsOfDirectory(at: voicesDir, includingPropertiesForKeys: nil)
            for file in files where file.pathExtension == "json" {
                let voiceName = file.deletingPathExtension().lastPathComponent
                if let embedding = try? loadVoiceEmbedding(from: file, styleDim: KokoroConfig.default.styleDim) {
                    voiceEmbeddings[voiceName] = embedding
                }
            }
            AudioLog.modelLoading.debug("Loaded \(voiceEmbeddings.count) voice presets")
        }

        // Load E2E CoreML model
        progressHandler?(0.85, "Loading CoreML model...")
        let network = try KokoroNetwork(directory: cacheDir, computeUnits: computeUnits)
        AudioLog.modelLoading.debug("Loaded Kokoro E2E model")

        let model = KokoroTTSModel(
            config: .default, network: network,
            phonemizer: phonemizer, voiceEmbeddings: voiceEmbeddings)

        // The first prediction pays CoreML's one-time setup; take it here
        // rather than on the caller's first reply.
        if warmUp {
            progressHandler?(0.95, "Warming up...")
            try model.warmUp()
        }

        progressHandler?(1.0, "Model loaded")
        AudioLog.modelLoading.info("Kokoro model loaded successfully")
        return model
    }

    /// Load voice embedding from JSON file.
    private static func loadVoiceEmbedding(from url: URL, styleDim: Int) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let embedding = json["embedding"] as? [Double] else { return [] }
        return embedding.prefix(styleDim).map { Float($0) }
    }
}

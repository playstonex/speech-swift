import Foundation
import MLX

/// Autoregressive state shared by buffered and streaming synthesis, owned by
/// one request under the model's inference permit.
struct VoxtralFrameGenerator {
    private let network: VoxtralTTSNetwork
    private let options: VoxtralTTSOptions
    private let offsets: MLXArray
    private var hidden: MLXArray
    private var language: VoxtralTTSLanguageState
    private var step = 0
    private var ended = false

    init(network: VoxtralTTSNetwork, config: VoxtralTTSConfig, prompt: [Int],
         voiceURL: URL, voiceFrames: Int, options: VoxtralTTSOptions) throws {
        self.network = network
        self.options = options
        offsets = MLXArray(config.audio.offsets).reshaped(1, -1)
        let weights = try MLX.loadArrays(url: voiceURL)
        guard let voice = weights["embedding"], voice.shape == [voiceFrames, config.transformer.dim] else {
            throw VoxtralTTSError.invalidCheckpoint("preset voice embedding shape mismatch")
        }
        let ids = MLXArray(prompt.map(Int32.init)).reshaped(1, -1)
        let embeddings = network.languageModel.embed(ids)
        let conditioned = concatenated([
            embeddings[0..., 0..<2, 0...],
            voice.asType(embeddings.dtype).expandedDimensions(axis: 0),
            embeddings[0..., (voiceFrames + 2)..., 0...],
        ], axis: 1)
        (hidden, language) = network.languageModel.forward(embeddings: conditioned,
            state: .initial(layerCount: config.transformer.layers))
    }

    mutating func next() throws -> MLXArray? {
        try Task.checkCancellation()
        if ended { return nil }
        guard step < options.maxFrames else { throw VoxtralTTSError.generationLimit(options.maxFrames) }
        // Frame zero is predicted directly from the final prompt hidden state.
        let codes = try network.acoustic.decodeFrame(hidden: hidden[0..., -1, 0...], options: options, frame: step)
        eval(codes)
        step += 1
        if codes[0, 0].item(Int32.self) <= 1 {
            ended = true
            guard step > 1 else { throw VoxtralTTSError.invalidCheckpoint("model ended without generating speech") }
            return nil
        }
        let feedback = sum(network.audioEmbeddings.embeddings(codes + offsets), axis: 1, keepDims: true)
        (hidden, language) = network.languageModel.forward(embeddings: feedback, state: language)
        eval(hidden)
        return codes.expandedDimensions(axis: 1)
    }
}

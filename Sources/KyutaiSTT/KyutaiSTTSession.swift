import Foundation
import MLX
import PersonaPlex

/// A text token's approximate audio position, corrected for the trained text delay.
public struct KyutaiSTTToken: Sendable, Equatable {
    public let id: Int
    public let text: String
    public let timestamp: Double
}

/// Incremental decoder for mono 24 kHz Float32 PCM. Calls must be serialized.
/// Arbitrary chunk sizes are buffered into 1,920-sample frames; retained audio and
/// attention state are bounded independently of the recording's duration.
public final class KyutaiSTTSession {
    private let model: KyutaiSTTModel
    private let id: UUID
    let decoderCache: [KyutaiSTTCache]
    let encoderCache: [KyutaiSTTCache]
    private var tokenizer: KyutaiSTTTokenizer
    private var previousAudio: MLXArray
    private var previousText: Int
    private var position = 0
    private var pending: [Float] = []
    private var rawText = ""
    private var started = false
    private var finished = false
    public private(set) var receivedSamples = 0
    public var text: String { rawText.trimmingCharacters(in: .whitespacesAndNewlines) }

    init(model: KyutaiSTTModel, id: UUID) {
        self.model = model
        self.id = id
        // The attention window includes the current frame.
        self.decoderCache = (0..<model.config.numLayers).map { _ in
            KyutaiSTTCache(capacity: model.config.context)
        }
        self.encoderCache = (0..<model.mimi.cfg.transformer.numLayers).map { _ in
            KyutaiSTTCache(capacity: model.mimi.cfg.transformer.context)
        }
        self.tokenizer = KyutaiSTTTokenizer(vocabulary: model.vocabulary)
        self.previousAudio = MLXArray.ones([1, model.config.numCodebooks, 1], dtype: .int32) * Int32(model.config.card)
        self.previousText = model.config.textCard
        pending.reserveCapacity(KyutaiSTTConfig.frameSize)
    }

    deinit { model.releaseSession(id) }

    /// Append samples and return newly available text pieces. Call `finish()` at end of input.
    public func append(_ samples: [Float]) throws -> [KyutaiSTTToken] {
        guard !finished else { throw KyutaiSTTError.sessionFinished }
        guard samples.allSatisfy(\.isFinite) else {
            throw KyutaiSTTError.invalidAudio("samples must be finite")
        }
        guard !samples.isEmpty else { return [] }
        var output: [KyutaiSTTToken] = []
        if !started {
            started = true
            for _ in 0..<model.config.prefixFrames {
                if let token = process(Array(repeating: 0, count: KyutaiSTTConfig.frameSize)) { output.append(token) }
            }
        }
        receivedSamples += samples.count
        var cursor = 0
        while cursor < samples.count {
            let count = min(KyutaiSTTConfig.frameSize - pending.count, samples.count - cursor)
            pending.append(contentsOf: samples[cursor..<(cursor + count)])
            cursor += count
            if pending.count == KyutaiSTTConfig.frameSize {
                if let token = process(pending) { output.append(token) }
                pending.removeAll(keepingCapacity: true)
            }
        }
        return output
    }

    /// Pads a partial audio frame and supplies silence to complete delayed words.
    /// Repeated calls return no further tokens.
    public func finish() throws -> [KyutaiSTTToken] {
        guard !finished else { return [] }
        defer { cancel() }
        guard started else { return [] }
        var output: [KyutaiSTTToken] = []
        if !pending.isEmpty {
            pending.append(contentsOf: repeatElement(0, count: KyutaiSTTConfig.frameSize - pending.count))
            if let token = process(pending) { output.append(token) }
            pending.removeAll(keepingCapacity: true)
        }
        let silence = Array(repeating: Float(0), count: KyutaiSTTConfig.frameSize)
        for _ in 0..<model.config.flushFrames {
            if let token = process(silence) { output.append(token) }
        }
        return output
    }

    public func cancel() {
        guard !finished else { return }
        finished = true
        pending.removeAll()
        for cache in decoderCache + encoderCache { cache.trim(cache.offset) }
        model.releaseSession(id)
    }

    private func process(_ pcm: [Float]) -> KyutaiSTTToken? {
        let audioTokens = model.encodeFrame(pcm, cache: encoderCache)
        // Match LMGen: at step zero use initial embeddings; at later steps use
        // the previous frame's audio and the previous generated text token.
        let logits = model.decoder.step(
            text: previousText, audio: previousAudio, cache: decoderCache, position: position)
        let nextToken = logits.argMax(axis: -1)
        // Submit independent codec and decoder work together, then synchronize once.
        eval(nextToken, audioTokens)
        let tokenID = nextToken.item(Int.self)
        previousAudio = audioTokens
        previousText = tokenID
        let timestamp = max(0, Double(position - model.config.prefixFrames) / KyutaiSTTConfig.frameRate
                            - model.config.stt.audioDelaySeconds)
        position += 1
        let piece = tokenizer.append(tokenID)
        guard !piece.isEmpty else { return nil }
        rawText += piece
        return KyutaiSTTToken(id: tokenID, text: piece, timestamp: timestamp)
    }
}

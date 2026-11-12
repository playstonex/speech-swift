import Foundation
import MLX

extension Drama3Network {
    func generate(_ input: FishAudioModelInput, options: FishAudioDrama3Options,
                  memoryOptions: FishAudioMemoryOptions = .unrestricted) throws -> FishAudioGeneratedCodebooks {
        let generator = try Drama3FrameGenerator(network: self, input: input, options: options, memoryOptions: memoryOptions)
        var codes = Array(repeating: [Int](), count: config.tower.codebookCount)
        while let frame = try generator.next() {
            for i in codes.indices { codes[i].append(frame[i]) }
        }
        return FishAudioGeneratedCodebooks(codes: codes)
    }
}

/// One utterance's frame iterator, shared by buffered and incremental synthesis.
final class Drama3FrameGenerator {
    private let network: Drama3Network
    private let options: FishAudioDrama3Options
    private let memoryOptions: FishAudioMemoryOptions
    private let backboneCaches: [Drama3Cache]
    private let towerCaches: [Drama3Cache]
    private var h: MLXArray
    private var frame: MLXArray
    private var history: [Int]
    private var sampler: Drama3Sampler
    private var count = 0
    private var finished = false

    init(network: Drama3Network, input: FishAudioModelInput, options: FishAudioDrama3Options,
         memoryOptions: FishAudioMemoryOptions = .unrestricted) throws {
        try options.validate()
        let config = network.config
        guard input.rowCount == config.tower.codebookCount + 1, input.tokenCount > 0,
              input.tokenCount <= FishAudioDrama3Prompt.maximumRows,
              input.tokenCount + options.maxFrames <= config.text.maxSequenceLength else {
            throw FishAudioError.invalidInput("Drama 3 prompt does not fit the context")
        }
        let frames = input.asMLXArray().transposed(0, 2, 1)
        let backboneCaches = (0..<config.text.layerCount).map { _ in Drama3Cache() }
        let towerCaches = (0..<(network.tower.sharedLayers.count + network.tower.residualLayers.count)).map { _ in Drama3Cache() }
        var lastHidden: MLXArray?
        for start in stride(from: 0, to: input.tokenCount, by: 128) {
            try Task.checkCancellation()
            let end = min(start + 128, input.tokenCount)
            let h = network.hidden(frames[0..., start..<end, 0...], caches: backboneCaches)
            let pairedEnd = min(end, input.tokenCount - 1)
            if pairedEnd > start {
                network.tower.prefill(h[0..., 0..<(pairedEnd - start), 0...], frames: frames[0..., start..<pairedEnd, 0...],
                                      next: frames[0..., (start + 1)..<(pairedEnd + 1), 0...], caches: towerCaches)
            }
            eval([h] + backboneCaches.flatMap(\.arrays) + towerCaches.flatMap(\.arrays))
            memoryOptions.trimCache()
            lastHidden = h[0..., (h.dim(1) - 1)..., 0...]
        }
        self.network = network
        self.options = options
        self.memoryOptions = memoryOptions
        self.backboneCaches = backboneCaches
        self.towerCaches = towerCaches
        h = lastHidden!
        frame = frames[0..., (input.tokenCount - 1)..., 0...]
        history = input.rows[0].map(Int.init)
        sampler = Drama3Sampler(state: options.seed)
    }

    func next() throws -> [Int]? {
        try Task.checkCancellation()
        if finished { return nil }
        guard count < options.maxFrames else { throw FishAudioError.generationLimit(options.maxFrames) }
        let config = network.config
        let eosIndex = config.tower.vocabularySize
        let logits = network.semanticHead(h).asType(.float32).asArray(Float.self)
        let suppressed = count < options.minFrames ? eosIndex : nil
        var main = try sampler.sample(logits, temperature: options.temperature, topK: options.topK, topP: options.topP, suppressed: suppressed)
        if main != eosIndex, history.suffix(16).contains(config.semanticStartTokenId + main), options.temperature > 0 {
            main = try sampler.sample(logits, temperature: 1, topK: 30, topP: 0.95, suppressed: suppressed)
        }
        if main == eosIndex {
            guard count > 0 else { throw FishAudioError.invalidInput("Drama 3 generated no audio") }
            finished = true
            return nil
        }
        let next = try network.tower.decode(h, code: main, frame: frame, caches: towerCaches) { logits in
            try sampler.sample(logits.asType(.float32).asArray(Float.self), temperature: options.temperature,
                               topK: options.topK, topP: options.topP)
        }
        let token = config.semanticStartTokenId + main
        history.append(token)
        frame = MLXArray(([token] + next).map(Int32.init)).reshaped(1, 1, 1 + next.count)
        h = network.hidden(frame, caches: backboneCaches)
        eval([h] + backboneCaches.flatMap(\.arrays) + towerCaches.flatMap(\.arrays))
        memoryOptions.trimCache()
        count += 1
        return next
    }
}

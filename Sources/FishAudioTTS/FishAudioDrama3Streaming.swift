import AudioCommon
import Foundation
import SpeechRuntime

extension FishAudioDrama3Model {
    /// Deliver new PCM while generating codebooks. Awaiting the callback applies
    /// backpressure and retains the inference permit through the final delivery.
    /// Errors after partial output throw without a successful final marker.
    public func generateStreaming(
        text: String, references: [FishAudioReferencePrompt] = [], history: [FishAudioReferencePrompt] = [],
        options: FishAudioDrama3Options = .init(), onChunk: (AudioChunk) async throws -> Void
    ) async throws {
        let state = Drama3StreamState(model: self, text: text, references: references, history: history, options: options)
        do {
            while let chunk = try await state.next() {
                try await onChunk(chunk)
                try Task.checkCancellation()
            }
        } catch {
            await state.close()
            throw error
        }
    }

    public func generateStreaming(
        text: String, referenceAudio: [Float], referenceSampleRate: Int, referenceText: String,
        options: FishAudioDrama3Options = .init(), onChunk: (AudioChunk) async throws -> Void
    ) async throws {
        let start = ProcessInfo.processInfo.systemUptime
        try FishAudioDrama3Prompt.validateText(text)
        try options.validate()
        let reference = try await encodeReferencePrompt(audio: referenceAudio, sampleRate: referenceSampleRate, text: referenceText)
        try await generateStreaming(text: text, references: [reference], options: options) { chunk in
            try await onChunk(AudioChunk(samples: chunk.samples, sampleRate: chunk.sampleRate,
                frameIndex: chunk.frameIndex, isFinal: chunk.isFinal,
                elapsedTime: ProcessInfo.processInfo.systemUptime - start, textTokens: chunk.textTokens))
        }
    }

    public func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error> {
        generateStream(text: text, options: .init())
    }

    /// Pull-driven decoding: no background producer or queued PCM. Releasing the
    /// stream and its iterator releases request state and admission. Input text must be complete.
    public func generateStream(
        text: String, references: [FishAudioReferencePrompt] = [], history: [FishAudioReferencePrompt] = [],
        options: FishAudioDrama3Options = .init()
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        let state = Drama3StreamState(model: self, text: text, references: references, history: history, options: options)
        return AsyncThrowingStream(unfolding: {
            let chunk = try await state.next()
            if chunk?.isFinal == true { await state.close() }
            return chunk
        })
    }
}

private actor Drama3StreamState {
    let model: FishAudioDrama3Model
    let text: String
    let references: [FishAudioReferencePrompt]
    let history: [FishAudioReferencePrompt]
    let options: FishAudioDrama3Options
    var permit: InferencePermit?
    var generator: Drama3FrameGenerator?
    var decoder = FishAudioCodecStreamingState()
    var emittedFrames = 0
    var start: Double?
    var finished = false
    var advancing = false

    init(model: FishAudioDrama3Model, text: String, references: [FishAudioReferencePrompt],
         history: [FishAudioReferencePrompt], options: FishAudioDrama3Options) {
        self.model = model; self.text = text; self.references = references
        self.history = history; self.options = options
    }

    deinit { permit?.release() }

    func next() async throws -> AudioChunk? {
        guard !advancing else { throw FishAudioError.invalidInput("A Drama 3 stream supports one consumer") }
        advancing = true
        defer { advancing = false }
        if finished { close(); return nil }
        do {
            try Task.checkCancellation()
            if permit == nil {
                start = ProcessInfo.processInfo.systemUptime
                let input = try model.validatedInput(text: text, references: references, history: history, options: options)
                permit = try await model.inference.acquire()
                try Task.checkCancellation()
                generator = try Drama3FrameGenerator(network: model.network, input: input, options: options,
                                                     memoryOptions: model.memoryOptions)
            }
            var codes = Array(repeating: [Int](), count: model.config.tower.codebookCount)
            while codes[0].count < options.streamingChunkFrames {
                guard let frame = try generator!.next() else { finished = true; break }
                for i in codes.indices { codes[i].append(frame[i]) }
            }
            let samples = try model.codec.decodeStreaming(.init(codes: codes), state: &decoder)
            model.memoryOptions.trimCache()
            try Task.checkCancellation()
            let chunk = AudioChunk(samples: samples, sampleRate: model.sampleRate, frameIndex: emittedFrames,
                                   isFinal: finished, elapsedTime: ProcessInfo.processInfo.systemUptime - start!)
            emittedFrames += codes[0].count
            // Exact chunk multiples finish with an empty marker after EOS.
            // Admission remains held until the final callback returns.
            return chunk
        } catch {
            close()
            throw error
        }
    }

    func close() {
        finished = true
        generator = nil
        decoder = FishAudioCodecStreamingState()
        model.memoryOptions.trimCache()
        permit?.release()
        permit = nil
    }
}

import AudioCommon

extension KokoroTTSModel: SpeechGenerationModel {
    public var sampleRate: Int { config.sampleRate }

    /// The bundle's default voice: `af_heart` when present, otherwise the
    /// bundle's single voice (the German bundles ship one voice each).
    public var defaultBundleVoice: String {
        voiceEmbeddings[Self.defaultVoice] != nil ? Self.defaultVoice : (availableVoices.first ?? Self.defaultVoice)
    }

    public func generate(text: String, language: String?) async throws -> [Float] {
        try synthesize(text: text, voice: defaultBundleVoice, language: language ?? "en")
    }

    public func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // Kokoro is non-autoregressive, but long text is rendered
                    // chunk by chunk; each chunk is yielded as soon as it is
                    // ready, so playback can start after the first sentence.
                    var frameIndex = 0
                    try self.synthesizeChunks(
                        text: text, voice: self.defaultBundleVoice,
                        language: language ?? "en", speed: 1.0
                    ) { samples, isFinal in
                        continuation.yield(AudioChunk(
                            samples: samples,
                            sampleRate: self.config.sampleRate,
                            frameIndex: frameIndex,
                            isFinal: isFinal
                        ))
                        frameIndex += samples.count
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

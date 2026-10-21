import AudioCommon

extension KyutaiSTTModel: StreamingRecognitionModel {
    /// Language hints are ignored, as with the batch recognition protocol.
    public func makeStreamingSession(
        language _: String? = nil
    ) throws -> any StreamingRecognitionSession {
        KyutaiRecognitionSession(session: try makeSession())
    }
}

private final class KyutaiRecognitionSession: StreamingRecognitionSession {
    let inputSampleRate = KyutaiSTTConfig.sampleRate
    private let session: KyutaiSTTSession
    private var lastText = ""
    private var finished = false

    init(session: KyutaiSTTSession) {
        self.session = session
    }

    deinit { session.cancel() }

    func push(_ chunk: CapturedAudioChunk) throws -> [StreamingRecognitionUpdate] {
        guard chunk.sampleRate == inputSampleRate else {
            throw StreamingRecognitionError.sampleRateMismatch(
                expected: inputSampleRate, actual: chunk.sampleRate)
        }
        _ = try session.append(chunk.samples)
        let text = session.text
        guard text != lastText else { return [] }
        lastText = text
        return [StreamingRecognitionUpdate(text: text, isFinal: false, segmentIndex: 0)]
    }

    func finish() throws -> [StreamingRecognitionUpdate] {
        guard !finished else { return [] }
        _ = try session.finish()
        finished = true
        guard !session.text.isEmpty else { return [] }
        return [StreamingRecognitionUpdate(text: session.text, isFinal: true, segmentIndex: 0)]
    }
}

import AudioCommon
import Foundation
import KyutaiSTT

/// Uses native 80 ms streaming frames, including end-of-input flushing.
final class KyutaiSTTEngine: BenchEngine, @unchecked Sendable {
    let name: String
    private let modelPath: String
    private var model: KyutaiSTTModel?
    private(set) var loadElapsed: Double = 0

    init(variant: KyutaiSTTVariant) {
        name = "kyutai-stt-mlx-\(variant.rawValue)"
        let local = ProcessInfo.processInfo.environment["KYUTAI_STT_\(variant.rawValue.uppercased())_MODEL_PATH"]
        modelPath = local.flatMap { $0.isEmpty ? nil : $0 } ?? variant.modelId
    }

    func load(warmupAudio: [Float], sampleRate: Int) async throws {
        let start = Date()
        model = try await KyutaiSTTModel.load(modelPath)
        _ = try await transcribe(audio: warmupAudio, sampleRate: sampleRate, language: nil)
        loadElapsed = Date().timeIntervalSince(start)
    }

    func transcribe(
        audio: [Float], sampleRate: Int, language: String?
    ) async throws -> (text: String, timings: Timings) {
        guard let model else { throw BenchError.notLoaded(name) }
        let start = Date()
        let pcm = sampleRate == model.inputSampleRate ? audio : AudioFileLoader.resample(
            audio, from: sampleRate, to: model.inputSampleRate)
        let session = try model.makeSession()
        defer { session.cancel() }
        var firstToken: Double?
        var firstAudio: Double?
        var frameLatencies: [Double] = []
        for cursor in stride(from: 0, to: pcm.count, by: KyutaiSTTConfig.frameSize) {
            let end = min(cursor + KyutaiSTTConfig.frameSize, pcm.count)
            let frameStart = Date()
            let tokens = try session.append(Array(pcm[cursor..<end]))
            frameLatencies.append(Date().timeIntervalSince(frameStart))
            if firstToken == nil, !tokens.isEmpty {
                firstToken = Date().timeIntervalSince(start)
                firstAudio = Double(end) / Double(model.inputSampleRate)
            }
        }
        let flushStart = Date()
        let tail = try session.finish()
        let flushSeconds = Date().timeIntervalSince(flushStart)
        if firstToken == nil, !tail.isEmpty {
            firstToken = Date().timeIntervalSince(start)
            firstAudio = Double(pcm.count) / Double(model.inputSampleRate)
        }
        return (session.text, Timings(
            elapsed: Date().timeIntervalSince(start),
            audioDuration: Double(audio.count) / Double(sampleRate),
            firstTokenSeconds: firstToken, firstTokenAudioSeconds: firstAudio,
            frameLatencyP50Seconds: percentile(frameLatencies, 0.5),
            frameLatencyP95Seconds: percentile(frameLatencies, 0.95), flushSeconds: flushSeconds))
    }
}

private func percentile(_ samples: [Double], _ fraction: Double) -> Double? {
    guard !samples.isEmpty else { return nil }
    let sorted = samples.sorted()
    return sorted[max(0, min(sorted.count - 1, Int(ceil(Double(sorted.count) * fraction)) - 1))]
}

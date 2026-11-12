import AudioCommon
import Foundation
import os
import XCTest
@testable import Qwen3ASR

final class E2EDownloadOptionsLoadingTests: XCTestCase {
    func testCachedLoadWithObserverPreservesTranscription() async throws {
        let modelId = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"
        let directory = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        guard HuggingFaceDownloader.weightsExist(in: directory) else {
            throw XCTSkip("Cache the Qwen3-ASR 0.6B 4-bit bundle before running this smoke test")
        }
        let retries = OSAllocatedUnfairLock(initialState: 0)
        var progress: [Double] = []
        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: modelId, cacheDir: directory, offlineMode: true,
            downloadOptions: .init(retryDelaysSeconds: [0, 0], onRetry: { _ in
                retries.withLock { $0 += 1 }
            }),
            progressHandler: { fraction, _ in progress.append(fraction) })
        XCTAssertEqual(retries.withLock { $0 }, 0)
        XCTAssertEqual(progress.last, 1.0)

        let fixture = try XCTUnwrap(Bundle.module.url(forResource: "test_audio", withExtension: "wav"))
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: fixture)
        let targetRate = 24000
        let audio = sampleRate == targetRate ? samples : AudioFileLoader.resample(
            samples, from: sampleRate, to: targetRate)
        let text = model.transcribe(audio: audio, sampleRate: targetRate)
        XCTAssertEqual(text, "Can you guarantee that the replacement part will be shipped tomorrow?")
    }
}

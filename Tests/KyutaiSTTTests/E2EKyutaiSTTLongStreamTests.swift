import AudioCommon
import Foundation
import MLX
import XCTest
@testable import KyutaiSTT

final class E2EKyutaiSTTLongStreamTests: XCTestCase {
    func testLongStreamRollsContextAndRetainsFinalWord() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["KYUTAI_STT_MODEL_PATH"] else {
            throw XCTSkip("Set KYUTAI_STT_MODEL_PATH for a real-weight long stream")
        }
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Qwen3ASRTests/Resources/test_audio.wav")
        let audio = try AudioFileLoader.load(url: fixture, targetSampleRate: 24000)
        let model = try await KyutaiSTTModel.load(modelPath, offlineMode: true)
        let session = try model.makeSession()
        // More than 60 s crosses the 750-frame decoder context as well as Mimi's window.
        let repeats = Int(ceil(65 * 24000 / Double(audio.count)))
        var tokens: [KyutaiSTTToken] = []
        for _ in 0..<repeats {
            for start in stride(from: 0, to: audio.count, by: 2881) {
                tokens += try session.append(Array(audio[start..<min(start + 2881, audio.count)]))
            }
        }
        let processedFrames = session.receivedSamples / KyutaiSTTConfig.frameSize
        XCTAssertGreaterThan(processedFrames, model.config.context)
        for cache in session.decoderCache {
            XCTAssertEqual(cache.keysArray?.shape[2], model.config.context)
            XCTAssertEqual(cache.offset, processedFrames)
        }
        for cache in session.encoderCache {
            XCTAssertEqual(cache.keysArray?.shape[2], model.mimi.cfg.transformer.context)
            XCTAssertEqual(cache.offset, processedFrames * 2)
        }
        tokens += try session.finish()
        print("Long stream (\(Double(session.receivedSamples) / 24000)s): \(session.text)")
        XCTAssertEqual(session.text.lowercased().components(separatedBy: "guarantee").count - 1, repeats)
        XCTAssertTrue(session.text.lowercased().hasSuffix("tomorrow?"))
        XCTAssertTrue(zip(tokens, tokens.dropFirst()).allSatisfy { $0.timestamp <= $1.timestamp })
        XCTAssertTrue(session.decoderCache.allSatisfy { $0.keysArray == nil })
        let clean = try model.transcribe(audio: audio, sampleRate: 24000)
        XCTAssertEqual(clean, "Can you guarantee that the replacement part will be shipped tomorrow?")
    }
}

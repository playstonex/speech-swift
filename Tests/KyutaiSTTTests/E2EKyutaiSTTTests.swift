import AudioCommon
import Foundation
import MLX
import XCTest
@testable import KyutaiSTT

final class E2EKyutaiSTTTests: XCTestCase {
    func testSharedStreamingRecognitionProtocol() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["KYUTAI_STT_MODEL_PATH"], !modelPath.isEmpty else {
            throw XCTSkip("Set KYUTAI_STT_MODEL_PATH to an exported Kyutai STT bundle")
        }
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Qwen3ASRTests/Resources/test_audio.wav")
        let audioURL = ProcessInfo.processInfo.environment["KYUTAI_STT_AUDIO_PATH"].map {
            URL(fileURLWithPath: $0)
        } ?? fixture
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 24_000)
        let model = try await KyutaiSTTModel.load(modelPath, offlineMode: true)
        let recognizer: any StreamingRecognitionModel = model
        let session = try recognizer.makeStreamingSession(language: nil)
        XCTAssertEqual(session.inputSampleRate, 24_000)
        XCTAssertThrowsError(try session.push(samples: [0], sampleRate: 16_000)) { error in
            XCTAssertEqual(error as? StreamingRecognitionError,
                           .sampleRateMismatch(expected: 24_000, actual: 16_000))
        }

        var partials: [StreamingRecognitionUpdate] = []
        for start in stride(from: 0, to: audio.count, by: 997) {
            let end = min(start + 997, audio.count)
            partials += try session.push(
                samples: Array(audio[start..<end]), sampleRate: 24_000,
                frameIndex: Int64(start), isFinal: end == audio.count)
        }
        XCTAssertFalse(partials.isEmpty)
        XCTAssertTrue(partials.allSatisfy { !$0.isFinal && $0.segmentIndex == 0 })
        let final = try session.finish()
        XCTAssertEqual(final.count, 1)
        let update = try XCTUnwrap(final.first)
        XCTAssertEqual(update.text, "Can you guarantee that the replacement part will be shipped tomorrow?")
        XCTAssertTrue(update.isFinal)
        XCTAssertFalse(update.endOfUtterance)
        XCTAssertEqual(update.segmentIndex, 0)
        XCTAssertEqual(try session.finish(), [])
        XCTAssertThrowsError(try session.push(samples: [0], sampleRate: 24_000))

        let empty = try recognizer.makeStreamingSession(language: nil)
        XCTAssertEqual(try empty.finish(), [])
        var abandoned: (any StreamingRecognitionSession)? = try recognizer.makeStreamingSession(language: nil)
        XCTAssertNotNil(abandoned)
        XCTAssertThrowsError(try model.makeSession())
        abandoned = nil
        let next = try model.makeSession()
        next.cancel()
    }

    func testMimiReferenceStages() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["KYUTAI_STT_MODEL_PATH"],
              let referencePath = ProcessInfo.processInfo.environment["KYUTAI_STT_CODEC_REFERENCE_PATH"],
              let audioPath = ProcessInfo.processInfo.environment["KYUTAI_STT_AUDIO_PATH"] else {
            throw XCTSkip("Set the Kyutai codec reference, model, and audio paths")
        }
        let model = try await KyutaiSTTModel.load(modelPath, offlineMode: true)
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: referencePath))
        let audio = try AudioFileLoader.load(url: URL(fileURLWithPath: audioPath), targetSampleRate: 24000)
        let session = try model.makeSession()
        defer { session.cancel() }
        struct CodeReference: Decodable { let mimi_codebooks: [[Int32]] }
        let codeReference = try ProcessInfo.processInfo.environment["KYUTAI_STT_CODEC_CODES_REFERENCE_PATH"].map {
            try JSONDecoder().decode(CodeReference.self, from: Data(contentsOf: URL(fileURLWithPath: $0)))
        }
        var codeDifferences = Array(repeating: 0, count: model.config.numCodebooks)
        let cache = (0..<model.mimi.cfg.transformer.numLayers).map { _ in
            KyutaiSTTCache(capacity: model.mimi.cfg.transformer.context)
        }
        let referenceFrames = Set(reference.keys.compactMap { Int($0.split(separator: ".").last ?? "") })
        let featureLastFrame = try XCTUnwrap(referenceFrames.max())
        let audioFrames = (audio.count + 1919) / 1920
        let lastFrame = max(featureLastFrame, codeReference.map { min(audioFrames, $0.mimi_codebooks.count) - 1 } ?? 0)
        guard lastFrame < (audio.count + 1919) / 1920 else {
            XCTFail("Reference frame lies beyond the audio's padded final block")
            return
        }
        for frame in 0...lastFrame {
            var pcm = Array(audio[(frame * 1920)..<min((frame + 1) * 1920, audio.count)])
            pcm += repeatElement(0, count: 1920 - pcm.count)
            let encoder = model.mimi.encoder.step(MLXArray(pcm).reshaped(1, 1, 1920))
            let transformer = model.mimi.encoder_transformer(encoder, cache: cache)[0]
            let downsample = model.mimi.downsample.step(transformer)
            eval(downsample)
            if let codeReference {
                let codes = model.mimi.quantizer.encode(downsample).asArray(Int32.self)
                for i in codes.indices where codes[i] != codeReference.mimi_codebooks[frame][i] {
                    codeDifferences[i] += 1
                }
            }
            guard referenceFrames.contains(frame) else { continue }
            for (stage, actual) in [("encoder", encoder), ("transformer", transformer), ("downsample", downsample)] {
                let expected = try XCTUnwrap(reference["\(stage).\(frame)"])
                let difference = abs(actual - expected).max().item(Float.self)
                print("\(stage).\(frame): maximum difference \(difference)")
                XCTAssertLessThan(difference, 0.001, "PyTorch codec parity at \(stage), frame \(frame)")
            }
        }
        if codeReference != nil {
            print("Codebook decision differences over \(lastFrame + 1) audio frames: \(codeDifferences)")
        }
    }

    func testKnownSpeechStreamingAndSessionReset() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["KYUTAI_STT_MODEL_PATH"], !modelPath.isEmpty else {
            throw XCTSkip("Set KYUTAI_STT_MODEL_PATH to an upstream or exported Kyutai STT bundle")
        }
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Qwen3ASRTests/Resources/test_audio.wav")
        let audioURL = ProcessInfo.processInfo.environment["KYUTAI_STT_AUDIO_PATH"].map {
            URL(fileURLWithPath: $0)
        } ?? fixture
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: KyutaiSTTConfig.sampleRate)
        let model = try await KyutaiSTTModel.load(modelPath, offlineMode: true)
        let batch = try model.transcribe(audio: audio, sampleRate: KyutaiSTTConfig.sampleRate)
        print("Kyutai transcript: \(batch)")
        let lower = batch.lowercased()
        for word in ["guarantee", "replacement", "shipped", "tomorrow"] {
            XCTAssertTrue(lower.contains(word), "Missing \(word): \(batch)")
        }

        let session = try model.makeSession()
        XCTAssertThrowsError(try model.makeSession())
        XCTAssertThrowsError(try session.append([Float.nan]))
        var tokens: [KyutaiSTTToken] = []
        // Deliberately break Mimi frame boundaries.
        for start in stride(from: 0, to: audio.count, by: 997) {
            tokens += try session.append(Array(audio[start..<min(start + 997, audio.count)]))
        }
        tokens += try session.finish()
        XCTAssertEqual(session.text, batch, "Chunk boundaries or codec state changed the transcript")
        XCTAssertEqual(session.receivedSamples, audio.count)
        if let referencePath = ProcessInfo.processInfo.environment["KYUTAI_STT_REFERENCE_PATH"] {
            struct Reference: Decodable {
                let transcript: String
                let emitted_token_ids: [Int]
            }
            let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: URL(fileURLWithPath: referencePath)))
            XCTAssertEqual(session.text, reference.transcript)
            XCTAssertEqual(tokens.map(\.id), reference.emitted_token_ids, "Native text tokens differ from Kyutai's reference runtime")
        }
        XCTAssertTrue(zip(tokens, tokens.dropFirst()).allSatisfy { $0.timestamp <= $1.timestamp })
        XCTAssertTrue(tokens.allSatisfy { $0.timestamp.isFinite && $0.timestamp >= 0 })
        XCTAssertEqual(try session.finish(), [])
        XCTAssertThrowsError(try session.append([0]))

        let empty = try model.makeSession()
        XCTAssertEqual(try empty.finish(), [])
        XCTAssertEqual(empty.text, "")
        let cancelled = try model.makeSession()
        _ = try cancelled.append(Array(audio.prefix(2000)))
        cancelled.cancel()
        let clean = try model.makeSession()
        clean.cancel()
        XCTAssertThrowsError(try model.transcribe(audio: [0], sampleRate: 0))
        XCTAssertEqual(try model.transcribe(audio: [], sampleRate: 24000), "")
    }
}

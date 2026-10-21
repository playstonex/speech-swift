import AudioCommon
import Foundation
import KokoroTTS
import XCTest
@testable import KyutaiSTT

final class E2EKyutaiSTTRoundTripTests: XCTestCase {
    func testKokoroEnglishRoundTrip() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["KYUTAI_STT_MODEL_PATH"] else {
            throw XCTSkip("Set KYUTAI_STT_MODEL_PATH for real-weight round trips")
        }
        let sentences = [
            ("The quick brown fox jumps over the lazy dog.", "af_heart", "en", ["quick", "brown", "fox", "lazy", "dog"]),
            ("Please open the window and turn off the lights.", "af_heart", "en", ["open", "window", "turn", "lights"]),
        ]
        // Release the TTS pipeline before loading the recognition weights.
        let clips: [[Float]] = try await {
            let tts = try await KokoroTTSModel.fromPretrained(offlineMode: true)
            return try sentences.map { try tts.synthesize(text: $0.0, voice: $0.1, language: $0.2) }
        }()
        let asr = try await KyutaiSTTModel.load(modelPath, offlineMode: true)
        for (sentence, audio) in zip(sentences, clips) {
            XCTAssertGreaterThan(audio.count, 2400)
            let text = try asr.transcribe(audio: audio, sampleRate: 24000)
            print("Kokoro → Kyutai [\(sentence.2)]: \(sentence.0) → \(text)")
            XCTAssertEqual(normalize(text), normalize(sentence.0), "Round-trip transcript changed")
            for word in sentence.3 {
                XCTAssertTrue(text.lowercased().contains(word), "Missing \(word): \(text)")
            }
            let streamed = try asr.makeSession()
            for start in stride(from: 0, to: audio.count, by: 1301) {
                _ = try streamed.append(Array(audio[start..<min(start + 1301, audio.count)]))
            }
            _ = try streamed.finish()
            XCTAssertEqual(streamed.text, text)
        }
    }

    /// A separately synthesized, known-transcript French fixture, with clear pronunciation.
    /// Set KYUTAI_STT_FRENCH_ROUNDTRIP_AUDIO_PATH to the WAV created by the documented `say` command.
    func testFrenchRoundTrip() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["KYUTAI_STT_MODEL_PATH"],
              let audioPath = ProcessInfo.processInfo.environment["KYUTAI_STT_FRENCH_ROUNDTRIP_AUDIO_PATH"] else {
            throw XCTSkip("Set the model and French synthetic round-trip audio paths")
        }
        let audio = try AudioFileLoader.load(url: URL(fileURLWithPath: audioPath), targetSampleRate: 24000)
        let model = try await KyutaiSTTModel.load(modelPath, offlineMode: true)
        let text = try model.transcribe(audio: audio, sampleRate: 24000)
        print("French synthetic round trip: \(text)")
        XCTAssertEqual(normalize(text), normalize("Bonjour, je voudrais un café avec du lait."))
        for word in ["bonjour", "voudrais", "café", "lait"] {
            XCTAssertTrue(text.lowercased().contains(word), "Missing \(word): \(text)")
        }
        let session = try model.makeSession()
        for start in stride(from: 0, to: audio.count, by: 1301) {
            _ = try session.append(Array(audio[start..<min(start + 1301, audio.count)]))
        }
        _ = try session.finish()
        XCTAssertEqual(session.text, text)
    }

    private func normalize(_ text: String) -> String {
        text.lowercased().precomposedStringWithCanonicalMapping.unicodeScalars
            .map { CharacterSet.alphanumerics.contains($0) ? String($0) : " " }
            .joined().split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}

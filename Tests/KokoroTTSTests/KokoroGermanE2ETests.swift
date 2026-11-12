import XCTest
@testable import KokoroTTS
@testable import Qwen3ASR

/// E2E tests for the German Kokoro bundles — downloads
/// `aufklarer/Kokoro-82M-German-Martin-CoreML` (and Victoria) from HuggingFace.
/// Run with: swift test --filter E2EKokoroGermanTests
final class E2EKokoroGermanTests: XCTestCase {

    private static var _martin: KokoroTTSModel?
    private static var _asr: Qwen3ASRModel?

    private func martin() async throws -> KokoroTTSModel {
        if let m = Self._martin { return m }
        let m = try await KokoroTTSModel.fromPretrained(modelId: KokoroTTSModel.germanMartinModelId)
        Self._martin = m
        return m
    }

    private func asrModel() async throws -> Qwen3ASRModel {
        if let m = Self._asr { return m }
        let m = try await Qwen3ASRModel.fromPretrained { progress, status in
            print("[ASR \(Int(progress * 100))%] \(status)")
        }
        Self._asr = m
        return m
    }

    func testGermanBundleHasItsSingleVoice() async throws {
        let m = try await martin()
        XCTAssertEqual(m.availableVoices, ["dm_martin"])
        XCTAssertGreaterThan(m.phonemizer.germanDictionarySize, 40_000,
            "dict_de.json must be downloaded with the bundle")
        // SpeechGenerationModel conformance falls back to the bundle's voice.
        let viaProtocol = try await m.generate(text: "Hallo Welt.", language: "de")
        XCTAssertGreaterThan(viaProtocol.count, 1000)
    }

    func testGermanRoundTrip() async throws {
        let m = try await martin()
        let asr = try await asrModel()

        let inputText = "Guten Tag, wie geht es Ihnen? Die Bücher liegen auf dem großen Tisch."
        let audio = try m.synthesize(text: inputText, voice: "dm_martin", language: "de")
        XCTAssertGreaterThan(audio.count, 24000, "Should produce more than a second of audio")
        let duration = Double(audio.count) / 24000.0
        // Two sentences may render as two chunks; only rule out runaway output.
        XCTAssertLessThan(duration, 12.0)
        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        XCTAssertGreaterThan(rms, 0.001, "Audio should not be silence")

        let transcription = asr.transcribe(audio: audio, sampleRate: 24000).lowercased()
        print("Input:  \"\(inputText)\"")
        print("Output: \"\(transcription)\"")
        let keywords = ["guten", "tag", "bücher", "tisch"]
        let matched = keywords.filter { transcription.contains($0) }
        print("Matched \(matched.count)/\(keywords.count) keywords: \(matched)")
        XCTAssertGreaterThanOrEqual(matched.count, 2,
            "At least 2 of \(keywords) should appear in: \"\(transcription)\"")
    }

    func testNormalizedNumbersAreSpoken() async throws {
        // Numbers, a date, a time and a currency amount go through the German
        // normalizer before phonemization; the result must be real speech,
        // not silence or a clipped buffer.
        let m = try await martin()
        let audio = try m.synthesize(text: "Am 3. Oktober um 18:30 Uhr kostet es 12,80 €.",
                                     voice: "dm_martin", language: "de")
        XCTAssertGreaterThan(audio.count, 48000)
        XCTAssertLessThan(audio.map { abs($0) }.max() ?? 0, 1.0)
    }

    func testVictoriaBundleSynthesizes() async throws {
        let m = try await KokoroTTSModel.fromPretrained(modelId: KokoroTTSModel.germanVictoriaModelId)
        XCTAssertEqual(m.availableVoices, ["df_victoria"])
        let audio = try m.synthesize(text: "Schön, dass du da bist.", voice: "df_victoria", language: "de")
        XCTAssertGreaterThan(audio.count, 12000)
        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        XCTAssertGreaterThan(rms, 0.001)
    }
}

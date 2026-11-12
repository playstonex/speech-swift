import XCTest
@testable import KokoroTTS
@testable import Qwen3ASR
import AudioCommon
import CoreML

/// E2E tests using fromPretrained() — downloads models from HuggingFace.
/// Run with: swift test --filter E2EKokoroTests
final class E2EKokoroTests: XCTestCase {

    private static var _sharedModel: KokoroTTSModel?
    private static var _sharedASRModel: Qwen3ASRModel?

    private func model() async throws -> KokoroTTSModel {
        if let m = Self._sharedModel { return m }
        let m = try await KokoroTTSModel.fromPretrained()
        Self._sharedModel = m
        return m
    }

    private func asrModel() async throws -> Qwen3ASRModel {
        if let m = Self._sharedASRModel { return m }
        let m = try await Qwen3ASRModel.fromPretrained { progress, status in
            print("[ASR \(Int(progress * 100))%] \(status)")
        }
        Self._sharedASRModel = m
        return m
    }

    func testModelLoadsAndHasE2E() async throws {
        let m = try await model()
        XCTAssertTrue(m.availableVoices.contains("af_heart"))
        // Regression (#212): `fromPretrained` used to download only one voice JSON,
        // so `availableVoices` reported a single entry and non-default voices failed.
        XCTAssertGreaterThan(m.availableVoices.count, 1,
            "fromPretrained must download the full voice catalog, not just the default voice")
        XCTAssertTrue(m.availableVoices.contains("jf_alpha"),
            "Japanese voice preset must be available without re-downloading")
    }

    func testSynthesizeEnglish() async throws {
        let m = try await model()
        let audio = try m.synthesize(text: "The quick brown fox jumps over the lazy dog.", voice: "af_heart")

        XCTAssertGreaterThan(audio.count, 1000, "Should produce meaningful audio")
        let duration = Double(audio.count) / 24000.0
        print("English: \(audio.count) samples (\(String(format: "%.2f", duration))s)")
        XCTAssertGreaterThan(duration, 0.3)
        XCTAssertLessThan(duration, 10.0)

        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        XCTAssertGreaterThan(rms, 0.001, "Audio should not be silence")
    }

    func testLongTextIsNotClippedToOnePass() async throws {
        // One pass of the bounded graph holds 5 s; three sentences need more.
        let m = try await model()
        let asr = try await asrModel()
        let text = "The meeting starts at nine tomorrow morning. Please bring the budget report "
            + "and the travel plans for next month. We will finish with a short review of the new product launch."
        let audio = try m.synthesize(text: text, voice: "af_heart")
        let duration = Double(audio.count) / 24000.0
        print("Long text: \(String(format: "%.2f", duration))s")
        XCTAssertGreaterThan(duration, 6.0, "Three sentences must not be clipped to a single 5 s pass")
        XCTAssertLessThanOrEqual(audio.map { abs($0) }.max() ?? 0, 1.0)

        let transcription = asr.transcribe(audio: audio, sampleRate: 24000).lowercased()
        print("Output: \"\(transcription)\"")
        let keywords = ["meeting", "morning", "budget", "travel", "review", "product", "launch"]
        let matched = keywords.filter { transcription.contains($0) }
        XCTAssertGreaterThanOrEqual(matched.count, 6, "Expected \(keywords) in \"\(transcription)\"")
        XCTAssertTrue(transcription.contains("launch"), "The last sentence must be spoken")
    }

    func testFirstSynthesisAfterLoadIsWarm() async throws {
        // fromPretrained warms the model up, so the first reply costs what
        // later ones do. Without it the first call is 3-6x slower.
        let bundles = [
            (KokoroTTSModel.defaultModelId, "af_heart", "en", "Hello there, how are you today?"),
            (KokoroTTSModel.germanMartinModelId, "dm_martin", "de", "Guten Tag, wie geht es Ihnen?"),
        ]
        for (modelId, voice, language, text) in bundles {
            let m = try await KokoroTTSModel.fromPretrained(modelId: modelId)
            func seconds() throws -> Double {
                let t0 = CFAbsoluteTimeGetCurrent()
                _ = try m.synthesize(text: text, voice: voice, language: language)
                return CFAbsoluteTimeGetCurrent() - t0
            }
            let first = try seconds()
            let warm = try (0..<3).map { _ in try seconds() }.sorted()[1]
            print("\(modelId): first synthesis \(String(format: "%.3f", first)) s, warm \(String(format: "%.3f", warm)) s")
            XCTAssertLessThan(first, warm * 2.5, "\(modelId): the first synthesis after loading was not warm")
        }
    }

    func testTextThatFitsOnePassStreamsAsOneChunk() async throws {
        // Text within one chunk's budget renders in one call.
        let m = try await model()
        var chunks: [AudioChunk] = []
        for try await chunk in m.generateStream(text: "Hello there. How are you today?", language: "en") {
            chunks.append(chunk)
        }
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.isFinal, true)
        XCTAssertEqual(chunks.first?.frameIndex, 0)
        XCTAssertGreaterThan(chunks.first?.samples.count ?? 0, 24000)
    }

    func testStreamYieldsEachChunkAsItIsRendered() async throws {
        // Well past one 5 s pass, so the text is split.
        let m = try await model()
        let text = "The first sentence is spoken right away. The second one follows right after it. "
            + "A third sentence keeps the stream going for a while longer. "
            + "The fourth one adds a few more words to the paragraph. "
            + "And the fifth sentence finally closes the stream."
        var chunks: [AudioChunk] = []
        for try await chunk in m.generateStream(text: text, language: "en") {
            chunks.append(chunk)
        }
        XCTAssertGreaterThanOrEqual(chunks.count, 2, "Three sentences should stream as several chunks")
        XCTAssertEqual(chunks.filter(\.isFinal).count, 1)
        XCTAssertEqual(chunks.last?.isFinal, true)
        var offset = 0
        for chunk in chunks {
            XCTAssertEqual(chunk.frameIndex, offset, "frameIndex is the chunk's first sample in the stream")
            XCTAssertEqual(chunk.sampleRate, 24000)
            XCTAssertFalse(chunk.samples.isEmpty)
            offset += chunk.samples.count
        }
        let joined = try m.synthesize(text: text, voice: m.defaultBundleVoice)
        XCTAssertEqual(Double(offset), Double(joined.count), accuracy: Double(joined.count) * 0.05)
    }

    func testSynthesizeShortText() async throws {
        let m = try await model()
        let audio = try m.synthesize(text: "Hi", voice: "af_heart")
        XCTAssertGreaterThan(audio.count, 0, "Even short text should produce audio")
    }

    func testPhonemizerTokenizes() async throws {
        let m = try await model()
        // Verify phonemizer works via synthesis (if tokenization fails, synthesis throws)
        let audio = try m.synthesize(text: "Hello world", voice: "af_heart")
        XCTAssertGreaterThan(audio.count, 1000)
    }

    // MARK: - Chinese Roundtrip

    /// Synthesize Chinese text with Kokoro, transcribe with Qwen3-ASR, verify output.
    /// Kokoro-82M is a lightweight model — phonemization is approximate, so we verify
    /// that the output is recognizable Chinese speech (not silence or garbled noise).
    func testChineseRoundTrip() async throws {
        let tts = try await model()
        let asr = try await asrModel()

        let inputText = "你好世界，这是一个测试。"
        let audio = try tts.synthesize(text: inputText, voice: "zf_xiaobei", language: "zh")

        XCTAssertGreaterThan(audio.count, 1000, "Should produce meaningful audio")
        let duration = Double(audio.count) / 24000.0
        print("Chinese TTS: \(audio.count) samples (\(String(format: "%.2f", duration))s)")

        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        XCTAssertGreaterThan(rms, 0.001, "Audio should not be silence")

        let transcription = asr.transcribe(audio: audio, sampleRate: 24000)
        print("Input:  \"\(inputText)\"")
        print("Output: \"\(transcription)\"")

        // Verify ASR produces Chinese characters (not empty or ASCII-only)
        let hasChinese = transcription.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) }
        XCTAssertTrue(hasChinese, "ASR should recognize Chinese speech: \"\(transcription)\"")

        // Check overlap: individual characters from input that appear in transcription
        let inputChars = Set(inputText.filter { $0 != "，" && $0 != "。" })
        let matchedChars = inputChars.filter { transcription.contains($0) }
        print("Matched \(matchedChars.count)/\(inputChars.count) characters: \(String(matchedChars.sorted()))")

        XCTAssertGreaterThanOrEqual(matchedChars.count, 3,
            "At least 3 input characters should appear in: \"\(transcription)\"")
    }

    // MARK: - Japanese Roundtrip

    /// Synthesize Japanese text with Kokoro, transcribe with Qwen3-ASR, verify keywords.
    func testJapaneseRoundTrip() async throws {
        let tts = try await model()
        let asr = try await asrModel()

        let inputText = "こんにちは世界。"
        let audio = try tts.synthesize(text: inputText, voice: "jf_alpha", language: "ja")

        XCTAssertGreaterThan(audio.count, 1000, "Should produce meaningful audio")
        let duration = Double(audio.count) / 24000.0
        print("Japanese TTS: \(audio.count) samples (\(String(format: "%.2f", duration))s)")

        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        XCTAssertGreaterThan(rms, 0.001, "Audio should not be silence")

        let transcription = asr.transcribe(audio: audio, sampleRate: 24000)
        print("Input:  \"\(inputText)\"")
        print("Output: \"\(transcription)\"")

        // Check key Japanese text appears
        let expectedPhrases = ["こんにちは", "世界"]
        let matchedPhrases = expectedPhrases.filter { transcription.contains($0) }
        print("Matched \(matchedPhrases.count)/\(expectedPhrases.count) phrases: \(matchedPhrases)")

        XCTAssertGreaterThanOrEqual(matchedPhrases.count, 1,
            "At least 1 of \(expectedPhrases) should appear in: \"\(transcription)\"")
    }

    // MARK: - Custom Pronunciations

    /// The neural `bartG2P` fallback guesses from *English* spelling rules, so
    /// names outside the shipped dictionaries come out wrong. Proves an injected
    /// entry is resolved instead of that guess — the unit tests in
    /// `KokoroPronunciationTests` can only pin the resolution order, because
    /// `bartG2P` needs the downloaded CoreML G2P models.
    func testCustomPronunciationBeatsNeuralG2P() async throws {
        let m = try await model()
        let target = "tɑtiˈɑnə"   // "tah-tee-AH-nuh"

        let guessed = m.phonemizer.textToPhonemes("Tatiana")
        print("bartG2P guess:            \"\(guessed)\"")
        XCTAssertNotEqual(guessed, "tatiana",
            "Expected a phonemic guess from bartG2P, not the raw-letter last resort")
        XCTAssertNotEqual(guessed, target,
            "The name must not already resolve correctly, or the test proves nothing")

        m.addPronunciations(["tatiana": target])

        let injected = m.phonemizer.textToPhonemes("Tatiana")
        print("after addPronunciations:  \"\(injected)\"")
        XCTAssertEqual(injected, target,
            "Injected entry must be resolved ahead of the neural G2P fallback")
    }
}

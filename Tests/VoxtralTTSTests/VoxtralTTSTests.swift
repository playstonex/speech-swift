import Foundation
import XCTest
@testable import VoxtralTTS

final class VoxtralTTSConfigurationTests: XCTestCase {
    func testPublishedConfiguration() throws {
        let config = try JSONDecoder().decode(VoxtralTTSConfig.self, from: fixtureData())
        XCTAssertEqual(config.sampleRate, 24_000)
        XCTAssertEqual(config.frameRate, 12.5)
        XCTAssertEqual(config.quantization?.bits, 4)
        XCTAssertEqual(config.transformer.layers, 26)
        XCTAssertEqual(config.audio.offsets.prefix(3), [0, 8194, 8217])
        XCTAssertEqual(config.audio.offsets.count, 37)
        XCTAssertEqual(config.audio.offsets.last, 8999)
        XCTAssertEqual(config.audio.embeddingCount, 9088)
        XCTAssertEqual(config.codec.samplesPerFrame, 1920)
    }

    func testDownloadSelectionIncludesBackboneAlongsideExplicitVoiceWeights() {
        XCTAssertTrue(VoxtralTTSModel.bundleFiles.contains("model.safetensors"))
        XCTAssertEqual(VoxtralTTSModel.bundleFiles.filter { $0.hasPrefix("voice_embedding/") }.count, 20)
    }

    func testRejectsUnsupportedConfigurationBeforeAllocatingWeights() throws {
        for (key, value) in [("model_type", "voxtral" as Any), ("n_kv_heads", 0 as Any), ("max_seq_len", 0 as Any)] {
            var json = try XCTUnwrap(JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any])
            json[key] = value
            XCTAssertThrowsError(try JSONDecoder().decode(VoxtralTTSConfig.self, from: JSONSerialization.data(withJSONObject: json)))
        }
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any])
        json["quantization"] = ["bits": 7, "group_size": 64, "mode": "affine"]
        XCTAssertThrowsError(try JSONDecoder().decode(VoxtralTTSConfig.self, from: JSONSerialization.data(withJSONObject: json)))
    }

    func testOptionsRejectNonFiniteAndUnboundedWork() throws {
        try VoxtralTTSOptions().validate()
        for options in [VoxtralTTSOptions(maxFrames: 0), .init(maxFrames: 4097),
                        .init(denoisingSteps: 1), .init(guidance: .nan), .init(guidance: .infinity),
                        .init(streamingChunkFrames: 0), .init(streamingChunkFrames: 126)] {
            XCTAssertThrowsError(try options.validate())
        }
    }

    func testRejectsMismatchedAcousticDimensionsBeforeInference() throws {
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any])
        var multimodal = try XCTUnwrap(json["multimodal"] as? [String: Any])
        var audio = try XCTUnwrap(multimodal["audio_model_args"] as? [String: Any])
        var transformer = try XCTUnwrap(audio["acoustic_transformer_args"] as? [String: Any])
        transformer["dim"] = 3073
        audio["acoustic_transformer_args"] = transformer
        multimodal["audio_model_args"] = audio
        json["multimodal"] = multimodal
        XCTAssertThrowsError(try JSONDecoder().decode(VoxtralTTSConfig.self, from: JSONSerialization.data(withJSONObject: json)))
    }

    func testLanguageVoiceSelectionDoesNotSilentlyFallback() throws {
        XCTAssertEqual(try VoxtralTTSModel.voice(for: nil), "casual_male")
        XCTAssertEqual(try VoxtralTTSModel.voice(for: "fr-FR"), "fr_male")
        XCTAssertEqual(try VoxtralTTSModel.voice(for: "Arabic"), "ar_male")
        XCTAssertThrowsError(try VoxtralTTSModel.voice(for: "japanese"))
        XCTAssertEqual(Set(VoxtralTTSModel.voices).count, 20)
    }

    func testAlibiSlopeValues() {
        let slopes = VoxtralCodecAttention.alibiSlopes(heads: 8)
        XCTAssertEqual(slopes, [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625])
        XCTAssertEqual(VoxtralCodecAttention.alibiSlopes(heads: 6).count, 6)
    }

    private func fixtureData() throws -> Data {
        try Data(contentsOf: XCTUnwrap(Bundle.module.url(forResource: "voxtral_config", withExtension: "json", subdirectory: "Resources")))
    }
}

final class VoxtralTTSTokenizerTests: XCTestCase {
    func testBytePairMergeOrderAndSpecialOffset() throws {
        let tokenizer = try makeTokenizer()
        XCTAssertEqual(try tokenizer.encode("abc ab"), [1258, 1032, 1256])
        XCTAssertEqual(try tokenizer.encode("é"), [1195, 1169])
    }

    func testVoicePromptEndsAtBeginAudioWithoutExtraAudioStep() throws {
        let tokenizer = try makeTokenizer()
        XCTAssertEqual(try tokenizer.speechPrompt(text: "abc", voice: "casual_male"),
                       [1, 25, 24, 24, 36, 1258, 35, 25])
        XCTAssertThrowsError(try tokenizer.speechPrompt(text: "abc", voice: "missing"))
    }

    func testControlTokenSpellingsRemainOrdinaryText() throws {
        let tokenizer = try makeTokenizer()
        XCTAssertTrue(try tokenizer.encode("[AUDIO]").allSatisfy { $0 >= 1000 })
    }

    func testInputValidationAndWhitespacePreserveContent() throws {
        XCTAssertEqual(try VoxtralTTSTokenizer.normalizedText("  don't\nclip\twords.  "), "don't clip words.")
        XCTAssertThrowsError(try VoxtralTTSTokenizer.normalizedText(" \n\t"))
        XCTAssertThrowsError(try VoxtralTTSTokenizer.normalizedText("bad\u{0}text"))
        XCTAssertThrowsError(try VoxtralTTSTokenizer.normalizedText(String(repeating: "a", count: 16_001)))
    }

    private func makeTokenizer() throws -> VoxtralTTSTokenizer {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tekken.json")
        var vocab: [[String: Any]] = (0...255).map { ["rank": $0, "token_bytes": Data([UInt8($0)]).base64EncodedString()] }
        for (index, text) in ["ab", "bc", "abc"].enumerated() {
            vocab.append(["rank": 256 + index, "token_bytes": Data(text.utf8).base64EncodedString()])
        }
        let file: [String: Any] = [
            "config": ["pattern": "[\\p{L}]+|[^\\p{L}]", "default_vocab_size": 1259, "default_num_special_tokens": 1000],
            "vocab": vocab,
            "special_tokens": [("<s>", 1), ("[BEGIN_AUDIO]", 25), ("[AUDIO]", 24),
                               ("[NEXT_AUDIO_TEXT]", 36), ("[REPEAT_AUDIO_TEXT]", 35)].map { ["token_str": $0.0, "rank": $0.1] as [String: Any] },
            "audio": ["voice_num_audio_tokens": ["casual_male": 2]],
        ]
        try JSONSerialization.data(withJSONObject: file).write(to: url)
        return try VoxtralTTSTokenizer(tekkenURL: url)
    }
}

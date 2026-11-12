import Foundation
@testable import FireRedTTS3TTS
import XCTest

final class FireRedTTS3ConfigurationTests: XCTestCase {
    func testPromptPreservesExactScriptAndInstruction() throws {
        let text = "  Wait—don't change 42!\n你好，世界。  "
        let instruction = "A warm voice.\nThen whisper softly."
        XCTAssertEqual(try FireRedTTS3Prompt.build(text: text, instruction: instruction),
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\n\(instruction)\n\n根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频：\n\(text) /no_think<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n<|sot|>")
    }

    func testEmptyOversizedAndReservedInputsAreRejected() {
        for text in ["", " \n", "<|image_pad|>", "Hi <|eosp|>", String(repeating: "x", count: 32769)] {
            XCTAssertThrowsError(try FireRedTTS3Prompt.build(text: text, instruction: "Calm voice."))
            XCTAssertThrowsError(try FireRedTTS3Prompt.build(text: "Hello.", instruction: text))
        }
    }

    func testUTF8FieldBoundaryPreservesUnicodeInput() throws {
        let boundary = String(repeating: "你", count: 10922) + "ab"
        XCTAssertEqual(boundary.utf8.count, 32768)
        XCTAssertTrue(try FireRedTTS3Prompt.build(text: boundary, instruction: "Calm voice.").contains(boundary))
        let oversized = String(repeating: "你", count: 10923)
        XCTAssertThrowsError(try FireRedTTS3Prompt.build(text: oversized, instruction: "Calm voice."))
        XCTAssertThrowsError(try FireRedTTS3Prompt.build(text: "Hello.", instruction: oversized))
    }

    func testSamplingLimitsRejectNonfiniteAndUnboundedRequests() throws {
        try FireRedTTS3Options().validate()
        var value = FireRedTTS3Options()
        value.guidanceScale = .nan
        XCTAssertThrowsError(try value.validate())
        value = .init(); value.maxAudioPatches = value.minAudioPatches
        XCTAssertThrowsError(try value.validate())
        value = .init(); value.textTopP = 0
        XCTAssertThrowsError(try value.validate())
        value = .init(); value.flowSteps = 0
        XCTAssertThrowsError(try value.validate())
    }

    func testBundleValidationRejectsWrongTaskAndPrecision() throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "network-fixture", withExtension: "json", subdirectory: "Resources"))
        let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        var config = fixture["config"] as! [String: Any]
        // Tiny numerical graphs deliberately do not claim to be production bundles.
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        var decoded = try decoder.decode(FireRedTTS3Config.self, from: JSONSerialization.data(withJSONObject: config))
        XCTAssertThrowsError(try decoded.validate())
        config["latent_dimension"] = 64; config["audio_patch_size"] = 480
        config["storage_dtype"] = "bfloat16"
        decoded = try decoder.decode(FireRedTTS3Config.self, from: JSONSerialization.data(withJSONObject: config))
        XCTAssertThrowsError(try decoded.validate())
    }

    func testMissingOfflineBundleReportsARequiredFile() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            _ = try await FireRedTTS3Model.fromBundle(directory)
            XCTFail("Expected missing local bundle file")
        } catch FireRedTTS3Error.missingFile(let name) { XCTAssertEqual(name, "config.json") }
    }
}

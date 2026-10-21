import XCTest
import VoxtralTTS
@testable import AudioServer

final class VoxtralRoutingTests: XCTestCase {
    func testAliasesSelectTheTTSModel() throws {
        for name in ["voxtral", "voxtral-tts", "voxtral-tts-4bit", "voxtral-4b-tts-2603-mlx-4bit"] {
            let variant = try XCTUnwrap(resolveModelToTTSVariant(name))
            XCTAssertEqual(variant.engine, "voxtral-tts")
            XCTAssertEqual(variant.modelId, VoxtralTTSModel.defaultModelId)
            XCTAssertEqual(variant.kind, .tts)
        }
        XCTAssertNil(resolveModelToASRVariant("voxtral-tts"))
        XCTAssertEqual(MODEL_REGISTRY.first(where: { $0.kind == .tts })?.engine, "kokoro")
    }

    func testPresetVoicesPassThroughTheOpenAIAdapter() {
        XCTAssertEqual(resolvedLocalVoice("fr_female", engine: "voxtral-tts"), "fr_female")
        XCTAssertNil(resolvedLocalVoice("alloy", engine: "voxtral-tts"))
    }

    func testInvalidVoicesAndLanguagesAreClientErrors() throws {
        XCTAssertEqual(try validatedVoxtralVoice(nil, language: "french"), "fr_male")
        XCTAssertEqual(try validatedVoxtralVoice("alloy", language: "english"), "casual_male")
        XCTAssertEqual(try validatedVoxtralVoice("fr_female", language: "english"), "fr_female")
        XCTAssertThrowsError(try validatedVoxtralVoice("missing", language: "english"))
        XCTAssertThrowsError(try validatedVoxtralVoice(nil, language: "japanese"))
        XCTAssertEqual(inferenceErrorStatus(VoxtralTTSError.unknownVoice("missing")), .badRequest)
        XCTAssertEqual(inferenceErrorStatus(VoxtralTTSError.invalidCheckpoint("missing")), .internalServerError)
    }

    func testDispatchRejectsInvalidVoiceBeforeTryingToLoadWeights() async throws {
        let variant = try XCTUnwrap(resolveModelToTTSVariant("voxtral-tts"))
        do {
            _ = try await dispatchSynthesize(text: "Hello.", variant: variant, language: "english",
                                             state: ModelState(), voice: "missing")
            XCTFail("Expected invalid voice")
        } catch VoxtralTTSError.unknownVoice(let voice) {
            XCTAssertEqual(voice, "missing")
        }
    }
}

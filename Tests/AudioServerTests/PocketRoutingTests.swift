import PocketTTS
import XCTest
@testable import AudioServer

final class PocketRoutingTests: XCTestCase {
    func testAliasesAndExistingDefault() throws {
        for alias in ["pocket","pocket-tts","pocket-tts-100m-coreml-int8"] {
            let variant = try XCTUnwrap(resolveModelToTTSVariant(alias))
            XCTAssertEqual(variant.engine,"pocket-tts")
            XCTAssertEqual(variant.modelId,PocketTTSModel.defaultModelId)
        }
        XCTAssertNil(resolveModelToASRVariant("pocket-tts"))
        XCTAssertEqual(MODEL_REGISTRY.first { $0.kind == .tts }?.engine,"kokoro")
        XCTAssertEqual(resolvedLocalVoice("missing",engine: "pocket-tts"),"missing")
        XCTAssertNil(resolvedLocalVoice("alloy",engine: "pocket-tts"))
    }
    func testClientErrorsAndBundleErrorsStayDistinct() {
        XCTAssertEqual(inferenceErrorStatus(PocketTTSError.invalidInput("language")),.badRequest)
        XCTAssertEqual(inferenceErrorStatus(PocketTTSError.invalidConfiguration("weights")),.internalServerError)
        XCTAssertEqual(inferenceErrorStatus(PocketTTSError.generationLimit),.internalServerError)
    }
    func testDispatchRejectsUnsupportedVoiceWithoutLoading() async throws {
        let variant = try XCTUnwrap(resolveModelToTTSVariant("pocket"))
        do {
            _ = try await dispatchSynthesize(text: "Hello.",variant: variant,language: "en",state: ModelState(),voice: "missing")
            XCTFail("Expected voice error")
        } catch PocketTTSError.invalidInput { }
    }
}

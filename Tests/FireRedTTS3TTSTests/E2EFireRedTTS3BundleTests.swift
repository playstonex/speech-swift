import Foundation
@testable import FireRedTTS3TTS
import XCTest

final class E2EFireRedTTS3BundleTests: XCTestCase {
    private func loadModel() async throws -> FireRedTTS3Model {
        guard let path = ProcessInfo.processInfo.environment["FIREREDTTS3_E2E_BUNDLE"] else {
            throw XCTSkip("Set FIREREDTTS3_E2E_BUNDLE to the local exported Instruct bundle")
        }
        return try await FireRedTTS3Model.fromBundle(URL(fileURLWithPath: path))
    }

    func testEnglishAndChineseInstructionPlanningAndAudio() async throws {
        let model = try await loadModel()
        struct Row: Decodable { let text: String; let ids: [Int] }
        struct TokenizerFixture: Decodable { let rows: [Row] }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "tokenizer-fixture", withExtension: "json", subdirectory: "Resources"))
        for row in try JSONDecoder().decode(TokenizerFixture.self, from: Data(contentsOf: url)).rows {
            XCTAssertEqual(model.tokenIDs(row.text), row.ids)
        }
        for (text, instruction) in [
            ("The weather is beautiful today.", "A calm, clear female voice, speaking warmly."),
            ("你好，欢迎使用语音合成。", "一位声音清晰、语气亲切的普通话女声。")
        ] {
            let output = try await model.generateWithPlan(text: text, instruction: instruction)
            XCTAssertFalse(output.voiceDescription.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            XCTAssertGreaterThan(output.samples.count, 12_000)
            XCTAssertTrue(output.samples.allSatisfy(\.isFinite))
            XCTAssertEqual(output.samples.count, output.audioPatches * 3840)
            XCTAssertGreaterThan(output.samples.map { abs($0) }.max() ?? 0, 0.001)
            print("FireRedTTS3 patches=\(output.audioPatches) plan=\(output.planningSeconds)s acoustic=\(output.acousticSeconds)s decoder=\(output.decodingSeconds)s attributes=\(output.voiceDescription)")
        }
        XCTAssertTrue(model.isLoaded)
        XCTAssertGreaterThan(model.memoryFootprint, 9_000_000_000)
        model.unload()
        XCTAssertFalse(model.isLoaded)
        XCTAssertEqual(model.memoryFootprint, 0)
        do {
            _ = try await model.generate(text: "Hello.")
            XCTFail("Unloaded model must reject generation")
        } catch FireRedTTS3Error.modelUnloaded {}
    }

    func testCancellationReleasesGenerationAdmission() async throws {
        let model = try await loadModel()
        let task = Task { try await model.generate(text: "Hello.", instruction: "A clear voice.") }
        task.cancel()
        do { _ = try await task.value; XCTFail("Expected cancellation") } catch is CancellationError {}
        var options = FireRedTTS3Options(); options.maxPlanningTokens = 1
        do {
            _ = try await model.generate(text: "Hello.", instruction: "A clear voice.", options: options)
            XCTFail("A one-token voice plan must reach its limit")
        } catch FireRedTTS3Error.planningLimit {}
    }
}

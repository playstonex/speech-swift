import XCTest
import Qwen3ASRRuntime
import AudioCommon

final class Qwen3ASRRuntimeTests: XCTestCase {
    func testDescriptorsDistinguishBackendAndBufferedOutput() {
        XCTAssertEqual(Qwen3ASRDescriptors.small.backend.rawValue, "mlx")
        XCTAssertEqual(Qwen3ASRDescriptors.coreML.backend.rawValue, "coreml")
        XCTAssertEqual(Qwen3ASRDescriptors.small.streaming.rawValue, "buffered")
        XCTAssertEqual(Qwen3ASRDescriptors.small.inputSampleRate, 16000)
        XCTAssertEqual(Qwen3ASRDescriptors.resolveModelID("large"), Qwen3ASRDescriptors.large.modelID)
        XCTAssertEqual(Qwen3ASRDescriptors.resolveModelID("custom/repo"), "custom/repo")
    }

    func testInvalidAudioFailsBeforeAnyDownload() async throws {
        let runtime = Qwen3ASRRuntime(offlineMode: true)
        for (samples, rate) in [([Float.nan], 16000), ([Float(0)], 0)] {
            do {
                _ = try await runtime.transcribe(audio: samples, sampleRate: rate)
                XCTFail("Invalid input must be rejected before loading")
            } catch AudioModelError.invalidConfiguration { }
        }
        let empty = try await runtime.transcribe(audio: [])
        XCTAssertEqual(empty, "")
    }
}

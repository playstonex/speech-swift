import Foundation
import XCTest

/// Start speech-server with a staged inference manifest before enabling this suite.
final class E2ENativeInferenceTests: XCTestCase {
    func testRealNativeSpeechAndFullWhisperRoundTrip() async throws {
        guard let endpoint = ProcessInfo.processInfo.environment["SPEECH_SWIFT_INFERENCE_URL"] else {
            throw XCTSkip("Set SPEECH_SWIFT_INFERENCE_URL to a native inference server")
        }
        func post(_ payload: [String: Any]) async throws -> [String: Any] {
            var request = URLRequest(url: URL(string: endpoint + "/v1/inference")!)
            request.httpMethod = "POST"
            request.timeoutInterval = 300
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
            let (data, response) = try await URLSession.shared.data(for: request)
            XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200, String(data: data, encoding: .utf8) ?? "")
            return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        }
        let audio = try await post(["operation": "speech", "model": "voxcpm2", "requestID": UUID().uuidString,
                                    "text": "Your appointment is confirmed for tomorrow morning.", "seed": 42])
        XCTAssertEqual(audio["sampleRate"] as? Int, 48000)
        let encoded = try XCTUnwrap(audio["audioBase64"] as? String)
        XCTAssertGreaterThan(try XCTUnwrap(Data(base64Encoded: encoded)).count, 48000)
        let result = try await post(["operation": "transcribe", "model": "whisper-large-v3", "requestID": UUID().uuidString,
                                     "audioBase64": encoded, "sampleRate": 48000])
        let text = try XCTUnwrap(result["text"] as? String).lowercased()
        XCTAssertTrue(text.contains("appointment"), text)
        XCTAssertTrue(text.contains("tomorrow"), text)
    }
}

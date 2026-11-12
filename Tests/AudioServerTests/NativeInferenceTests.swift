import XCTest
import Qwen3Chat
@testable import AudioServer

final class NativeInferenceTests: XCTestCase {
    private func request(_ extra: [String: Any]) throws -> NativeInferenceRequest {
        var fields: [String: Any] = ["operation": "chat", "requestID": "test", "model": "gemma4-e4b-it", "messages": [["role": "user", "content": "Hello"]]]
        fields.merge(extra) { _, new in new }
        return try JSONDecoder().decode(NativeInferenceRequest.self, from: JSONSerialization.data(withJSONObject: fields))
    }

    func testRejectsModelSubstitutionsAndInvalidRequests() throws {
        XCTAssertNoThrow(try request([:]).validate())
        XCTAssertNoThrow(try request(["model": "gemma4-12b-it"]).validate())
        for fields: [String: Any] in [["model": "other"], ["requestID": ""], ["messages": []], ["maxTokens": 0], ["temperature": -1]] {
            XCTAssertThrowsError(try request(fields).validate())
        }
    }

    func testOptionalAgentDirectoryPreservesExistingManifest() throws {
        var manifest: [String: Any] = ["runtimeRevision": "test", "gemmaDirectory": "/gemma",
                                     "voxDirectory": "/vox", "whisperDirectory": "/whisper", "vadDirectory": "/vad"]
        let original = try JSONDecoder().decode(NativeInferenceConfiguration.self,
                                                from: JSONSerialization.data(withJSONObject: manifest))
        XCTAssertNil(original.agentDirectory)
        XCTAssertNil(original.failureCaptureDirectory)
        manifest["agentDirectory"] = "/gemma12b"
        manifest["failureCaptureDirectory"] = "/private/failures"
        let expanded = try JSONDecoder().decode(NativeInferenceConfiguration.self,
                                                from: JSONSerialization.data(withJSONObject: manifest))
        XCTAssertEqual(expanded.agentDirectory, "/gemma12b")
        XCTAssertEqual(expanded.failureCaptureDirectory, "/private/failures")
    }

    func testFailureCapturePreservesWireBodyAndRawCompletionWithoutUsingRequestIDAsPath() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let captures = try NativeInferenceFailures(directory: directory)
        let body = Data("{ \"requestID\": \"../../outside\", \"messages\": [] }\n".utf8)
        let traceData = try JSONSerialization.data(withJSONObject: [
            "prompt": "<bos>prompt", "promptTokenIDs": [2, 17], "generatedTokenIDs": [49],
            "rawOutput": "<tool_call|>", "stopTokenID": 106, "finishReason": "stop"])
        let trace = try JSONDecoder().decode(Gemma4GenerationTrace.self, from: traceData)
        var files: [URL] = []
        for _ in 0..<2 {
            files.append(try captures.write(requestBody: body, requestID: "../../outside", model: "gemma4-12b-it",
                                            runtimeRevision: "revision", error: Gemma4ToolError.malformed("orphan tool call terminator"),
                                            generation: trace))
        }
        XCTAssertNotEqual(files[0], files[1])
        for file in files {
            XCTAssertEqual(file.deletingLastPathComponent().standardizedFileURL.path, directory.standardizedFileURL.path)
            XCTAssertNotNil(UUID(uuidString: file.deletingPathExtension().lastPathComponent))
            let record = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
            XCTAssertEqual((record["requestBody"] as? String).map { Data($0.utf8) }, body)
            XCTAssertEqual((record["requestBodyBase64"] as? String).flatMap { Data(base64Encoded: $0) }, body)
            XCTAssertEqual(record["requestID"] as? String, "../../outside")
            let generation = try XCTUnwrap(record["generation"] as? [String: Any])
            XCTAssertEqual(generation["rawOutput"] as? String, "<tool_call|>")
            XCTAssertEqual(generation["generatedTokenIDs"] as? [Int], [49])
            XCTAssertEqual(generation["finishReason"] as? String, "stop")
            let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
            XCTAssertEqual((attributes[.posixPermissions] as? Int).map { $0 & 0o777 }, 0o600)
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: directory.path)
        XCTAssertEqual((attributes[.posixPermissions] as? Int).map { $0 & 0o777 }, 0o700)
    }

    func testFailureCaptureRejectsPublicDirectoryWithoutChangingPermissions() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: directory.path)
        XCTAssertThrowsError(try NativeInferenceFailures(directory: directory))
        let attributes = try FileManager.default.attributesOfItem(atPath: directory.path)
        XCTAssertEqual((attributes[.posixPermissions] as? Int).map { $0 & 0o777 }, 0o755)
    }

    func testAudioDecodingAndVADFrameValidation() throws {
        let data = Data([0, 128, 0, 0, 255, 127]).base64EncodedString()
        let input = try request(["operation": "transcribe", "model": "whisper-large-v3", "audioBase64": data, "sampleRate": 16000])
        try input.validate()
        XCTAssertEqual(input.audio(), [-1, 0, Float(32767) / 32768])
        XCTAssertThrowsError(try request(["operation": "vad", "model": "silero-vad", "audioBase64": data, "sampleRate": 16000, "streamID": "a"]).validate())
        XCTAssertNoThrow(try request(["operation": "vad", "model": "silero-vad", "audioBase64": Data(repeating: 0, count: 1024).base64EncodedString(), "sampleRate": 16000, "streamID": "a"]).validate())
    }

    func testCancellationTargetsOnlyTheSelectedActiveRequest() throws {
        let registry = NativeInferenceCancellations()
        let first = try registry.insert("first")
        let second = try registry.insert("second")
        XCTAssertThrowsError(try registry.insert("first"))
        XCTAssertTrue(registry.cancel("first"))
        XCTAssertTrue(first.isCancelled)
        XCTAssertFalse(second.isCancelled)
        registry.remove("first")
        XCTAssertFalse(registry.cancel("first"))
    }

    func testGateRetainsPermitAcrossSuspension() async {
        let gate = NativeInferenceGate()
        actor Counter {
            var active = 0
            var maximum = 0
            func enter() { active += 1; maximum = max(maximum, active) }
            func leave() { active -= 1 }
        }
        let counter = Counter()
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<8 {
                group.addTask {
                    await gate.acquire()
                    await counter.enter()
                    await Task.yield()
                    await counter.leave()
                    await gate.release()
                }
            }
        }
        let maximum = await counter.maximum
        XCTAssertEqual(maximum, 1)
    }
}

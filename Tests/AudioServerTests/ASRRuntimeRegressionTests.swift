import XCTest
import AudioCommon
import SpeechRuntime
@testable import AudioServer

private actor InferenceProbe {
    var active = 0
    var maximum = 0
    func enter() { active += 1; maximum = max(maximum, active) }
    func leave() { active -= 1 }
    func maxActive() -> Int { maximum }
}

final class ASRRuntimeRegressionTests: XCTestCase {
    private let port = 19431
    private let client = URLSession(configuration: {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 10
        return config
    }())

    override func tearDown() {
        client.invalidateAndCancel()
        super.tearDown()
    }

    private func start(
        gate: InferenceGate = InferenceGate(),
        vad: (any StreamingVADProvider)? = nil,
        transcribe: @escaping @Sendable ([Float], Int, String, String?) async throws -> String
    ) async throws {
        let state = FailingRealtimeModelLoading(
            vadOverride: vad, inferenceGate: gate, qwenTranscription: transcribe)
        let port = self.port
        let fixture = TestAudioServer(port: port) {
            let service = AudioServer(host: "127.0.0.1", port: port,
                                      state: ModelState(inferenceGate: gate), realtimeState: state)
            try await service.run()
        }
        addTeardownBlock { await fixture.stop() }
        try await fixture.start()
    }

    private func url(_ path: String) -> URL { URL(string: "http://127.0.0.1:\(port)\(path)")! }

    private func transcribe(_ path: String = "/v1/audio/transcriptions") async throws -> (Int, [String: Any]) {
        let wav = try encodeWAV(samples: [Float](repeating: 0, count: 1600), sampleRate: 16000)
        var request = URLRequest(url: url(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "qwen3", "audio_base64": wav.base64EncodedString(), "language": "german"
        ])
        let (data, response) = try await client.data(for: request)
        return ((response as! HTTPURLResponse).statusCode,
                try JSONSerialization.jsonObject(with: data) as! [String: Any])
    }

    func testHTTPInferenceFailurePreservesBackendError() async throws {
        try await start { _, _, _, _ in throw RealtimeModelLoadingFailure("decoder failed") }
        let (status, body) = try await transcribe()
        XCTAssertEqual(status, 500)
        XCTAssertNil(body["text"])
        XCTAssertTrue(((body["error"] as? [String: Any])?["message"] as? String)?.contains("decoder failed") == true)
        let (nativeStatus, nativeBody) = try await transcribe("/transcribe")
        XCTAssertEqual(nativeStatus, 500)
        XCTAssertNil(nativeBody["text"])
    }

    func testValidEmptyTranscriptIsNotReclassifiedAsFailure() async throws {
        try await start { _, sampleRate, modelID, language in
            XCTAssertEqual(sampleRate, 16000)
            XCTAssertEqual(modelID, "aufklarer/Qwen3-ASR-0.6B-MLX-4bit")
            XCTAssertEqual(language, "german")
            return ""
        }
        let (status, body) = try await transcribe()
        XCTAssertEqual(status, 200)
        XCTAssertEqual(body["text"] as? String, "")
    }

    func testConcurrentHTTPRequestsDoNotOverlapModelExecution() async throws {
        let probe = InferenceProbe()
        try await start { _, _, _, _ in
            await probe.enter()
            try await Task.sleep(nanoseconds: 10_000_000)
            await probe.leave()
            return "hello"
        }
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<8 {
                group.addTask { [self] in
                    let (status, body) = try await transcribe()
                    XCTAssertEqual(status, 200)
                    XCTAssertEqual(body["text"] as? String, "hello")
                }
            }
            try await group.waitForAll()
        }
        let maximum = await probe.maxActive()
        XCTAssertEqual(maximum, 1)
    }

    func testWebSocketInferenceFailureKeepsSessionUsable() async throws {
        try await start { _, _, _, language in
            XCTAssertNil(language, "Default Qwen ASR must retain automatic language detection")
            throw RealtimeModelLoadingFailure("decoder failed")
        }
        let ws = client.webSocketTask(with: URL(string: "ws://127.0.0.1:\(port)/v1/realtime")!)
        ws.resume()
        defer { ws.cancel(with: .normalClosure, reason: nil) }
        _ = try await receive(ws)
        try await send(ws, ["type": "session.update", "session": ["model": "qwen3"]])
        _ = try await receive(ws)
        try await send(ws, ["type": "input_audio_buffer.append",
                            "audio": Data(repeating: 0, count: 3200).base64EncodedString()])
        try await send(ws, ["type": "input_audio_buffer.commit"])
        let committed = try await receive(ws)
        XCTAssertEqual(committed["type"] as? String, "input_audio_buffer.committed")
        let error = try await receive(ws)
        XCTAssertEqual(error["type"] as? String, "error")
        XCTAssertEqual((error["error"] as? [String: Any])?["event_type"] as? String,
                       "input_audio_buffer.commit")
        try await send(ws, ["type": "input_audio_buffer.clear"])
        let cleared = try await receive(ws)
        XCTAssertEqual(cleared["type"] as? String, "input_audio_buffer.cleared")
    }

    func testHTTPAndWebSocketShareOneInferenceLane() async throws {
        let probe = InferenceProbe()
        try await start { _, _, _, language in
            XCTAssertTrue(language == "german" || language == "french")
            await probe.enter()
            try await Task.sleep(nanoseconds: 20_000_000)
            await probe.leave()
            return "hello"
        }
        let ws = client.webSocketTask(with: URL(string: "ws://127.0.0.1:\(port)/v1/realtime")!)
        ws.resume()
        defer { ws.cancel(with: .normalClosure, reason: nil) }
        _ = try await receive(ws)
        try await send(ws, ["type": "session.update", "session": ["model": "qwen3", "language": "french"]])
        _ = try await receive(ws)
        try await send(ws, ["type": "input_audio_buffer.append",
                            "audio": Data(repeating: 0, count: 3200).base64EncodedString()])
        try await send(ws, ["type": "input_audio_buffer.commit"])
        async let http = transcribe()
        var completed = false
        while !completed {
            let event = try await receive(ws)
            if event["type"] as? String == "conversation.item.input_audio_transcription.completed" {
                XCTAssertEqual(event["transcript"] as? String, "hello")
            }
            completed = event["type"] as? String == "response.done"
        }
        let (status, _) = try await http
        XCTAssertEqual(status, 200)
        let maximum = await probe.maxActive()
        XCTAssertEqual(maximum, 1)
    }

    func testVADCommitAndHTTPShareTheLaneAndPreserveAutomaticLanguage() async throws {
        let gate = InferenceGate()
        let probe = InferenceProbe()
        let vad = ScriptedVAD(sampleRate: 24_000, chunkSize: 6_000,
                              probabilities: [0, 0.8, 0.8, 0.8, 0.1, 0.1])
        try await start(gate: gate, vad: vad) { _, _, _, language in
            XCTAssertTrue(language == nil || language == "german")
            await probe.enter()
            try await Task.sleep(nanoseconds: 20_000_000)
            await probe.leave()
            return language == nil ? "automatic language" : "explicit language"
        }
        let permit = try await gate.acquire()
        defer { permit.release() }
        let ws = client.webSocketTask(with: URL(string: "ws://127.0.0.1:\(port)/v1/realtime")!)
        ws.resume()
        defer { ws.cancel(with: .normalClosure, reason: nil) }
        _ = try await receive(ws)
        try await send(ws, ["type": "session.update", "session": [
            "model": "qwen3", "turn_detection": [
                "type": "server_vad", "prefix_padding_ms": 250, "silence_duration_ms": 250
            ]
        ]])
        _ = try await receive(ws)
        try await send(ws, ["type": "input_audio_buffer.append",
                            "audio": Data(repeating: 0, count: 36_000 * 2).base64EncodedString()])
        for expected in ["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
                         "input_audio_buffer.committed"] {
            let event = try await receive(ws)
            XCTAssertEqual(event["type"] as? String, expected)
        }
        let blockedMaximum = await probe.maxActive()
        XCTAssertEqual(blockedMaximum, 0)
        async let http = transcribe()
        permit.release()
        var completed = false
        while !completed {
            let event = try await receive(ws)
            if event["type"] as? String == "conversation.item.input_audio_transcription.completed" {
                XCTAssertEqual(event["transcript"] as? String, "automatic language")
            }
            completed = event["type"] as? String == "response.done"
        }
        let (status, body) = try await http
        XCTAssertEqual(status, 200)
        XCTAssertEqual(body["text"] as? String, "explicit language")
        let maximum = await probe.maxActive()
        XCTAssertEqual(maximum, 1)
    }

    func testHTTPQueueOverflowIsServiceUnavailable() async throws {
        let gate = InferenceGate(maximumPending: 0)
        try await start(gate: gate) { _, _, _, _ in
            XCTFail("Rejected request must never reach inference")
            return ""
        }
        let permit = try await gate.acquire()
        defer { permit.release() }
        let (status, body) = try await transcribe()
        XCTAssertEqual(status, 503)
        XCTAssertEqual((body["error"] as? [String: Any])?["type"] as? String, "server_error")
    }

    func testCoreMLRegistryEntryHasExplicitBackend() throws {
        let coreML = try XCTUnwrap(resolveModelToASRVariant("qwen3-coreml")?.descriptor)
        XCTAssertEqual(coreML.backend.rawValue, "coreml")
        XCTAssertEqual(resolveModelToASRVariant("qwen3")?.descriptor?.backend.rawValue, "mlx")
        for variant in MODEL_REGISTRY where variant.engine == "qwen3-asr" {
            let descriptor = try XCTUnwrap(variant.descriptor)
            XCTAssertEqual(descriptor.modelID, variant.modelId)
            XCTAssertEqual(descriptor.id, variant.name)
        }
    }

    func testNativeTTSQueueOverflowIsServiceUnavailable() async throws {
        let gate = InferenceGate(maximumPending: 0)
        try await start(gate: gate) { _, _, _, _ in "" }
        let permit = try await gate.acquire()
        defer { permit.release() }
        var request = URLRequest(url: url("/speak"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["text": "hello", "model": "kokoro"])
        let (_, response) = try await client.data(for: request)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 503)
    }

    private func send(_ ws: URLSessionWebSocketTask, _ payload: [String: Any]) async throws {
        let data = try JSONSerialization.data(withJSONObject: payload)
        try await ws.send(.string(String(decoding: data, as: UTF8.self)))
    }

    private func receive(_ ws: URLSessionWebSocketTask) async throws -> [String: Any] {
        let message = try await ws.receive()
        guard case .string(let string) = message else { return [:] }
        return try JSONSerialization.jsonObject(with: Data(string.utf8)) as! [String: Any]
    }
}

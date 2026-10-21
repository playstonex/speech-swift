import Foundation
import SpeechRuntime
import VoxtralTTS
import XCTest
@testable import AudioServer

final class E2EVoxtralTransportTests: XCTestCase {
    func testSDKHTTPAndWebSocketReturnTheSamePresetAudio() async throws {
        let model: VoxtralTTSModel
        if let directory = ProcessInfo.processInfo.environment["VOXTRAL_TTS_BUNDLE_DIR"] {
            model = try VoxtralTTSModel.fromBundle(URL(fileURLWithPath: directory))
        } else {
            model = try await VoxtralTTSModel.fromPretrained()
        }
        let cache = ModelCache()
        let _: VoxtralTTSModel = try await cache.load(key: VoxtralTTSModel.defaultModelId) { model }
        let state = ModelState(cache: cache)
        let port = 19_440
        let fixture = TestAudioServer(port: port) {
            try await AudioServer(host: "127.0.0.1", port: port, state: state, realtimeState: state).run()
        }
        addTeardownBlock { await fixture.stop() }
        try await fixture.start()

        let text = "Bonjour, nous testons la synthèse vocale."
        let expected = floatToPCM16LE(try await model.generate(text: text, voice: "fr_female"))
        var streamSamples: [Float] = []
        try await model.generateStreaming(text: text, voice: "fr_female") { streamSamples += $0.samples }
        let expectedStreaming = floatToPCM16LE(streamSamples)
        XCTAssertEqual(expectedStreaming.count, expected.count)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/audio/speech")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "voxtral-tts", "input": text, "voice": "fr_female", "response_format": "pcm",
        ])
        let (httpAudio, response) = try await URLSession.shared.data(for: request)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertEqual(httpAudio, expected, "HTTP must preserve the selected voice and all samples")

        var streamRequest = request
        streamRequest.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "voxtral-tts", "input": text, "voice": "fr_female", "response_format": "pcm", "stream": true,
        ])
        let httpStart = ProcessInfo.processInfo.systemUptime
        let (bytes, streamResponse) = try await URLSession.shared.bytes(for: streamRequest)
        XCTAssertEqual((streamResponse as? HTTPURLResponse)?.statusCode, 200)
        var httpStreaming = Data()
        var httpFirstAudio: Double?
        for try await byte in bytes {
            if httpFirstAudio == nil { httpFirstAudio = ProcessInfo.processInfo.systemUptime - httpStart }
            httpStreaming.append(byte)
        }
        let httpTotal = ProcessInfo.processInfo.systemUptime - httpStart
        XCTAssertEqual(httpStreaming, expectedStreaming, "Streaming HTTP must preserve SDK chunks and the final sample")

        let ws = URLSession.shared.webSocketTask(with: URL(string: "ws://127.0.0.1:\(port)/v1/realtime")!)
        ws.resume()
        let timeout = Task { try await Task.sleep(for: .seconds(120)); ws.cancel(with: .goingAway, reason: nil) }
        defer { timeout.cancel(); ws.cancel(with: .normalClosure, reason: nil) }
        _ = try await receive(ws)
        try await send(ws, ["type": "session.update", "session": ["model": "voxtral-tts", "voice": "fr_female"]])
        let updated = try await receive(ws)
        XCTAssertEqual(updated["type"] as? String, "session.updated")
        let websocketStart = ProcessInfo.processInfo.systemUptime
        try await send(ws, ["type": "response.create", "response": ["instructions": text]])
        var websocketFirstAudio: Double?
        var websocketAudio = Data()
        var completed = false
        for _ in 0..<2000 {
            let event = try await receive(ws)
            switch event["type"] as? String {
            case "response.audio.delta":
                if websocketFirstAudio == nil { websocketFirstAudio = ProcessInfo.processInfo.systemUptime - websocketStart }
                let encoded = try XCTUnwrap(event["delta"] as? String)
                websocketAudio.append(try XCTUnwrap(Data(base64Encoded: encoded)))
            case "response.done": completed = true
            case "error": XCTFail("Unexpected WebSocket failure: \(event)"); completed = true
            default: break
            }
            if completed { break }
        }
        XCTAssertTrue(completed)
        let websocketTotal = ProcessInfo.processInfo.systemUptime - websocketStart
        XCTAssertEqual(websocketAudio, expectedStreaming, "WebSocket must preserve the selected voice, order and final samples")

        // Disconnect after the first byte of a long response. A following HTTP
        // request must still acquire server admission and complete normally.
        var cancellationRequest = streamRequest
        cancellationRequest.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "voxtral-tts", "input": String(repeating: "This is a cancellation test. ", count: 20),
            "voice": "casual_male", "stream": true, "response_format": "pcm",
        ])
        let (cancelledBytes, _) = try await URLSession.shared.bytes(for: cancellationRequest)
        var cancelledIterator = cancelledBytes.makeAsyncIterator()
        _ = try await cancelledIterator.next()
        cancelledBytes.task.cancel()
        let (recovered, recoveryResponse) = try await URLSession.shared.data(for: request)
        XCTAssertEqual((recoveryResponse as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertEqual(recovered, expected)
        XCTAssertEqual(state.inferenceGate.pendingCount, 0)

        let timing: [String: Any] = [
            "text": text, "voice": "fr_female", "samples": streamSamples.count,
            "http_first_audio_seconds": try XCTUnwrap(httpFirstAudio), "http_total_seconds": httpTotal,
            "websocket_first_audio_seconds": try XCTUnwrap(websocketFirstAudio), "websocket_total_seconds": websocketTotal,
            "http_stream_pcm_identical_to_sdk": true, "websocket_pcm_identical_to_sdk_stream": true,
            "http_disconnect_recovery": true,
        ]
        print("Voxtral transport first audio: HTTP=\(httpFirstAudio ?? -1)s, WebSocket=\(websocketFirstAudio ?? -1)s")
        if let output = ProcessInfo.processInfo.environment["VOXTRAL_VALIDATION_OUTPUT"] {
            try JSONSerialization.data(withJSONObject: timing, options: [.prettyPrinted, .sortedKeys])
                .write(to: URL(fileURLWithPath: output).appendingPathComponent("transport-streaming.json"))
        }
    }

    private func send(_ socket: URLSessionWebSocketTask, _ object: [String: Any]) async throws {
        let data = try JSONSerialization.data(withJSONObject: object)
        try await socket.send(.string(String(decoding: data, as: UTF8.self)))
    }

    private func receive(_ socket: URLSessionWebSocketTask) async throws -> [String: Any] {
        guard case .string(let text) = try await socket.receive() else { throw URLError(.cannotDecodeContentData) }
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }
}

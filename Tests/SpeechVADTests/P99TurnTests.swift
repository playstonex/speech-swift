#if canImport(CoreML)
import Foundation
import XCTest
@testable import SpeechVAD

final class P99TurnTests: XCTestCase {
    func testChunkingAndAvailabilityTimes() throws {
        var calls = [[Float]]()
        let session = P99TurnSession(reset: {}, step: { samples in
            calls.append(samples)
            return [Float](repeating: 0.25, count: 8)
        })
        XCTAssertTrue(try session.pushAudio([Float](repeating: 1, count: 2_559)).isEmpty)
        let frames = try session.pushAudio([2])
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].last, 2)
        XCTAssertEqual(frames.count, 8)
        XCTAssertEqual(frames.first!.frameEndSeconds, 0.02, accuracy: 1e-9)
        XCTAssertTrue(frames.allSatisfy { $0.availableAfterSeconds == 0.16 })
        XCTAssertEqual(session.processedSamples, 2_560)
    }

    func testResetDiscardsPendingAudioAndTime() throws {
        var resets = 0
        let session = P99TurnSession(reset: { resets += 1 }, step: { _ in
            [Float](repeating: 0.5, count: 8)
        })
        _ = try session.pushAudio([Float](repeating: 0, count: 2_600))
        try session.reset()
        XCTAssertEqual(resets, 1)
        XCTAssertEqual(session.processedSamples, 0)
        XCTAssertTrue(try session.pushAudio([Float](repeating: 0, count: 2_520)).isEmpty)
    }

    func testInvalidInputDoesNotEnterBuffer() throws {
        var calls = 0
        let session = P99TurnSession(reset: {}, step: { _ in
            calls += 1
            return [Float](repeating: 0, count: 8)
        })
        XCTAssertThrowsError(try session.pushAudio([.nan]))
        XCTAssertTrue(try session.pushAudio([Float](repeating: 0, count: 2_559)).isEmpty)
        XCTAssertEqual(calls, 0)
    }

    func testFailureDoesNotReplayPreviouslyCompletedSteps() throws {
        var calls = 0
        let session = P99TurnSession(reset: {}, step: { _ in
            calls += 1
            if calls == 2 { throw NSError(domain: "test", code: 1) }
            return [Float](repeating: 0, count: 8)
        })
        XCTAssertThrowsError(try session.pushAudio([Float](repeating: 0, count: 5_120)))
        XCTAssertEqual(session.processedSamples, 2_560)
        let retried = try session.pushAudio([])
        XCTAssertEqual(calls, 3)
        XCTAssertEqual(retried.count, 8)
        XCTAssertEqual(retried[0].frameEndSeconds, 0.18, accuracy: 1e-9)
    }

    func testIncompatibleStateConfigurationIsRejected() throws {
        let config: [String: Any] = [
            "model_type": "p99-turn-1-mini-coreml", "sample_rate": 16_000,
            "step_samples": 2_560, "frame_samples": 320, "frames_per_step": 8,
            "compiled_model": "p99_turn_step.mlmodelc", "state_shapes": P99TurnModel.stateShapes,
        ]
        let bytes = try JSONSerialization.data(withJSONObject: config)
        XCTAssertNoThrow(try P99TurnModel.decodeConfiguration(bytes))
        var bad = config
        bad["state_shapes"] = ["audio": [1, 240]]
        XCTAssertThrowsError(try P99TurnModel.decodeConfiguration(JSONSerialization.data(withJSONObject: bad)))
    }
}
#endif

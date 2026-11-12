import AudioCommon
import Foundation
import MLX
import XCTest
@testable import FishAudioTTS

final class E2EFishAudioDrama3StreamingControlTests: XCTestCase {
    private enum Expected: Error { case delivery, admissionTimeout }
    private let text = "The quick brown fox jumps over the lazy dog."

    private func model() async throws -> FishAudioDrama3Model {
        let env = ProcessInfo.processInfo.environment
        guard env["FISH_DRAMA3_E2E"] == "1", let path = env["FISH_DRAMA3_BUNDLE"] else {
            throw XCTSkip("Set FISH_DRAMA3_E2E=1 and FISH_DRAMA3_BUNDLE")
        }
        return try await FishAudioDrama3Model.fromBundle(URL(fileURLWithPath: path))
    }

    private func assertAdmissionAvailable(_ model: FishAudioDrama3Model) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                let permit = try await model.inference.acquire()
                permit.release()
            }
            group.addTask {
                try await Task.sleep(for: .seconds(2))
                throw Expected.admissionTimeout
            }
            defer { group.cancelAll() }
            _ = try await group.next()
        }
    }

    func testCallbackFailureAndFrameLimitReleaseAdmission() async throws {
        let model = try await model()
        var delivered = 0
        do {
            try await model.generateStreaming(text: text, options: .init(maxFrames: 512, seed: 42)) { chunk in
                XCTAssertFalse(chunk.isFinal)
                delivered += 1
                throw Expected.delivery
            }
            XCTFail("Expected delivery error")
        } catch Expected.delivery { }
        XCTAssertEqual(delivered, 1)
        try await assertAdmissionAvailable(model)
        var frames = 0, finals = 0
        do {
            try await model.generateStreaming(text: text,
                options: .init(maxFrames: 3, minFrames: 3, seed: 42, streamingChunkFrames: 1)) { chunk in
                frames += chunk.samples.count / 2048
                if chunk.isFinal { finals += 1 }
            }
            XCTFail("Expected explicit frame limit")
        } catch FishAudioError.generationLimit(3) { }
        XCTAssertEqual(frames, 3)
        XCTAssertEqual(finals, 0)
        try await assertAdmissionAvailable(model)
        Memory.clearCache()
    }

    func testActiveCancellationReleasesAdmission() async throws {
        let model = try await model()
        let text = self.text
        let (events, signal) = AsyncStream<Void>.makeStream()
        let task = Task {
            defer { signal.finish() }
            try await model.generateStreaming(text: text, options: .init(maxFrames: 512, seed: 42)) { _ in
                signal.yield(())
                signal.finish()
                try await Task.sleep(for: .seconds(30))
            }
        }
        for await _ in events { break }
        let waiting = Task {
            try await model.generateStreaming(text: text, options: .init(maxFrames: 512, seed: 42)) { _ in
                XCTFail("A queued cancelled stream must not deliver audio")
            }
        }
        for _ in 0..<100 where model.inference.pendingCount == 0 {
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTAssertEqual(model.inference.pendingCount, 1)
        waiting.cancel()
        do { try await waiting.value; XCTFail("Expected queued cancellation") }
        catch is CancellationError { }
        XCTAssertEqual(model.inference.pendingCount, 0)
        task.cancel()
        do { try await task.value; XCTFail("Expected cancellation") }
        catch is CancellationError { }
        try await assertAdmissionAvailable(model)
        Memory.clearCache()
    }

    func testAdmissionHeldUntilFinalCallbackReturns() async throws {
        let model = try await model()
        var waiting: Task<Void, Error>?
        var finals = 0
        try await model.generateStreaming(text: text,
            options: .init(maxFrames: 512, seed: 42, streamingChunkFrames: 128)) { chunk in
            guard chunk.isFinal else { return }
            finals += 1
            waiting = Task {
                let permit = try await model.inference.acquire()
                permit.release()
            }
            for _ in 0..<100 where model.inference.pendingCount == 0 {
                try await Task.sleep(for: .milliseconds(2))
            }
            XCTAssertEqual(model.inference.pendingCount, 1, "Final delivery must retain admission")
        }
        XCTAssertEqual(finals, 1)
        try await waiting?.value
        try await assertAdmissionAvailable(model)
        Memory.clearCache()
    }

    func testDroppingPullStreamReleasesAdmission() async throws {
        let model = try await model()
        try await consumeOneChunk(model)
        try await assertAdmissionAvailable(model)
        // A later complete request still succeeds after abandoning partial output.
        let audio = try await model.generate(text: text, options: .init(maxFrames: 512, seed: 42))
        XCTAssertGreaterThan(audio.count, 44100)
        XCTAssertTrue(audio.allSatisfy(\.isFinite))
        Memory.clearCache()
    }

    private func consumeOneChunk(_ model: FishAudioDrama3Model) async throws {
        let speech: any SpeechGenerationModel = model
        let stream = speech.generateStream(text: text, language: nil)
        var iterator = stream.makeAsyncIterator()
        let next = try await iterator.next()
        let chunk = try XCTUnwrap(next)
        XCTAssertEqual(chunk.samples.count, 3 * 2048)
        XCTAssertEqual(chunk.frameIndex, 0)
        XCTAssertFalse(chunk.isFinal)
    }
}

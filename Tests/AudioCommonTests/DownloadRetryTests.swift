import Foundation
import os
import XCTest
@testable import AudioCommon

final class DownloadRetryTests: XCTestCase {
    private enum Failure: Error, Equatable { case attempt(Int) }

    private final class Recorder: Sendable {
        private struct State: Sendable {
            var attempts = 0
            var events: [DownloadRetryEvent] = []
        }
        private let state = OSAllocatedUnfairLock(initialState: State())
        func attempt() -> Int {
            state.withLock { $0.attempts += 1; return $0.attempts }
        }
        func record(_ event: DownloadRetryEvent) { state.withLock { $0.events.append(event) } }
        var events: [DownloadRetryEvent] { state.withLock { $0.events } }
        var attempts: Int { state.withLock { $0.attempts } }
    }

    func testDefaultOptionsPreserveTheExistingLadder() {
        let options = DownloadOptions()
        XCTAssertNil(options.retryDelaysSeconds)
        XCTAssertNil(options.onRetry)
        XCTAssertEqual(HuggingFaceDownloader.downloadRetryDelaysSeconds, [5, 15, 30, 60])
    }

    func testEveryFailureReportsItsOriginalErrorAndNextDelay() async throws {
        let recorder = Recorder()
        do {
            let _: Void = try await HuggingFaceDownloader.withDownloadRetries(
                modelId: "test/retries",
                downloadOptions: .init(retryDelaysSeconds: [0, 0], onRetry: recorder.record)
            ) {
                throw Failure.attempt(recorder.attempt())
            }
            XCTFail("Expected exhaustion")
        } catch let failure as Failure {
            XCTAssertEqual(failure, .attempt(3))
        }
        XCTAssertEqual(recorder.attempts, 3)
        let events = recorder.events
        XCTAssertEqual(events.map(\.attempt), [1, 2, 3])
        XCTAssertEqual(events.map(\.maxAttempts), [3, 3, 3])
        XCTAssertEqual(events.map(\.retryInSeconds), [0, 0, nil])
        XCTAssertEqual(events.map(\.modelId), Array(repeating: "test/retries", count: 3))
        XCTAssertEqual(events.map { $0.error as? Failure }, [.attempt(1), .attempt(2), .attempt(3)])
    }

    func testSuccessWithoutRetryEmitsNoEvent() async throws {
        let recorder = Recorder()
        let result = try await HuggingFaceDownloader.withDownloadRetries(
            modelId: "test/success",
            downloadOptions: .init(retryDelaysSeconds: [0, 0], onRetry: recorder.record)
        ) {
            recorder.attempt()
        }
        XCTAssertEqual(result, 1)
        XCTAssertTrue(recorder.events.isEmpty)
    }

    func testRecoveryReportsOnlyTheFailedAttemptBeforeSleeping() async throws {
        let recorder = Recorder()
        let clock = ContinuousClock()
        let failureTime = OSAllocatedUnfairLock(initialState: clock.now)
        let result = try await HuggingFaceDownloader.withDownloadRetries(
            modelId: "test/recovery",
            downloadOptions: .init(retryDelaysSeconds: [1], onRetry: { event in
                failureTime.withLock { $0 = clock.now }
                recorder.record(event)
            })
        ) {
            if recorder.attempt() == 1 { throw Failure.attempt(1) }
            let elapsed = failureTime.withLock { $0.duration(to: clock.now) }
            XCTAssertGreaterThanOrEqual(elapsed, .milliseconds(900))
            return "recovered"
        }
        XCTAssertEqual(result, "recovered")
        XCTAssertEqual(recorder.attempts, 2)
        XCTAssertEqual(recorder.events.map(\.attempt), [1])
        XCTAssertEqual(recorder.events.first?.retryInSeconds, 1)
    }

    func testExplicitLegacyDelaysOverrideOptions() async throws {
        let recorder = Recorder()
        do {
            let _: Void = try await HuggingFaceDownloader.withDownloadRetries(
                modelId: "test/override", retryDelaysSeconds: [],
                downloadOptions: .init(retryDelaysSeconds: [0, 0], onRetry: recorder.record)
            ) {
                throw Failure.attempt(recorder.attempt())
            }
            XCTFail("Expected one failed attempt")
        } catch let failure as Failure {
            XCTAssertEqual(failure, .attempt(1))
        }
        XCTAssertEqual(recorder.attempts, 1)
        XCTAssertEqual(recorder.events.map(\.maxAttempts), [1])
        XCTAssertNil(try XCTUnwrap(recorder.events.first).retryInSeconds)
    }

    func testAbsentObserverKeepsRetryBehavior() async throws {
        let recorder = Recorder()
        do {
            let _: Void = try await HuggingFaceDownloader.withDownloadRetries(
                modelId: "test/no-observer", retryDelaysSeconds: [0]
            ) {
                throw Failure.attempt(recorder.attempt())
            }
            XCTFail("Expected exhaustion")
        } catch let failure as Failure {
            XCTAssertEqual(failure, .attempt(2))
        }
        XCTAssertEqual(recorder.attempts, 2)
    }

    func testCancellationBeforeStartingDoesNotAttemptOrNotify() async throws {
        let recorder = Recorder()
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await HuggingFaceDownloader.withDownloadRetries(
                modelId: "test/cancelled",
                downloadOptions: .init(retryDelaysSeconds: [0], onRetry: recorder.record)
            ) { recorder.attempt() }
        }
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {}
        XCTAssertEqual(recorder.attempts, 0)
        XCTAssertTrue(recorder.events.isEmpty)
    }

    func testCancellationDuringAttemptDoesNotNotify() async throws {
        let recorder = Recorder()
        let started = expectation(description: "Attempt started")
        let task = Task {
            try await HuggingFaceDownloader.withDownloadRetries(
                modelId: "test/cancel-attempt",
                downloadOptions: .init(retryDelaysSeconds: [0], onRetry: recorder.record)
            ) {
                _ = recorder.attempt()
                started.fulfill()
                try await Task.sleep(for: .seconds(60))
            }
        }
        await fulfillment(of: [started], timeout: 2)
        task.cancel()
        do {
            try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {}
        XCTAssertEqual(recorder.attempts, 1)
        XCTAssertTrue(recorder.events.isEmpty)
    }

    func testCancellationDuringBackoffDoesNotStartAnotherAttempt() async throws {
        let recorder = Recorder()
        let waiting = expectation(description: "Retry announced")
        let task = Task {
            let _: Void = try await HuggingFaceDownloader.withDownloadRetries(
                modelId: "test/cancel-backoff",
                downloadOptions: .init(retryDelaysSeconds: [60], onRetry: { event in
                    recorder.record(event)
                    waiting.fulfill()
                })
            ) { throw Failure.attempt(recorder.attempt()) }
        }
        await fulfillment(of: [waiting], timeout: 2)
        task.cancel()
        do {
            try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {}
        XCTAssertEqual(recorder.attempts, 1)
        XCTAssertEqual(recorder.events.count, 1)
        XCTAssertEqual(recorder.events.first?.retryInSeconds, 60)
    }

    func testCancellationErrorIsNotRetriedOrWrapped() async throws {
        let recorder = Recorder()
        do {
            try await HuggingFaceDownloader.performDownload(
                modelId: "test/cancel-error", directory: FileManager.default.temporaryDirectory,
                retryDelaysSeconds: [],
                downloadOptions: .init(onRetry: recorder.record), progressHandler: nil
            ) {
                _ = recorder.attempt()
                throw CancellationError()
            }
            XCTFail("Expected CancellationError")
        } catch is CancellationError {}
        XCTAssertEqual(recorder.attempts, 1)
        XCTAssertTrue(recorder.events.isEmpty)
    }

    func testCancellationInFinalObserverStillPropagates() async throws {
        let recorder = Recorder()
        let task = Task {
            let _: Void = try await HuggingFaceDownloader.withDownloadRetries(
                modelId: "test/cancel-observer",
                downloadOptions: .init(retryDelaysSeconds: [], onRetry: { event in
                    recorder.record(event)
                    withUnsafeCurrentTask { $0?.cancel() }
                })
            ) { throw Failure.attempt(recorder.attempt()) }
        }
        do {
            try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {}
        XCTAssertEqual(recorder.events.count, 1)
        XCTAssertEqual(recorder.attempts, 1)
    }

    func testConcurrentLoadsKeepTheirOptionsAndEventsSeparate() async throws {
        let first = Recorder()
        let second = Recorder()
        let results = try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask { try await Self.recover(modelId: "test/first", recorder: first) }
            group.addTask { try await Self.recover(modelId: "test/second", recorder: second) }
            var values: [String] = []
            for try await value in group { values.append(value) }
            return values.sorted()
        }
        XCTAssertEqual(results, ["test/first", "test/second"])
        XCTAssertEqual(first.events.map(\.modelId), ["test/first"])
        XCTAssertEqual(second.events.map(\.modelId), ["test/second"])
        XCTAssertEqual(first.attempts, 2)
        XCTAssertEqual(second.attempts, 2)
    }

    private static func recover(modelId: String, recorder: Recorder) async throws -> String {
        try await HuggingFaceDownloader.withDownloadRetries(
            modelId: modelId,
            downloadOptions: .init(retryDelaysSeconds: [0], onRetry: recorder.record)
        ) {
            let attempt = recorder.attempt()
            await Task.yield()
            if attempt == 1 { throw Failure.attempt(attempt) }
            return modelId
        }
    }
}

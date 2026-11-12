import AudioCommon
import Foundation
import os
import XCTest
@testable import CohereTranscribeASR

/// Loader failures happen before weights are loaded, so these require no model
/// downloads or GPU inference.
final class CohereDownloadRetryTests: XCTestCase {
    private func withUnreachableHub(_ body: (URL) async throws -> Void) async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("cohere-retries-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let keys = ["HF_ENDPOINT", "QWEN3_CACHE_DIR"]
        let previous = keys.map { ProcessInfo.processInfo.environment[$0] }
        setenv(keys[0], "http://127.0.0.1:9", 1)
        setenv(keys[1], directory.path, 1)
        defer {
            for (key, value) in zip(keys, previous) {
                if let value { setenv(key, value, 1) } else { unsetenv(key) }
            }
        }
        try await body(directory)
    }

    func testFromPretrainedForwardsRetryOptions() async throws {
        try await withUnreachableHub { directory in
            let events = OSAllocatedUnfairLock(initialState: [DownloadRetryEvent]())
            let options = DownloadOptions(retryDelaysSeconds: [0, 0], onRetry: { event in
                events.withLock { $0.append(event) }
            })
            var progress: [Double] = []
            do {
                _ = try await CohereTranscribeModel.fromPretrained(
                    "test/cohere-retries", cacheDir: directory,
                    downloadOptions: options, progressHandler: { progress.append($0) })
                XCTFail("Expected a connectivity failure before model loading")
            } catch let error as DownloadError {
                guard case .networkUnavailable = error else {
                    return XCTFail("Unexpected error: \(error)")
                }
            }
            let reported = events.withLock { $0 }
            XCTAssertEqual(reported.map(\.attempt), [1, 2, 3])
            XCTAssertEqual(reported.map(\.maxAttempts), [3, 3, 3])
            XCTAssertEqual(reported.map(\.retryInSeconds), [0, 0, nil])
            XCTAssertEqual(reported.map(\.modelId), Array(repeating: "test/cohere-retries", count: 3))
            XCTAssertTrue(reported.allSatisfy { $0.error is URLError })
            XCTAssertTrue(progress.isEmpty, "Retry notifications must not manufacture byte progress")
        }
    }

    func testLoadConvenienceMethodForwardsRetryOptions() async throws {
        try await withUnreachableHub { _ in
            let events = OSAllocatedUnfairLock(initialState: [DownloadRetryEvent]())
            do {
                _ = try await CohereTranscribeModel.load(
                    "test/cohere-load-retries",
                    downloadOptions: .init(retryDelaysSeconds: [], onRetry: { event in
                        events.withLock { $0.append(event) }
                    }))
                XCTFail("Expected a connectivity failure")
            } catch let error as DownloadError {
                guard case .networkUnavailable = error else {
                    return XCTFail("Unexpected error: \(error)")
                }
            }
            let reported = events.withLock { $0 }
            XCTAssertEqual(reported.map(\.modelId), ["test/cohere-load-retries"])
            XCTAssertEqual(reported.map(\.attempt), [1])
            XCTAssertEqual(reported.map(\.maxAttempts), [1])
            XCTAssertNil(try XCTUnwrap(reported.first).retryInSeconds)
        }
    }

    func testOfflineCacheMissDoesNotReportRetries() async throws {
        try await withUnreachableHub { directory in
            let events = OSAllocatedUnfairLock(initialState: [DownloadRetryEvent]())
            do {
                _ = try await CohereTranscribeModel.fromPretrained(
                    "test/cohere-offline", cacheDir: directory, offlineMode: true,
                    downloadOptions: .init(retryDelaysSeconds: [0, 0], onRetry: { event in
                        events.withLock { $0.append(event) }
                    }))
                XCTFail("Expected an offline cache miss")
            } catch let error as DownloadError {
                guard case .failedToDownload = error else {
                    return XCTFail("Unexpected error: \(error)")
                }
            }
            XCTAssertTrue(events.withLock { $0.isEmpty })
        }
    }
}

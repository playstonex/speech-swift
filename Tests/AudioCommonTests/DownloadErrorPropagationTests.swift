import CryptoKit
import Network
import os
import XCTest
@testable import AudioCommon

final class DownloadErrorPropagationTests: XCTestCase {
    private let modelId = "test/download-errors"

    private func makeScratch(complete: Bool = false) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("download-errors-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        if complete {
            try Data("cached weights".utf8).write(to: directory.appendingPathComponent("model.safetensors"))
            try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        }
        return directory
    }

    private actor Attempts {
        var count = 0
        func record() { count += 1 }
    }

    private func errorAfterRetries(_ failure: Error) async throws -> DownloadError {
        let directory = try makeScratch()
        let attempts = Attempts()
        do {
            try await HuggingFaceDownloader.performDownload(
                modelId: modelId, directory: directory, retryDelaysSeconds: [0, 0],
                progressHandler: nil
            ) {
                await attempts.record()
                throw failure
            }
        } catch {
            let count = await attempts.count
            XCTAssertEqual(count, 3)
            return try XCTUnwrap(error as? DownloadError)
        }
        return try XCTUnwrap(nil as DownloadError?, "Expected the retry ladder to fail")
    }

    func testStalledErrorSurvivesRetries() async throws {
        let error = try await errorAfterRetries(DownloadError.stalled(modelId: modelId, seconds: 17))
        guard case .stalled(let id, let seconds) = error else {
            return XCTFail("Expected stalled, got \(error)")
        }
        XCTAssertEqual(id, modelId)
        XCTAssertEqual(seconds, 17)
    }

    func testChecksumMismatchSurvivesRetries() async throws {
        let error = try await errorAfterRetries(
            DownloadError.checksumMismatch(file: "weights", expected: "expected", actual: "actual"))
        guard case .checksumMismatch(let file, let expected, let actual) = error else {
            return XCTFail("Expected checksumMismatch, got \(error)")
        }
        XCTAssertEqual(file, "weights")
        XCTAssertEqual(expected, "expected")
        XCTAssertEqual(actual, "actual")
    }

    func testInvalidRemoteFileNameSurvivesRetries() async throws {
        let error = try await errorAfterRetries(DownloadError.invalidRemoteFileName("../weights"))
        guard case .invalidRemoteFileName(let file) = error else {
            return XCTFail("Expected invalidRemoteFileName, got \(error)")
        }
        XCTAssertEqual(file, "../weights")
    }

    func testNetworkUnavailableSurvivesRetriesWithoutRewrapping() async throws {
        let error = try await errorAfterRetries(
            DownloadError.networkUnavailable(modelId: "original/model", detail: "original detail"))
        guard case .networkUnavailable(let id, let detail) = error else {
            return XCTFail("Expected networkUnavailable, got \(error)")
        }
        XCTAssertEqual(id, "original/model")
        XCTAssertEqual(detail, "original detail")
    }

    func testForeignNetworkErrorStillBecomesNetworkUnavailable() async throws {
        let failure = URLError(.notConnectedToInternet)
        let error = try await errorAfterRetries(failure)
        guard case .networkUnavailable(let id, let detail) = error else {
            return XCTFail("Expected networkUnavailable, got \(error)")
        }
        XCTAssertEqual(id, modelId)
        XCTAssertEqual(detail, failure.localizedDescription)
    }

    func testGenericDownloadErrorStillIncludesRetryContext() async throws {
        let error = try await errorAfterRetries(DownloadError.failedToDownload("HTTP 404"))
        guard case .failedToDownload(let detail) = error else {
            return XCTFail("Expected failedToDownload, got \(error)")
        }
        XCTAssertTrue(detail.contains("\(modelId) after 3 attempts"))
        XCTAssertTrue(detail.contains("HTTP 404"))
    }

    func testLastAttemptDeterminesTheTypedError() async throws {
        let attempts = Attempts()
        do {
            try await HuggingFaceDownloader.performDownload(
                modelId: modelId, directory: makeScratch(), retryDelaysSeconds: [0],
                progressHandler: nil
            ) {
                await attempts.record()
                if await attempts.count == 1 {
                    throw DownloadError.stalled(modelId: "test/download-errors", seconds: 17)
                }
                throw DownloadError.checksumMismatch(file: "weights", expected: "a", actual: "b")
            }
            XCTFail("Expected checksum failure from the final attempt")
        } catch let error as DownloadError {
            guard case .checksumMismatch(let file, let expected, let actual) = error else {
                return XCTFail("Expected checksumMismatch, got \(error)")
            }
            XCTAssertEqual(file, "weights")
            XCTAssertEqual(expected, "a")
            XCTAssertEqual(actual, "b")
        }
        let count = await attempts.count
        XCTAssertEqual(count, 2)
    }

    func testRetryCanRecoverFromTypedFailure() async throws {
        let attempts = Attempts()
        try await HuggingFaceDownloader.performDownload(
            modelId: modelId, directory: makeScratch(), retryDelaysSeconds: [0],
            progressHandler: nil
        ) {
            await attempts.record()
            if await attempts.count == 1 {
                throw DownloadError.stalled(modelId: "test/download-errors", seconds: 17)
            }
            return []
        }
        let count = await attempts.count
        XCTAssertEqual(count, 2)
    }

    private enum EntryPoint: CaseIterable {
        case weights, files, byteWeighted

        func download(
            modelId: String, to directory: URL, files: [String] = ["model.safetensors"],
            downloadOptions: DownloadOptions = .init(),
            progress: ((Double) -> Void)? = nil
        ) async throws {
            switch self {
            case .weights:
                try await HuggingFaceDownloader.downloadWeights(
                    modelId: modelId, to: directory, additionalFiles: files,
                    retryDelaysSeconds: [0], downloadOptions: downloadOptions,
                    progressHandler: progress)
            case .files:
                try await HuggingFaceDownloader.downloadFiles(
                    modelId: modelId, to: directory, files: files,
                    retryDelaysSeconds: [0], downloadOptions: downloadOptions,
                    progressHandler: progress)
            case .byteWeighted:
                try await HuggingFaceDownloader.downloadFilesByteWeighted(
                    modelId: modelId, to: directory, files: files,
                    retryDelaysSeconds: [0],
                    downloadOptions: downloadOptions,
                    progressHandler: { fraction, _, _, _ in progress?(fraction) })
            }
        }
    }

    private func withHub(
        _ response: LocalHub.Response, _ body: () async throws -> Void
    ) async throws {
        let hub = try LocalHub(response: response)
        let endpoint = try await hub.start()
        defer { hub.stop() }
        let keys = ["HF_ENDPOINT", "HF_DOWNLOAD_STALL_TIMEOUT"]
        let previous = keys.map { ProcessInfo.processInfo.environment[$0] }
        setenv(keys[0], endpoint, 1)
        setenv(keys[1], "1", 1)
        defer {
            for (key, value) in zip(keys, previous) {
                if let value { setenv(key, value, 1) } else { unsetenv(key) }
            }
        }
        try await body()
    }

    func testStallUsesCompleteCacheForEveryEntryPoint() async throws {
        try await withHub(.stalled) {
            for entryPoint in EntryPoint.allCases {
                let directory = try makeScratch(complete: true)
                var progress = 0.0
                let events = OSAllocatedUnfairLock(initialState: [DownloadRetryEvent]())
                try await entryPoint.download(
                    modelId: modelId, to: directory,
                    downloadOptions: .init(onRetry: { event in
                        events.withLock { $0.append(event) }
                    })
                ) { progress = $0 }
                XCTAssertEqual(progress, 1.0, "\(entryPoint)")
                let reported = events.withLock { $0 }
                XCTAssertEqual(reported.map(\.attempt), [1, 2])
                XCTAssertEqual(reported.map(\.maxAttempts), [2, 2])
                XCTAssertEqual(reported.map(\.retryInSeconds), [0, nil])
                for event in reported {
                    guard let failure = event.error as? DownloadError,
                          case .stalled(let id, let seconds) = failure else {
                        XCTFail("Expected the original stall error")
                        continue
                    }
                    XCTAssertEqual(id, modelId)
                    XCTAssertEqual(seconds, 1)
                }
                let cached = try Data(contentsOf: directory.appendingPathComponent("model.safetensors"))
                XCTAssertEqual(cached, Data("cached weights".utf8))
            }
        }
    }

    func testStallRejectsIncompleteCacheForEveryEntryPoint() async throws {
        try await withHub(.stalled) {
            for entryPoint in EntryPoint.allCases {
                let directory = try makeScratch(complete: true)
                // Leave weights in place, but omit a requested tokenizer.
                do {
                    try await entryPoint.download(
                        modelId: modelId, to: directory, files: ["model.safetensors", "tokenizer.json"])
                    XCTFail("\(entryPoint) accepted an incomplete cache")
                } catch let error as DownloadError {
                    guard case .stalled(let id, let seconds) = error else {
                        XCTFail("\(entryPoint): expected stalled, got \(error)")
                        continue
                    }
                    XCTAssertEqual(id, modelId)
                    XCTAssertEqual(seconds, 1)
                }
            }
        }
    }

    func testCorruptedDownloadPropagatesForEveryEntryPoint() async throws {
        try await withHub(.corrupted) {
            for entryPoint in EntryPoint.allCases {
                let directory = try makeScratch(complete: true)
                do {
                    try await entryPoint.download(modelId: modelId, to: directory)
                    XCTFail("\(entryPoint) accepted corrupted bytes")
                } catch let error as DownloadError {
                    guard case .checksumMismatch(let file, let expected, let actual) = error else {
                        XCTFail("\(entryPoint): expected checksumMismatch, got \(error)")
                        continue
                    }
                    XCTAssertEqual(file, "\(modelId)/model.safetensors")
                    XCTAssertEqual(expected, String(repeating: "0", count: 64))
                    let digest = SHA256.hash(data: Data("bad".utf8))
                        .map { String(format: "%02x", $0) }.joined()
                    XCTAssertEqual(actual, digest)
                    XCTAssertFalse(FileManager.default.fileExists(
                        atPath: directory.appendingPathComponent("model.safetensors").path))
                }
            }
        }
    }

    func testUnsafeManifestPathPropagatesForEveryEntryPoint() async throws {
        try await withHub(.unsafePath) {
            for entryPoint in EntryPoint.allCases {
                do {
                    try await entryPoint.download(
                        modelId: modelId, to: makeScratch(), files: ["../model.safetensors"])
                    XCTFail("\(entryPoint) accepted an unsafe path")
                } catch let error as DownloadError {
                    guard case .invalidRemoteFileName(let file) = error else {
                        XCTFail("\(entryPoint): expected invalidRemoteFileName, got \(error)")
                        continue
                    }
                    XCTAssertEqual(file, "../model.safetensors")
                }
            }
        }
    }

    func testHTTPFailureDoesNotUseCacheForEveryEntryPoint() async throws {
        try await withHub(.notFound) {
            for entryPoint in EntryPoint.allCases {
                let events = OSAllocatedUnfairLock(initialState: [DownloadRetryEvent]())
                do {
                    try await entryPoint.download(
                        modelId: modelId, to: makeScratch(complete: true),
                        downloadOptions: .init(onRetry: { event in
                            events.withLock { $0.append(event) }
                        }))
                    XCTFail("\(entryPoint) hid an HTTP rejection with cached files")
                } catch let error as DownloadError {
                    guard case .failedToDownload(let detail) = error else {
                        XCTFail("\(entryPoint): expected failedToDownload, got \(error)")
                        continue
                    }
                    XCTAssertTrue(detail.contains("HTTP 404"))
                }
                let reported = events.withLock { $0 }
                XCTAssertEqual(reported.map(\.attempt), [1, 2])
                XCTAssertEqual(reported.map(\.retryInSeconds), [0, nil])
                for event in reported {
                    guard let failure = event.error as? DownloadError,
                          case .failedToDownload(let detail) = failure else {
                        XCTFail("Expected the original HTTP error")
                        continue
                    }
                    XCTAssertTrue(detail.contains("HTTP 404"))
                    XCTAssertFalse(detail.contains("after 2 attempts"))
                    XCTAssertFalse(detail.contains("target:"))
                }
            }
        }
    }
}

/// A loopback-only Hub fixture: no model weights or external network required.
/// Listener callbacks and connection storage are confined to `queue`.
private final class LocalHub: @unchecked Sendable {
    enum Response { case stalled, corrupted, unsafePath, notFound }

    private let listener: NWListener
    private let queue = DispatchQueue(label: "download-error-tests.hub")
    private let response: Response
    private var connections: [NWConnection] = []

    init(response: Response) throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
        self.response = response
    }

    func start() async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            listener.stateUpdateHandler = { [self] state in
                switch state {
                case .ready:
                    listener.stateUpdateHandler = nil
                    continuation.resume(returning: "http://127.0.0.1:\(listener.port!.rawValue)")
                case .failed(let error):
                    listener.stateUpdateHandler = nil
                    continuation.resume(throwing: error)
                default:
                    break
                }
            }
            listener.newConnectionHandler = { [self] connection in
                connections.append(connection)
                connection.start(queue: queue)
                guard response != .stalled else { return }
                connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) {
                    [self] data, _, _, _ in
                    guard let data else { connection.cancel(); return }
                    let request = String(decoding: data, as: UTF8.self)
                    let body: String
                    switch response {
                    case .corrupted where request.contains("/api/models/"):
                        body = """
                        [{"type":"file","path":"model.safetensors","size":3,
                          "lfs":{"size":3,"oid":"\(String(repeating: "0", count: 64))"}}]
                        """
                    case .unsafePath:
                        body = #"[{"type":"file","path":"../model.safetensors","size":3}]"#
                    default:
                        body = "bad"
                    }
                    let status = response == .notFound ? "404 Not Found" : "200 OK"
                    let header = "HTTP/1.1 \(status)\r\nContent-Length: \(body.utf8.count)\r\nConnection: close\r\n\r\n"
                    connection.send(content: Data((header + body).utf8), completion: .contentProcessed { _ in
                        connection.cancel()
                    })
                }
            }
            listener.start(queue: queue)
        }
    }

    func stop() {
        queue.sync {
            listener.cancel()
            listener.newConnectionHandler = nil
            connections.forEach { $0.cancel() }
            connections.removeAll()
        }
    }
}

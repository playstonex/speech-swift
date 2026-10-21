import AudioCommon
import Foundation
import XCTest
@testable import VoxtralTTS

final class VoxtralTTSDownloadTests: XCTestCase {
    func testFreshCacheIncludesVoicesOutsideTheBackboneIndex() async throws {
        let directory = try makeTemporaryDirectory()
        let previousEndpoint = ProcessInfo.processInfo.environment["HF_ENDPOINT"]
        setenv("HF_ENDPOINT", "https://\(VoxtralBundleURLProtocol.host)", 1)
        URLProtocol.registerClass(VoxtralBundleURLProtocol.self)
        defer {
            URLProtocol.unregisterClass(VoxtralBundleURLProtocol.self)
            if let previousEndpoint { setenv("HF_ENDPOINT", previousEndpoint, 1) }
            else { unsetenv("HF_ENDPOINT") }
        }

        // The deliberately incomplete config stops loading before tensor allocation.
        // Bundle acquisition still goes through the public loader and real downloader.
        do {
            _ = try await VoxtralTTSModel.fromPretrained(
                modelId: VoxtralBundleURLProtocol.modelID, cacheDir: directory)
            XCTFail("The fixture config should stop loading before inference")
        } catch is DecodingError {}

        for file in VoxtralTTSModel.bundleFiles {
            let url = directory.appendingPathComponent(file)
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "Missing bundle file: \(file)")
            if FileManager.default.fileExists(atPath: url.path) {
                XCTAssertEqual(try Data(contentsOf: url), VoxtralBundleURLProtocol.files[file], file)
            }
        }
    }

    func testOfflineCacheRejectsMissingPresetBeforeLoadingWeights() async throws {
        let directory = try makeTemporaryDirectory()
        let missing = "voice_embedding/casual_female.safetensors"
        for file in VoxtralTTSModel.bundleFiles where file != missing {
            let url = directory.appendingPathComponent(file)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try XCTUnwrap(VoxtralBundleURLProtocol.files[file]).write(to: url)
        }

        do {
            _ = try await VoxtralTTSModel.fromPretrained(
                modelId: VoxtralBundleURLProtocol.modelID, cacheDir: directory, offlineMode: true)
            XCTFail("An offline bundle must include every preset")
        } catch let error as DownloadError {
            guard case .failedToDownload(let detail) = error else {
                return XCTFail("Unexpected download error: \(error)")
            }
            XCTAssertTrue(detail.contains(missing), detail)
        }
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxtral-download-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }
}

private final class VoxtralBundleURLProtocol: URLProtocol, @unchecked Sendable {
    static let host = "voxtral-preset-download.test"
    static let modelID = "test/voxtral-tts"
    static let files: [String: Data] = {
        var result = Dictionary(uniqueKeysWithValues: VoxtralTTSModel.bundleFiles.map {
            ($0, Data("fixture: \($0)".utf8))
        })
        result["config.json"] = Data("{}".utf8)
        result["model.safetensors.index.json"] = Data(
            #"{"weight_map":{"backbone.weight":"model.safetensors"}}"#.utf8)
        return result
    }()

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == host
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url else { return }
        let body: Data?
        if url.path == "/api/models/\(Self.modelID)/tree/main" {
            let entries: [[String: Any]] = Self.files.keys.sorted().map { path in
                ["type": "file", "path": path, "size": Self.files[path]!.count]
            }
            body = try? JSONSerialization.data(withJSONObject: entries)
        } else {
            let prefix = "/\(Self.modelID)/resolve/main/"
            body = url.path.hasPrefix(prefix) ? Self.files[String(url.path.dropFirst(prefix.count))] : nil
        }
        let data = body ?? Data()
        let response = HTTPURLResponse(url: url, statusCode: body == nil ? 404 : 200,
            httpVersion: "HTTP/1.1", headerFields: ["Content-Length": "\(data.count)"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

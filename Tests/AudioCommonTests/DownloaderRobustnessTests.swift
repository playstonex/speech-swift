import XCTest
@testable import AudioCommon

/// Regression tests for two large-bundle download failure modes observed
/// with VoxCPM2 bf16 (4.3 GB shard):
/// 1. `weightsExist` accepted a partial multi-shard bundle (stall left
///    shard 1 missing, shard 2 complete) → model loaded half-initialized
///    and synthesized near-silence.
/// 2. The stall guard only ticked on `hub.snapshot` progress callbacks,
///    which fire at file completion — any shard needing longer than the
///    stall window produced zero ticks and a healthy transfer was killed
///    on every retry.
final class DownloaderRobustnessTests: XCTestCase {

    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dl-robust-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    private func write(_ name: String, _ contents: String, in dir: URL) throws {
        try contents.data(using: .utf8)!.write(to: dir.appendingPathComponent(name))
    }

    func testWeightsExistRejectsMissingShard() throws {
        let dir = try makeTempDir()
        try write("model.safetensors.index.json",
                  #"{"weight_map": {"a.w": "model-00001.safetensors", "b.w": "model-00002.safetensors"}}"#,
                  in: dir)
        try write("model-00002.safetensors", "shard2", in: dir)
        XCTAssertFalse(HuggingFaceDownloader.weightsExist(in: dir),
                       "partial multi-shard bundle must not count as cached")
    }

    func testWeightsExistAcceptsCompleteShardedBundle() throws {
        let dir = try makeTempDir()
        try write("model.safetensors.index.json",
                  #"{"weight_map": {"a.w": "model-00001.safetensors", "b.w": "model-00002.safetensors"}}"#,
                  in: dir)
        try write("model-00001.safetensors", "shard1", in: dir)
        try write("model-00002.safetensors", "shard2", in: dir)
        XCTAssertTrue(HuggingFaceDownloader.weightsExist(in: dir))
    }

    func testWeightsExistAcceptsSingleFileWithoutIndex() throws {
        let dir = try makeTempDir()
        try write("model.safetensors", "weights", in: dir)
        XCTAssertTrue(HuggingFaceDownloader.weightsExist(in: dir))
    }

    /// A slow but healthy transfer must survive the stall guard.
    ///
    /// The guard used to watch the destination directory for growth, because
    /// `hub.snapshot` only reported progress once a whole file had landed and a
    /// single multi-GB shard could therefore go quiet for longer than the stall
    /// window. Ranged transfer reports every megabyte of real bytes, so ticks
    /// arrive throughout — and directory size is no longer a progress signal at
    /// all, since the staging file is allocated at full length up front.
    func testStallGuardSurvivesSlowButTickingTransfer() async throws {
        try await HuggingFaceDownloader.withDownloadStallGuard(
            modelId: "test/slow", stallTimeoutSeconds: 2
        ) { tick in
            // 5 s of transfer, each tick well inside the 2 s window.
            for step in 0..<10 {
                try await Task.sleep(for: .milliseconds(500))
                tick(Double(step) / 10.0)
            }
        }
    }

    /// No progress at all is a genuine stall.
    func testStallGuardFiresWhenNothingProgresses() async throws {
        do {
            try await HuggingFaceDownloader.withDownloadStallGuard(
                modelId: "test/stalled", stallTimeoutSeconds: 1
            ) { _ in
                try await Task.sleep(for: .seconds(10))
            }
            XCTFail("expected stall")
        } catch let error as DownloadError {
            guard case .stalled = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }
}

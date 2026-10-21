import XCTest
@testable import AudioCommon

/// Downloads every published model through the real download path.
///
/// Module suites overwhelmingly run against a warm cache, so the download path
/// they exercise is the cache-hit branch. This is the opposite: it fetches
/// bundles for real, across every layout the package ships — single-file MLX,
/// sharded MLX with an index, CoreML `.mlmodelc` directories, mixed bundles —
/// and lets the engine's own size and SHA-256 checks judge the result.
///
/// By default it downloads into the shared cache, so a run also warms models
/// for the suites that skip themselves with "model not cached". In ephemeral
/// mode it fetches to a temporary directory and deletes each bundle once
/// checked, so peak disk is a single model — which is how the full ~90 GB set
/// can be validated on a volume with a few gigabytes free.
///
/// Opt-in, because a full sweep is ~90 GB:
///
///     MODEL_DOWNLOAD_SWEEP=1 swift test --filter E2EModelDownloadSweepTests
///
/// Knobs:
///   `MODEL_DOWNLOAD_SWEEP_MAX_GB`  per-model size ceiling (default 2)
///   `MODEL_DOWNLOAD_SWEEP_FILTER`  substring match on the model id
///   `MODEL_DOWNLOAD_SWEEP_LIMIT`   stop after N models
///   `MODEL_DOWNLOAD_SWEEP_EPHEMERAL` fetch to a temp dir and delete each bundle
///                                    once checked, so peak disk is one model
///   `MODEL_DOWNLOAD_SWEEP_ONLY_UNCACHED` skip models already in the cache
final class E2EModelDownloadSweepTests: XCTestCase {

    /// Free space kept in reserve so a sweep never fills the volume. macOS
    /// degrades badly near zero, and an unchecked model is a much cheaper
    /// outcome than an unusable machine.
    private let reservedHeadroomBytes: Int64 = 3_000_000_000

    private struct Settings {
        let maxBytes: Int64
        let filter: String?
        let limit: Int?
        let ephemeral: Bool
        let onlyUncached: Bool
    }

    private func settings() throws -> Settings {
        let env = ProcessInfo.processInfo.environment
        guard env["MODEL_DOWNLOAD_SWEEP"] == "1" else {
            throw XCTSkip(
                "set MODEL_DOWNLOAD_SWEEP=1 to fetch model bundles for real "
                    + "(bounded by MODEL_DOWNLOAD_SWEEP_MAX_GB, default 2 GB per model)")
        }
        let maxGB = env["MODEL_DOWNLOAD_SWEEP_MAX_GB"].flatMap(Double.init) ?? 2.0
        return Settings(
            maxBytes: Int64(maxGB * 1_000_000_000),
            filter: env["MODEL_DOWNLOAD_SWEEP_FILTER"],
            limit: env["MODEL_DOWNLOAD_SWEEP_LIMIT"].flatMap(Int.init),
            ephemeral: env["MODEL_DOWNLOAD_SWEEP_EPHEMERAL"] == "1",
            onlyUncached: env["MODEL_DOWNLOAD_SWEEP_ONLY_UNCACHED"] == "1")
    }

    /// Free bytes on the volume backing `url`, or `nil` if unknown.
    private func availableBytes(at url: URL) -> Int64? {
        (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage
    }

    /// Fetch each repository in full and confirm every file lands intact.
    ///
    /// Asking for `*` is deliberate: this is a test of the transfer engine, so
    /// it should move everything a repository ships rather than the subset one
    /// module happens to want. `downloadFiles` verifies each file's size, and
    /// its SHA-256 where the Hub publishes one, so a corrupt or truncated
    /// transfer fails here rather than surviving into a cache.
    func testDownloadsEveryPublishedModel() async throws {
        let settings = try settings()
        var ids = try E2EModelInventoryTests.referencedModelIDs()
        if let filter = settings.filter {
            ids = ids.filter { $0.contains(filter) }
        }

        var attempted = 0
        var skippedTooLarge: [String] = []
        var failures: [String] = []
        var bytesFetched: Int64 = 0

        for modelId in ids {
            if let limit = settings.limit, attempted >= limit { break }

            let manifest: RepoManifest
            do {
                manifest = try await HuggingFaceDownloader.fetchManifest(modelId: modelId)
            } catch {
                failures.append("\(modelId): resolve failed — \(error.localizedDescription)")
                continue
            }

            let total = manifest.totalBytes
            guard total <= settings.maxBytes else {
                skippedTooLarge.append(
                    String(format: "%@ (%.2f GB)", modelId, Double(total) / 1e9))
                continue
            }

            let directory: URL
            do {
                directory = settings.ephemeral
                    ? FileManager.default.temporaryDirectory
                        .appendingPathComponent("sweep-\(UUID().uuidString)", isDirectory: true)
                    : try HuggingFaceDownloader.getCacheDirectory(for: modelId)
            } catch {
                failures.append("\(modelId): cache directory — \(error.localizedDescription)")
                continue
            }
            // In ephemeral mode each bundle is deleted as soon as it is
            // checked, so peak usage is one model rather than the whole set —
            // that is what makes validating ~90 GB of bundles possible on a
            // disk with a few gigabytes free.
            defer {
                if settings.ephemeral { try? FileManager.default.removeItem(at: directory) }
            }

            // Consult the real cache even when fetching to a temp directory,
            // otherwise an ephemeral run re-downloads everything already held.
            if settings.onlyUncached,
               let cached = try? HuggingFaceDownloader.getCacheDirectory(for: modelId),
               HuggingFaceDownloader.weightsExist(in: cached) {
                continue
            }

            // Refuse to start a transfer that cannot fit. Filling the disk is a
            // far worse outcome than an unchecked model.
            if let free = availableBytes(at: FileManager.default.temporaryDirectory),
               free < total + reservedHeadroomBytes {
                skippedTooLarge.append(String(
                    format: "%@ (%.2f GB — only %.2f GB free)",
                    modelId, Double(total) / 1e9, Double(free) / 1e9))
                continue
            }

            attempted += 1
            do {
                try await HuggingFaceDownloader.downloadFiles(
                    modelId: modelId, to: directory, files: ["*"])
            } catch {
                failures.append("\(modelId): download failed — \(error.localizedDescription)")
                continue
            }

            // Independently confirm what landed, rather than trusting the call
            // that just claimed success.
            var missing: [String] = []
            for file in manifest.files {
                let local = directory.appendingPathComponent(file.path)
                let size = HuggingFaceDownloader.localFileSize(local)
                if size != file.size {
                    missing.append("\(file.path) (\(size) vs \(file.size))")
                }
            }
            if missing.isEmpty {
                bytesFetched += total
                print(String(format: "[sweep] ok   %@  %.2f GB", modelId, Double(total) / 1e9))
            } else {
                failures.append("\(modelId): incomplete after download — \(missing.prefix(4))")
            }
        }

        print("[sweep] attempted \(attempted), "
            + String(format: "%.2f GB verified", Double(bytesFetched) / 1e9))
        if !skippedTooLarge.isEmpty {
            // Never let a bounded run read as full coverage.
            print("[sweep] over the size ceiling, not fetched: \(skippedTooLarge.count)")
            for entry in skippedTooLarge { print("[sweep]   \(entry)") }
        }

        XCTAssertTrue(failures.isEmpty, "download failures:\n" + failures.joined(separator: "\n"))
        XCTAssertGreaterThan(attempted, 0, "no model was within the size ceiling — raise it")
    }
}

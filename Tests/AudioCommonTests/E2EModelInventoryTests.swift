import XCTest
@testable import AudioCommon

/// Every model the package can ask for must actually exist and ship weights.
///
/// Module E2E suites only cover the variants they happen to instantiate, so a
/// model id that was never published stays invisible until a user selects it.
/// `StableAudio3MusicGen` advertises six variants and its E2E test exercises
/// one; four of the other five are unreachable or empty.
///
/// The inventory is read out of the sources rather than hardcoded, so adding a
/// model automatically brings it under this check instead of requiring someone
/// to remember to register it. Resolution is metadata-only — no weights are
/// downloaded — so the whole sweep costs a few seconds.
final class E2EModelInventoryTests: XCTestCase {

    /// Model ids allowed to be broken, so the sweep can stay green on known
    /// debt while still failing on anything newly broken.
    ///
    /// Empty, and worth keeping that way. It briefly held four Stable Audio
    /// variants — three never published, one uploaded with nothing but a
    /// 38-byte `config.json` — which were removed from `StableAudio3Variant`
    /// rather than tolerated here, since a selectable variant that fails at
    /// download time is worse than one that isn't offered.
    static let knownBroken: Set<String> = []

    /// Owners whose repositories this package downloads at run time. Others
    /// appear in sources as provenance strings (`nvidia/bigvgan…` credits the
    /// vocoder that ships *inside* the IndexTTS2 bundle) and are not fetched.
    static let downloadedOwners = ["aufklarer"]

    // MARK: - Inventory

    /// Repo root, derived from this file's location.
    static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)          // Tests/AudioCommonTests/<this>.swift
            .deletingLastPathComponent()         // Tests/AudioCommonTests
            .deletingLastPathComponent()         // Tests
            .deletingLastPathComponent()         // repo root
    }

    /// Model ids referenced by Swift sources, for owners we actually download.
    static func referencedModelIDs() throws -> [String] {
        let sources = repositoryRoot.appendingPathComponent("Sources")
        guard let walker = FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil)
        else { return [] }

        let pattern = try NSRegularExpression(
            pattern: "\"(" + downloadedOwners.joined(separator: "|") + ")/[A-Za-z0-9._-]+\"")
        var found = Set<String>()

        for case let url as URL in walker where url.pathExtension == "swift" {
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            let range = NSRange(text.startIndex..., in: text)
            for match in pattern.matches(in: text, range: range) {
                guard let r = Range(match.range, in: text) else { continue }
                found.insert(String(text[r]).trimmingCharacters(in: CharacterSet(charactersIn: "\"")))
            }
        }
        return found.sorted()
    }

    func testInventoryIsNotEmpty() throws {
        let ids = try Self.referencedModelIDs()
        XCTAssertGreaterThan(
            ids.count, 50,
            "inventory scan found only \(ids.count) model ids — the source walk is probably broken, "
                + "which would make the sweep below silently vacuous")
    }

    // MARK: - Sweep

    private struct Result: Sendable {
        let modelId: String
        let reachable: Bool
        let weightFiles: Int
        let detail: String
    }

    private func inspect(_ modelId: String) async -> Result {
        do {
            let manifest = try await HuggingFaceDownloader.fetchManifest(modelId: modelId)
            let weights = manifest.files.filter { file in
                let p = file.path
                return p.hasSuffix(".safetensors") || p.hasSuffix(".npz") || p.hasSuffix(".gguf")
                    || p.hasSuffix(".bin") || p.contains(".mlmodelc/") || p.contains(".mlpackage/")
            }
            return Result(
                modelId: modelId, reachable: true, weightFiles: weights.count,
                detail: "\(manifest.files.count) files")
        } catch {
            return Result(
                modelId: modelId, reachable: false, weightFiles: 0,
                detail: error.localizedDescription)
        }
    }

    /// Resolve every referenced model and confirm it ships something loadable.
    ///
    /// A repository that exists but contains no weights is as broken as one
    /// that does not exist — `Stable-Audio-3-DiT-Small-SFX-MLX-4bit` is
    /// published with nothing but a 38-byte config, and resolves 200.
    func testEveryReferencedModelResolvesAndShipsWeights() async throws {
        let ids = try Self.referencedModelIDs()
        var results: [Result] = []

        // Bounded concurrency: enough to keep the sweep quick, not enough to
        // look like abuse to the Hub.
        let batchSize = 8
        for start in stride(from: 0, to: ids.count, by: batchSize) {
            let batch = Array(ids[start..<min(start + batchSize, ids.count)])
            let batchResults = await withTaskGroup(of: Result.self) { group in
                for id in batch {
                    group.addTask { await self.inspect(id) }
                }
                var collected: [Result] = []
                for await result in group { collected.append(result) }
                return collected
            }
            results.append(contentsOf: batchResults)
        }

        let broken = results.filter { !$0.reachable || $0.weightFiles == 0 }
        let brokenIDs = Set(broken.map(\.modelId))

        let unexpected = broken.filter { !Self.knownBroken.contains($0.modelId) }
        XCTAssertTrue(
            unexpected.isEmpty,
            "model ids referenced in Sources that cannot be loaded:\n"
                + unexpected.map { "  \($0.modelId): \($0.detail)" }.joined(separator: "\n"))

        // If something on the known-broken list starts working, say so rather
        // than quietly carrying the entry forever.
        let fixed = Self.knownBroken.subtracting(brokenIDs).intersection(Set(ids))
        if !fixed.isEmpty {
            print("[inventory] now working, remove from knownBroken: \(fixed.sorted())")
        }

        print("[inventory] \(results.count) models checked, "
            + "\(results.count - broken.count) loadable, \(broken.count) broken")
    }
}

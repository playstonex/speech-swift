import AudioCommon
import Foundation
import GLiNER
import MLX
import MLXCommon
import MLXRandom
import OmniVoiceTTS
import SpeechVAD
import XCTest

/// Real model ownership and output checks. Uses cached bundles without downloads.
final class E2EDefaultCacheLimitTests: XCTestCase {
    private let gibibyte = 1024 * 1024 * 1024

    func testTTSOutputMatchesAcrossCacheBudgetsAndLastOwnerRestoresCallerLimit() async throws {
        let directory = try cachedBundle(OmniVoiceTTSModel.defaultModelId, required: "model.safetensors")
        _ = try cachedBundle(PyannoteVADModel.defaultModelId, required: "model.safetensors")
        let original = Memory.cacheLimit
        let callerBudget = 8 * gibibyte
        let ceiling = MLXCacheLimitPolicy.defaultCeiling(
            physicalMemoryBytes: Int(ProcessInfo.processInfo.physicalMemory))
        Memory.cacheLimit = callerBudget
        defer { Memory.clearCache(); Memory.cacheLimit = original }

        let vad = try await PyannoteVADModel.fromPretrained(offlineMode: true)
        defer { vad.unload() }
        XCTAssertEqual(Memory.cacheLimit, ceiling)
        var tts: OmniVoiceTTSModel? = try await OmniVoiceTTSModel.fromBundle(directory)
        defer { tts = nil }
        XCTAssertEqual(Memory.cacheLimit, ceiling)
        vad.unload()
        XCTAssertEqual(Memory.cacheLimit, ceiling, "The remaining TTS runtime must keep its lease")

        let reference = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MagpieTTSCoreMLTests/Resources/magpie-hello-world.wav")
        for (text, duration) in [("Hello there.", 1.2), ("This is a longer memory check.", 2.4)] {
            Memory.clearCache()
            // Compare the same loaded weights and seed with a large cache and the default cap.
            Memory.cacheLimit = callerBudget
            MLXRandom.seed(42)
            let baseline = try tts!.generate(
                text: text, referenceAudio: reference, referenceText: "Hello world.",
                duration: duration, numSteps: 16)
            Memory.clearCache()
            Memory.cacheLimit = ceiling
            MLXRandom.seed(42)
            let capped = try tts!.generate(
                text: text, referenceAudio: reference, referenceText: "Hello world.",
                duration: duration, numSteps: 16)
            XCTAssertFalse(capped.isEmpty)
            XCTAssertTrue(capped.allSatisfy(\.isFinite))
            XCTAssertGreaterThan(capped.map { abs($0) }.max() ?? 0, 0.001)
            XCTAssertTrue(capped.elementsEqual(baseline, by: { $0.bitPattern == $1.bitPattern }), text)
            XCTAssertEqual(Memory.cacheLimit, ceiling)
            // MLX trims on allocation. Its last freed buffer can temporarily
            // overshoot the budget until the next allocation reclaims it.
            let allocation = MLXArray.zeros([1])
            eval(allocation)
            withExtendedLifetime(allocation) {
                XCTAssertLessThanOrEqual(Memory.cacheMemory, ceiling)
            }
        }
        withExtendedLifetime(tts) {}
        tts = nil
        XCTAssertEqual(Memory.cacheLimit, callerBudget, "Last owner destruction restores the caller's budget")
    }

    func testDisabledCallerCachingSurvivesRealModelLoadInferenceAndUnload() async throws {
        _ = try cachedBundle(PyannoteVADModel.defaultModelId, required: "model.safetensors")
        let original = Memory.cacheLimit
        Memory.cacheLimit = 0
        defer { Memory.cacheLimit = original }
        let vad = try await PyannoteVADModel.fromPretrained(offlineMode: true)
        defer { vad.unload() }
        XCTAssertEqual(Memory.cacheLimit, 0)
        let segments = vad.detectSpeech(audio: [Float](repeating: 0, count: 32_000), sampleRate: 16_000)
        XCTAssertTrue(segments.isEmpty)
        XCTAssertEqual(Memory.cacheLimit, 0)
        vad.unload()
        XCTAssertEqual(Memory.cacheLimit, 0)
    }

    func testFailedLocalLoadRestoresCallerBudget() async throws {
        let original = Memory.cacheLimit
        let callerBudget = 8 * gibibyte
        Memory.cacheLimit = callerBudget
        defer { Memory.cacheLimit = original }
        let missingDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            _ = try await GLiNER.load(from: missingDirectory)
            XCTFail("An absent model bundle must fail to load")
        } catch {
            XCTAssertEqual(Memory.cacheLimit, callerBudget)
        }
    }

    func testCoreMLOnlyModelDoesNotChangeMLXBudget() async throws {
        _ = try cachedBundle(SileroVADModel.defaultCoreMLModelId, required: "silero_vad.mlmodelc/coremldata.bin")
        let original = Memory.cacheLimit
        let callerBudget = 8 * gibibyte
        Memory.cacheLimit = callerBudget
        defer { Memory.cacheLimit = original }
        let vad = try await SileroVADModel.fromPretrained(engine: .coreml, offlineMode: true)
        defer { vad.unload() }
        XCTAssertEqual(Memory.cacheLimit, callerBudget)
        vad.unload()
        XCTAssertEqual(Memory.cacheLimit, callerBudget)
    }

    private func cachedBundle(_ modelID: String, required file: String) throws -> URL {
        let directory = try HuggingFaceDownloader.getCacheDirectory(for: modelID)
        guard FileManager.default.fileExists(atPath: directory.appendingPathComponent(file).path) else {
            throw XCTSkip("Model bundle is not cached: \(modelID)")
        }
        return directory
    }
}

import MLX
import MLXCommon
import XCTest

@testable import ChatterboxTTS

final class ChatterboxMemoryOptionsTests: XCTestCase {
    func testBalancedDefaultsBoundCacheAndClearStages() {
        let options = ChatterboxMemoryOptions.balanced

        XCTAssertEqual(options.cacheLimitBytes, 512 * 1024 * 1024)
        XCTAssertTrue(options.clearCacheBetweenStages)
        XCTAssertTrue(options.clearCacheOnCompletion)
    }

    func testUnrestrictedPreservesMLXCacheBehavior() {
        let options = ChatterboxMemoryOptions.unrestricted

        XCTAssertNil(options.cacheLimitBytes)
        XCTAssertFalse(options.clearCacheBetweenStages)
        XCTAssertFalse(options.clearCacheOnCompletion)
    }

    func testWithOptionsRestoresPriorMLXCacheLimit() {
        let priorCap = Memory.cacheLimit
        defer { Memory.cacheLimit = priorCap }

        let initialCap = 3 * 1024 * 1024 * 1024
        let boundedCap = 512 * 1024 * 1024
        Memory.cacheLimit = initialCap

        ChatterboxMemory.withOptions(.balanced) {
            XCTAssertEqual(Memory.cacheLimit, boundedCap)
        }

        XCTAssertEqual(Memory.cacheLimit, initialCap)
    }

    func testWithOptionsDoesNotRaiseStricterCallerCacheLimit() {
        let priorCap = Memory.cacheLimit
        defer { Memory.cacheLimit = priorCap }

        let stricterCap = 256 * 1024 * 1024
        Memory.cacheLimit = stricterCap

        ChatterboxMemory.withOptions(.balanced) {
            XCTAssertEqual(Memory.cacheLimit, stricterCap)
        }

        XCTAssertEqual(Memory.cacheLimit, stricterCap)
    }

    func testModelLoadedDuringGenerationKeepsItsCapAfterGenerationReturns() {
        let priorCap = Memory.cacheLimit
        defer { Memory.cacheLimit = priorCap }
        let callerBudget = 8 * 1024 * 1024 * 1024
        Memory.cacheLimit = callerBudget
        var model: MLXCacheLimitCoordinator.Lease?
        defer { model?.release() }
        ChatterboxMemory.withOptions(.balanced) {
            model = MLXCacheLimitCoordinator.shared.acquireDefault()
            XCTAssertEqual(Memory.cacheLimit, 512 * 1024 * 1024)
        }
        XCTAssertEqual(Memory.cacheLimit, MLXCacheLimitPolicy.defaultCeiling(
            physicalMemoryBytes: Int(ProcessInfo.processInfo.physicalMemory)))
        model?.release()
        XCTAssertEqual(Memory.cacheLimit, callerBudget)
    }

    func testLastModelUnloadedDuringGenerationDoesNotLeaveItsCapBehind() {
        let priorCap = Memory.cacheLimit
        defer { Memory.cacheLimit = priorCap }
        let callerBudget = 8 * 1024 * 1024 * 1024
        Memory.cacheLimit = callerBudget
        let model = MLXCacheLimitCoordinator.shared.acquireDefault()
        defer { model?.release() }
        ChatterboxMemory.withOptions(.balanced) {
            model?.release()
            XCTAssertEqual(Memory.cacheLimit, 512 * 1024 * 1024)
        }
        XCTAssertEqual(Memory.cacheLimit, callerBudget)
    }

    func testZeroGenerationBudgetRestoresAfterThrowing() {
        enum GenerationError: Error { case failed }
        let priorCap = Memory.cacheLimit
        defer { Memory.cacheLimit = priorCap }
        Memory.cacheLimit = 128 * 1024 * 1024
        XCTAssertThrowsError(try ChatterboxMemory.withOptions(.init(cacheLimitBytes: 0)) {
            XCTAssertEqual(Memory.cacheLimit, 0)
            throw GenerationError.failed
        })
        XCTAssertEqual(Memory.cacheLimit, 128 * 1024 * 1024)
    }
}

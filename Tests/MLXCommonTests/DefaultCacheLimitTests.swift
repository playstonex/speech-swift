import MLXCommon
import XCTest

final class DefaultCacheLimitTests: XCTestCase {
    private let gibibyte = 1024 * 1024 * 1024

    func testDefaultCeilingScalesWithRAMAndStopsAtTwoGiB() {
        for (ram, expected) in [(1, 0.25), (4, 1), (8, 2), (16, 2), (48, 2), (128, 2)] {
            XCTAssertEqual(
                MLXCacheLimitPolicy.defaultCeiling(physicalMemoryBytes: ram * gibibyte),
                Int(expected * Double(gibibyte)))
        }
        for ram in [Int.min, -1, 0] {
            XCTAssertEqual(MLXCacheLimitPolicy.defaultCeiling(physicalMemoryBytes: ram), 0)
        }
        XCTAssertEqual(MLXCacheLimitPolicy.defaultCeiling(physicalMemoryBytes: Int.max), 2 * gibibyte)
    }

    func testInvalidRAMDoesNotDisableCaching() {
        var budget = gibibyte
        let coordinator = MLXCacheLimitCoordinator(getLimit: { budget }, setLimit: { budget = $0 })
        for ram in [Int.min, -1, 0, 1] {
            XCTAssertNil(coordinator.acquireDefault(physicalMemoryBytes: ram))
            XCTAssertEqual(budget, gibibyte)
        }
    }

    func testDefaultOwnersPreserveLowerCallerBudgetsIncludingZero() {
        for original in [0, gibibyte / 8, gibibyte, 8 * gibibyte] {
            var budget = original
            let coordinator = MLXCacheLimitCoordinator(getLimit: { budget }, setLimit: { budget = $0 })
            let first = coordinator.acquireDefault(physicalMemoryBytes: 48 * gibibyte)
            let second = coordinator.acquireDefault(physicalMemoryBytes: 48 * gibibyte)
            XCTAssertEqual(budget, min(original, 2 * gibibyte))
            first?.release()
            XCTAssertEqual(budget, min(original, 2 * gibibyte))
            second?.release()
            XCTAssertEqual(budget, original)
        }
    }

    func testTemporaryGenerationBudgetCooperatesWithNewModelInEitherUnloadOrder() {
        for unloadModelFirst in [false, true] {
            var budget = 8 * gibibyte
            let coordinator = MLXCacheLimitCoordinator(getLimit: { budget }, setLimit: { budget = $0 })
            let generation = coordinator.acquire(ceiling: gibibyte / 2)
            let model = coordinator.acquireDefault(physicalMemoryBytes: 48 * gibibyte)
            XCTAssertEqual(budget, gibibyte / 2)
            if unloadModelFirst {
                model?.release()
                XCTAssertEqual(budget, gibibyte / 2)
                generation?.release()
            } else {
                generation?.release()
                XCTAssertEqual(budget, 2 * gibibyte)
                model?.release()
            }
            XCTAssertEqual(budget, 8 * gibibyte)
        }
    }

    func testZeroCeilingDisablesCachingUntilItsOwnerReleases() {
        var budget = 8 * gibibyte
        let coordinator = MLXCacheLimitCoordinator(getLimit: { budget }, setLimit: { budget = $0 })
        let model = coordinator.acquireDefault(physicalMemoryBytes: 48 * gibibyte)
        let noCaching = coordinator.acquire(ceiling: 0)
        XCTAssertNotNil(noCaching)
        XCTAssertEqual(budget, 0)
        noCaching?.release()
        XCTAssertEqual(budget, 2 * gibibyte)
        model?.release()
        XCTAssertEqual(budget, 8 * gibibyte)
    }

    func testFailedLoadReleasesItsBudgetAndKeepsOtherModelsCapped() {
        enum LoadError: Error { case invalidWeights }
        var budget = 8 * gibibyte
        let coordinator = MLXCacheLimitCoordinator(getLimit: { budget }, setLimit: { budget = $0 })
        let model = coordinator.acquireDefault(physicalMemoryBytes: 48 * gibibyte)
        func failedLoad() throws {
            let loading = coordinator.acquireDefault(physicalMemoryBytes: 4 * gibibyte)
            defer { loading?.release() }
            XCTAssertEqual(budget, gibibyte)
            throw LoadError.invalidWeights
        }
        XCTAssertThrowsError(try failedLoad())
        XCTAssertEqual(budget, 2 * gibibyte)
        model?.release()
        XCTAssertEqual(budget, 8 * gibibyte)
    }

    func testCallerChangeDuringUseSurvivesLastOwnerDestruction() {
        var budget = 8 * gibibyte
        let coordinator = MLXCacheLimitCoordinator(getLimit: { budget }, setLimit: { budget = $0 })
        var model = coordinator.acquireDefault(physicalMemoryBytes: 48 * gibibyte)
        budget = gibibyte / 8
        withExtendedLifetime(model) {}
        model = nil
        XCTAssertEqual(budget, gibibyte / 8)
    }
}

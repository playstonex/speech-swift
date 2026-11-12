import Foundation
import MLXCommon
import XCTest
@testable import MossTranscribe

final class MossMLXCacheLimitTests: XCTestCase {
    private let gibibyte = 1024 * 1024 * 1024

    func testCacheCeilingLeavesRoomForLiveLongContextTensors() {
        for (ram, expected) in [(4, 1), (8, 2), (16, 2), (24, 2), (48, 2), (128, 2)] {
            XCTAssertEqual(
                MossMLXMemory.cacheLimit(physicalMemoryBytes: ram * gibibyte),
                expected * gibibyte,
                "at \(ram) GiB RAM"
            )
        }
    }

    func testInvalidMemorySizeDoesNotOverflowOrProduceANegativeCeiling() {
        for physicalMemory in [Int.min, -1, 0] {
            XCTAssertEqual(MossMLXMemory.cacheLimit(physicalMemoryBytes: physicalMemory), 0)
        }
        XCTAssertEqual(MossMLXMemory.cacheLimit(physicalMemoryBytes: Int.max), 2 * gibibyte)
    }

    func testCallerBudgetIncludingDisabledCachingIsPreserved() {
        for initial in [0, gibibyte / 2, gibibyte, 8 * gibibyte] {
            var budget = initial
            let coordinator = MLXCacheLimitCoordinator(
                getLimit: { budget }, setLimit: { budget = $0 }
            )
            let lease = MossMLXMemory.acquireCacheLimit(
                physicalMemoryBytes: 24 * gibibyte,
                coordinator: coordinator
            )
            XCTAssertEqual(budget, min(initial, 2 * gibibyte))
            lease?.release()
            XCTAssertEqual(budget, initial)
        }
    }

    func testMixedModelCeilingsSurviveEitherReleaseOrder() {
        for otherCeiling in [gibibyte, 4 * gibibyte] {
            for releaseMossFirst in [false, true] {
                var budget = 8 * gibibyte
                let coordinator = MLXCacheLimitCoordinator(
                    getLimit: { budget }, setLimit: { budget = $0 }
                )
                let moss = MossMLXMemory.acquireCacheLimit(
                    physicalMemoryBytes: 24 * gibibyte,
                    coordinator: coordinator
                )
                let other = coordinator.acquire(ceiling: otherCeiling)
                XCTAssertEqual(budget, min(2 * gibibyte, otherCeiling))
                if releaseMossFirst {
                    moss?.release()
                    XCTAssertEqual(budget, otherCeiling)
                    other?.release()
                } else {
                    other?.release()
                    XCTAssertEqual(budget, 2 * gibibyte)
                    moss?.release()
                }
                XCTAssertEqual(budget, 8 * gibibyte)
            }
        }
    }

    func testCallerTighteningBudgetDuringUseSurvivesDestruction() {
        var budget = 8 * gibibyte
        let coordinator = MLXCacheLimitCoordinator(
            getLimit: { budget }, setLimit: { budget = $0 }
        )
        var lease = MossMLXMemory.acquireCacheLimit(
            physicalMemoryBytes: 24 * gibibyte,
            coordinator: coordinator
        )
        XCTAssertEqual(budget, 2 * gibibyte)
        budget = gibibyte / 2
        withExtendedLifetime(lease) {}
        lease = nil
        XCTAssertEqual(budget, gibibyte / 2)
    }
}

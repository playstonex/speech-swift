import MLXCommon
import XCTest
@testable import VoxCPM2TTS

final class VoxCPM2MemoryTests: XCTestCase {
    private let gibibyte = 1024 * 1024 * 1024

    func testCacheCeilingIsAQuarterOfRAMUpToTwoGiB() {
        for (ram, expected) in [(4, 1), (8, 2), (16, 2), (48, 2), (128, 2)] {
            XCTAssertEqual(VoxCPM2Memory.cacheLimit(physicalMemoryBytes: ram * gibibyte), expected * gibibyte, "at \(ram) GiB RAM")
        }
    }

    func testInvalidMemorySizeGivesNoCeiling() {
        for physicalMemory in [Int.min, -1, 0] {
            XCTAssertEqual(VoxCPM2Memory.cacheLimit(physicalMemoryBytes: physicalMemory), 0)
        }
        XCTAssertEqual(VoxCPM2Memory.cacheLimit(physicalMemoryBytes: Int.max), 2 * gibibyte)
    }

    func testCallerBudgetIncludingDisabledCachingIsPreserved() {
        for initial in [0, gibibyte / 2, gibibyte, 8 * gibibyte] {
            var budget = initial
            let coordinator = MLXCacheLimitCoordinator(getLimit: { budget }, setLimit: { budget = $0 })
            let lease = VoxCPM2Memory.acquireCacheLimit(physicalMemoryBytes: 48 * gibibyte, coordinator: coordinator)
            XCTAssertEqual(budget, min(initial, 2 * gibibyte))
            lease?.release()
            XCTAssertEqual(budget, initial)
        }
    }
}

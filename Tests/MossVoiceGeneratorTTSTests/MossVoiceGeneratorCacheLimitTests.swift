import MLXCommon
import XCTest
@testable import MossVoiceGeneratorTTS

/// Uses injected integer accessors, without touching the real MLX allocator.
final class MossVoiceGeneratorCacheLimitTests: XCTestCase {
    private let mib = 1024 * 1024
    private let gib = 1024 * 1024 * 1024

    func testLoadingAndInstanceLeasesRestoreOtherOwnersWithoutChangingDefault() {
        var limit = 8 * gib
        let coordinator = MLXCacheLimitCoordinator(getLimit: { limit }, setLimit: { limit = $0 })
        let otherModel = coordinator.acquireDefault(physicalMemoryBytes: 48 * gib)
        XCTAssertEqual(limit, 2 * gib)
        let loading = MVGCachePolicy.acquire(using: coordinator)
        let instance = MVGCachePolicy.acquire(using: coordinator)
        XCTAssertNotNil(loading)
        XCTAssertNotNil(instance)
        XCTAssertEqual(limit, 512 * mib)
        loading?.release()
        XCTAssertEqual(limit, 512 * mib)
        instance?.release()
        XCTAssertEqual(limit, 2 * gib)
        instance?.release()
        XCTAssertEqual(limit, 2 * gib)
        otherModel?.release()
        XCTAssertEqual(limit, 8 * gib)
        XCTAssertEqual(MLXCacheLimitPolicy.defaultCeiling(physicalMemoryBytes: 48 * gib), 2 * gib)
    }

    func testOtherModelUnloadingFirstKeepsMossCeilingUntilItsLastLeaseEnds() {
        var limit = 8 * gib
        let coordinator = MLXCacheLimitCoordinator(getLimit: { limit }, setLimit: { limit = $0 })
        let otherModel = coordinator.acquireDefault(physicalMemoryBytes: 48 * gib)
        let loading = MVGCachePolicy.acquire(using: coordinator)
        let instance = MVGCachePolicy.acquire(using: coordinator)
        otherModel?.release()
        loading?.release()
        XCTAssertEqual(limit, 512 * mib)
        instance?.release()
        XCTAssertEqual(limit, 8 * gib)
    }

    func testTighterCallerBudgetsAreNeverRaised() {
        for original in [0, 128 * mib, 256 * mib] {
            var limit = original
            let coordinator = MLXCacheLimitCoordinator(getLimit: { limit }, setLimit: { limit = $0 })
            let otherModel = coordinator.acquireDefault(physicalMemoryBytes: 48 * gib)
            let loading = MVGCachePolicy.acquire(using: coordinator)
            let instance = MVGCachePolicy.acquire(using: coordinator)
            XCTAssertEqual(limit, original)
            loading?.release()
            instance?.release()
            otherModel?.release()
            XCTAssertEqual(limit, original)
        }
    }

    func testCallerChangesSurviveMossAndOtherOwnerRelease() {
        for callerLimit in [128 * mib, gib] {
            var limit = 8 * gib
            let coordinator = MLXCacheLimitCoordinator(getLimit: { limit }, setLimit: { limit = $0 })
            let otherModel = coordinator.acquireDefault(physicalMemoryBytes: 48 * gib)
            let instance = MVGCachePolicy.acquire(using: coordinator)
            limit = callerLimit
            instance?.release()
            XCTAssertEqual(limit, callerLimit)
            otherModel?.release()
            XCTAssertEqual(limit, callerLimit)
        }
    }

    func testFailedLoadReleasesOnlyItsLease() {
        enum LoadFailure: Error { case expected }
        var limit = 8 * gib
        let coordinator = MLXCacheLimitCoordinator(getLimit: { limit }, setLimit: { limit = $0 })
        let otherModel = coordinator.acquireDefault(physicalMemoryBytes: 48 * gib)
        func failedLoad() throws {
            let loading = MVGCachePolicy.acquire(using: coordinator)
            defer { loading?.release() }
            XCTAssertEqual(limit, 512 * mib)
            throw LoadFailure.expected
        }
        XCTAssertThrowsError(try failedLoad())
        XCTAssertEqual(limit, 2 * gib)
        otherModel?.release()
        XCTAssertEqual(limit, 8 * gib)
    }
}

import Foundation
import MLXCommon

enum MossMLXMemory {
    /// Bound reusable buffers independently of the live long-context KV cache.
    /// Clearing buffers between encoder batches does not bound decoder caching.
    static func cacheLimit(physicalMemoryBytes: Int) -> Int {
        MLXCacheLimitPolicy.defaultCeiling(physicalMemoryBytes: physicalMemoryBytes)
    }

    static func acquireCacheLimit(
        physicalMemoryBytes: Int = Int(ProcessInfo.processInfo.physicalMemory),
        coordinator: MLXCacheLimitCoordinator = .shared
    ) -> MLXCacheLimitCoordinator.Lease? {
        coordinator.acquireDefault(physicalMemoryBytes: physicalMemoryBytes)
    }
}

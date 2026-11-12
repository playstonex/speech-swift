import MLXCommon

/// Spare allocator buffers share one process-wide budget. MOSS registers a
/// ceiling instead of clearing buffers another loaded model may reuse.
enum MVGCachePolicy {
    static let ceilingBytes = 512 * 1024 * 1024

    static func acquire(
        using coordinator: MLXCacheLimitCoordinator = .shared
    ) -> MLXCacheLimitCoordinator.Lease? {
        coordinator.acquire(ceiling: ceilingBytes)
    }
}

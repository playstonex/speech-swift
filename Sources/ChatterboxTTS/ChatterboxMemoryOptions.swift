import Foundation
import MLX
import MLXCommon

public struct ChatterboxMemoryOptions: Sendable, Equatable {
    /// Temporary MLX cache limit while running Chatterbox generation.
    /// `nil` preserves the caller's existing global cache limit.
    public var cacheLimitBytes: Int?

    /// Clear reusable MLX buffers between the autoregressive T3 stage, S3Gen
    /// flow stage, and vocoder stage. Active tensors are retained.
    public var clearCacheBetweenStages: Bool

    /// Clear reusable MLX buffers before returning generated audio.
    public var clearCacheOnCompletion: Bool

    public init(
        cacheLimitBytes: Int? = 512 * 1024 * 1024,
        clearCacheBetweenStages: Bool = true,
        clearCacheOnCompletion: Bool = true
    ) {
        self.cacheLimitBytes = cacheLimitBytes
        self.clearCacheBetweenStages = clearCacheBetweenStages
        self.clearCacheOnCompletion = clearCacheOnCompletion
    }

    /// Memory-conscious default for app/runtime use.
    public static let balanced = ChatterboxMemoryOptions()

    /// Skip the temporary generation cap and stage clears. The model's default
    /// lifetime cap and any caller budget still apply.
    public static let unrestricted = ChatterboxMemoryOptions(
        cacheLimitBytes: nil,
        clearCacheBetweenStages: false,
        clearCacheOnCompletion: false
    )
}

enum ChatterboxMemory {
    private static let lock = NSRecursiveLock()

    static func withOptions<T>(_ options: ChatterboxMemoryOptions, _ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }

        let cacheLimitLease = options.cacheLimitBytes.flatMap {
            MLXCacheLimitCoordinator.shared.acquire(ceiling: max(0, $0))
        }
        if options.cacheLimitBytes != nil {
            Memory.clearCache()
        }
        defer {
            if options.clearCacheOnCompletion {
                Memory.clearCache()
            }
            cacheLimitLease?.release()
        }
        return try body()
    }

    static func clearStageCache(_ options: ChatterboxMemoryOptions) {
        if options.clearCacheBetweenStages {
            Memory.clearCache()
        }
    }
}

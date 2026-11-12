import MLX

/// Workspace reuse policy for Fish Audio Drama 3. Trimming frees idle MLX
/// buffers at stage boundaries; it leaves active tensors and global limits intact.
public struct FishAudioMemoryOptions: Sendable, Equatable {
    public var cacheTrimThresholdBytes: Int?
    public var lazyReferenceEncoder: Bool

    public init(cacheTrimThresholdBytes: Int? = 512 * 1024 * 1024,
                lazyReferenceEncoder: Bool = true) {
        self.cacheTrimThresholdBytes = cacheTrimThresholdBytes
        self.lazyReferenceEncoder = lazyReferenceEncoder
    }

    public static let balanced = FishAudioMemoryOptions()
    public static let unrestricted = FishAudioMemoryOptions(cacheTrimThresholdBytes: nil, lazyReferenceEncoder: false)

    public func validate() throws {
        if let threshold = cacheTrimThresholdBytes, threshold < 0 {
            throw FishAudioError.invalidInput("Fish Audio cache trimming threshold must be nonnegative")
        }
    }

    func trimCache(cacheBytes: () -> Int = { Memory.cacheMemory },
                   clearCache: () -> Void = { Memory.clearCache() }) {
        guard let threshold = cacheTrimThresholdBytes else { return }
        if cacheBytes() > threshold { clearCache() }
    }

    func evaluateStage(_ array: MLXArray) {
        guard cacheTrimThresholdBytes != nil else { return }
        eval(array)
        trimCache()
    }
}

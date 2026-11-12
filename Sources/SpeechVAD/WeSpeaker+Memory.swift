import AudioCommon

extension WeSpeakerModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        network?.clearParameters()
        #if canImport(CoreML)
        coremlModel = nil
        #endif
        mlxCacheLimitLease?.release()
        mlxCacheLimitLease = nil
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        return network?.parameterMemoryBytes() ?? 0
    }
}

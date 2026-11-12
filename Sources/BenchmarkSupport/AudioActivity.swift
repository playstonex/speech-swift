/// Locates the first complete PCM window whose RMS reaches a fixed threshold.
/// This measures waveform activity for latency benchmarks, rather than speech recognition.
public struct BenchmarkAudioActivity {
    public let windowSamples: Int
    public let rmsThreshold: Float
    public private(set) var firstActiveSample: Int?
    public private(set) var processedSamples = 0
    private var windowCount = 0
    private var squaredSum: Double = 0

    public init(windowSamples: Int = 240, rmsThreshold: Float = 0.0031622777) {
        precondition(windowSamples > 0 && rmsThreshold.isFinite && rmsThreshold > 0)
        self.windowSamples = windowSamples
        self.rmsThreshold = rmsThreshold
    }

    public mutating func append(_ samples: [Float]) {
        guard firstActiveSample == nil else {
            processedSamples += samples.count
            return
        }
        for (index, sample) in samples.enumerated() {
            processedSamples += 1
            windowCount += 1
            squaredSum += Double(sample) * Double(sample)
            if windowCount == windowSamples {
                if squaredSum / Double(windowSamples) >= Double(rmsThreshold) * Double(rmsThreshold) {
                    firstActiveSample = processedSamples - windowSamples
                    processedSamples += samples.count - index - 1
                    return
                }
                windowCount = 0
                squaredSum = 0
            }
        }
    }
}

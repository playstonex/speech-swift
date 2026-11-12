/// Retrospective scheduling estimates from decoded PCM chunk arrival times.
/// Audio rendering and output-device latency are outside these estimates.
public struct BenchmarkStreamingLatency {
    public let sampleRate: Int
    public let prebufferSamples: Int
    public private(set) var scheduledSamples = 0
    public private(set) var prebufferReadySeconds: Double?
    public private(set) var minimumContinuousStartSeconds = 0.0

    public var additionalBufferSeconds: Double? {
        prebufferReadySeconds.map { max(0, minimumContinuousStartSeconds - $0) }
    }

    public init(sampleRate: Int, prebufferSeconds: Double) {
        precondition(sampleRate > 0 && prebufferSeconds.isFinite && prebufferSeconds >= 0)
        let count = prebufferSeconds * Double(sampleRate)
        precondition(count.isFinite && count < Double(Int.max))
        self.sampleRate = sampleRate
        prebufferSamples = Int(count)
    }

    public mutating func append(samples: Int, arrivalSeconds: Double) {
        precondition(samples >= 0 && arrivalSeconds.isFinite && arrivalSeconds >= 0)
        guard samples > 0 else { return }
        minimumContinuousStartSeconds = max(minimumContinuousStartSeconds,
            arrivalSeconds - Double(scheduledSamples) / Double(sampleRate))
        scheduledSamples += samples
        if prebufferReadySeconds == nil && scheduledSamples >= prebufferSamples {
            prebufferReadySeconds = arrivalSeconds
        }
    }

    /// A player can start a short clip once generation finishes, even if it
    /// never reaches the requested prebuffer size.
    public mutating func finish(at seconds: Double) {
        precondition(seconds.isFinite && seconds >= 0)
        if prebufferReadySeconds == nil && scheduledSamples > 0 {
            prebufferReadySeconds = seconds
        }
    }
}

import Foundation

public enum MusicAnalysisError: Error, LocalizedError {
    case invalidOptions(String)
    case invalidSampleRate
    case unsupportedChannels
    case emptyAudio
    case unequalChannels
    case nonFiniteSamples
    case rangeOutsideAudio
    case conversionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidOptions(let message): return message
        case .invalidSampleRate: return "The sample rate must be finite and between 4,000 and 384,000 Hz."
        case .unsupportedChannels: return "Music analysis supports mono or stereo audio."
        case .emptyAudio: return "The audio contains no readable samples."
        case .unequalChannels: return "Audio channels must have the same frame count."
        case .nonFiniteSamples: return "The audio contains non-finite samples."
        case .rangeOutsideAudio: return "The analysis start lies outside the recording."
        case .conversionFailed(let message): return "Audio conversion failed: \(message)"
        }
    }
}

public struct MusicAnalysisOptions: Sendable, Equatable {
    public static let maximumDurationSeconds: Double = 120
    public var startSeconds: Double
    public var maxDurationSeconds: Double

    public init(startSeconds: Double = 0, maxDurationSeconds: Double = 120) {
        self.startSeconds = startSeconds
        self.maxDurationSeconds = maxDurationSeconds
    }

    public func validate() throws {
        guard startSeconds.isFinite, startSeconds >= 0 else {
            throw MusicAnalysisError.invalidOptions("--start must be a finite, nonnegative number of seconds.")
        }
        guard maxDurationSeconds.isFinite, maxDurationSeconds > 0,
              maxDurationSeconds <= Self.maximumDurationSeconds else {
            throw MusicAnalysisError.invalidOptions("--max-seconds must be greater than zero and no more than 120.")
        }
    }
}

public struct MusicAudioMetadata: Codable, Sendable, Equatable {
    public let fileName: String?
    public let sampleRate: Double
    public let channelCount: Int
    public let durationSeconds: Double
}

public struct MusicAnalyzedRange: Codable, Sendable, Equatable {
    public let startSeconds: Double
    public let durationSeconds: Double
    public let analysisSampleRate: Double
    public let analysisFrameCount: Int
    public let isPartial: Bool
}

public struct MusicSignalSummary: Codable, Sendable, Equatable {
    public let rms: Double
    public let peak: Double
}

public struct MusicAnalysis: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let source: MusicAudioMetadata
    public let analyzedRange: MusicAnalyzedRange
    public let signal: MusicSignalSummary
    public let tempo: MusicTempoAnalysis
    public let key: MusicKeyAnalysis
    public let notes: [String]

    /// Deterministic JSON with explicit nulls for unavailable estimates.
    public func jsonData(prettyPrinted: Bool = true) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = prettyPrinted ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
        return try encoder.encode(self)
    }
}

/// Model-free estimates over a bounded window, rather than full-song annotations.
public enum MusicAnalyzer {
    public static func analyze(
        url: URL, options: MusicAnalysisOptions = MusicAnalysisOptions()
    ) throws -> MusicAnalysis {
        try options.validate()
        let decoded = try AnalysisAudioReader.read(url: url, options: options)
        return try analyzeWindow(
            channels: decoded.channels, sampleRate: decoded.analysisSampleRate,
            source: decoded.source, startSeconds: decoded.startSeconds,
            selectedDuration: decoded.selectedDuration, isPartial: decoded.isPartial
        )
    }

    public static func analyze(
        samples: [Float], sampleRate: Double,
        options: MusicAnalysisOptions = MusicAnalysisOptions()
    ) throws -> MusicAnalysis {
        try analyze(channels: [samples], sampleRate: sampleRate, options: options)
    }

    /// Stereo channel powers are combined after the FFT, avoiding cancellation
    /// from out-of-phase left and right channels.
    public static func analyze(
        channels: [[Float]], sampleRate: Double,
        options: MusicAnalysisOptions = MusicAnalysisOptions()
    ) throws -> MusicAnalysis {
        try options.validate()
        try validate(channels: channels, sampleRate: sampleRate)
        let frameCount = channels[0].count
        let duration = Double(frameCount) / sampleRate
        guard options.startSeconds < duration else {
            throw MusicAnalysisError.rangeOutsideAudio
        }
        let start = Int((options.startSeconds * sampleRate).rounded(.down))
        let end = min(frameCount, start + Int((options.maxDurationSeconds * sampleRate).rounded(.down)))
        guard end > start else { throw MusicAnalysisError.emptyAudio }
        return try analyzeWindow(
            channels: channels.map { Array($0[start..<end]) }, sampleRate: sampleRate,
            source: MusicAudioMetadata(
                fileName: nil, sampleRate: sampleRate, channelCount: channels.count,
                durationSeconds: duration
            ),
            startSeconds: Double(start) / sampleRate,
            selectedDuration: Double(end - start) / sampleRate,
            isPartial: start > 0 || end < frameCount
        )
    }

    private static func validate(channels: [[Float]], sampleRate: Double) throws {
        guard sampleRate.isFinite, (4_000.0...384_000.0).contains(sampleRate) else {
            throw MusicAnalysisError.invalidSampleRate
        }
        guard (1...2).contains(channels.count) else {
            throw MusicAnalysisError.unsupportedChannels
        }
        guard let count = channels.first?.count, count > 0 else {
            throw MusicAnalysisError.emptyAudio
        }
        guard channels.allSatisfy({ $0.count == count }) else {
            throw MusicAnalysisError.unequalChannels
        }
        guard channels.allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
            throw MusicAnalysisError.nonFiniteSamples
        }
    }

    private static func analyzeWindow(
        channels: [[Float]], sampleRate: Double, source: MusicAudioMetadata,
        startSeconds: Double, selectedDuration: Double, isPartial: Bool
    ) throws -> MusicAnalysis {
        try validate(channels: channels, sampleRate: sampleRate)
        var squareSum = 0.0
        var peak = 0.0
        for channel in channels {
            for sample in channel {
                let value = Double(sample)
                squareSum += value * value
                peak = max(peak, abs(value))
            }
        }
        let rms = sqrt(squareSum / Double(channels.count * channels[0].count))
        let analysisChannels = peak > 1
            ? channels.map { channel in channel.map { Float(Double($0) / peak) } }
            : channels
        let features: AnalysisSpectralFeatures
        if rms > 0.00001 {
            features = try AnalysisSpectrum.extract(channels: analysisChannels, sampleRate: sampleRate)
        } else {
            features = .empty(sampleRate: sampleRate)
        }
        let tempo = TempoEstimator.estimate(
            novelty: features.novelty, frameRate: features.frameRate,
            durationSeconds: selectedDuration, rms: rms
        )
        let key = KeyEstimator.estimate(chroma: features.chroma, pitchedFrameCount: features.pitchedFrameCount)
        var notes = [
            "Estimates describe the selected window; tempo and key may change elsewhere.",
            "Fit scores describe signal/template agreement, not calibrated probabilities."
        ]
        if isPartial {
            notes.append("Only part of the recording was analyzed; inspect analyzedRange.")
        }
        return MusicAnalysis(
            schemaVersion: 1, source: source,
            analyzedRange: MusicAnalyzedRange(
                startSeconds: startSeconds, durationSeconds: selectedDuration,
                analysisSampleRate: sampleRate, analysisFrameCount: channels[0].count,
                isPartial: isPartial
            ),
            signal: MusicSignalSummary(rms: rms, peak: peak),
            tempo: tempo, key: key, notes: notes
        )
    }
}

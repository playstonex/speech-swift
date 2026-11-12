import Foundation

public struct MusicTempoCandidate: Codable, Sendable, Equatable {
    public let bpm: Double
    public let fitScore: Double
}

public struct MusicTempoAnalysis: Codable, Sendable, Equatable {
    public let estimatedBPM: Double?
    public let candidates: [MusicTempoCandidate]
    public let onsetCount: Int
    public let ambiguous: Bool
    public let method: String
    public let scoreMeaning: String
    public let notes: [String]

    enum CodingKeys: String, CodingKey {
        case estimatedBPM, candidates, onsetCount, ambiguous, method, scoreMeaning, notes
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(estimatedBPM, forKey: .estimatedBPM)
        try container.encode(candidates, forKey: .candidates)
        try container.encode(onsetCount, forKey: .onsetCount)
        try container.encode(ambiguous, forKey: .ambiguous)
        try container.encode(method, forKey: .method)
        try container.encode(scoreMeaning, forKey: .scoreMeaning)
        try container.encode(notes, forKey: .notes)
    }
}

enum TempoEstimator {
    static func estimate(novelty: [Double], frameRate: Double, durationSeconds: Double, rms: Double) -> MusicTempoAnalysis {
        func result(
            _ bpm: Double? = nil, candidates: [MusicTempoCandidate] = [],
            onsetCount: Int = 0, ambiguous: Bool = false, notes: [String]
        ) -> MusicTempoAnalysis {
            MusicTempoAnalysis(
                estimatedBPM: bpm, candidates: candidates, onsetCount: onsetCount,
                ambiguous: ambiguous, method: "spectral_flux_autocorrelation",
                scoreMeaning: "Normalized onset-envelope correlation; not a probability.",
                notes: notes
            )
        }
        guard rms > 0.00001 else {
            return result(notes: ["No meaningful audio energy was found."])
        }
        guard durationSeconds >= 4, novelty.count >= 32 else {
            return result(notes: ["At least four seconds of rhythmic audio are needed for a tempo estimate."])
        }
        let localRadius = max(2, Int(frameRate / 2))
        var prefix = [Double](repeating: 0, count: novelty.count + 1)
        for i in novelty.indices { prefix[i + 1] = prefix[i] + novelty[i] }
        var envelope = [Double](repeating: 0, count: novelty.count)
        for i in novelty.indices {
            let first = max(0, i - localRadius)
            let last = min(novelty.count, i + localRadius + 1)
            let localMean = (prefix[last] - prefix[first]) / Double(last - first)
            envelope[i] = max(0, novelty[i] - localMean)
        }
        guard let maximum = envelope.max(), maximum > 1e-9 else {
            return result(notes: ["No clear onset changes were found."])
        }
        let mean = envelope.reduce(0, +) / Double(envelope.count)
        let deviation = sqrt(envelope.reduce(0) { $0 + pow($1 - mean, 2) } / Double(envelope.count))
        let threshold = max(maximum * 0.10, mean + 0.50 * deviation)
        let minimumSeparation = max(1, Int(frameRate * 0.12))
        var peaks: [Int] = []
        for i in 1..<(envelope.count - 1) {
            guard envelope[i] > threshold, envelope[i] >= envelope[i - 1],
                  envelope[i] > envelope[i + 1] else { continue }
            if let previous = peaks.last, i - previous < minimumSeparation {
                if envelope[i] > envelope[previous] { peaks[peaks.count - 1] = i }
            } else {
                peaks.append(i)
            }
        }
        guard peaks.count >= 4, let firstPeak = peaks.first, let lastPeak = peaks.last,
              Double(lastPeak - firstPeak) / frameRate >= 2 else {
            return result(onsetCount: peaks.count, notes: ["Too few separated onsets were found to establish a pulse."])
        }
        let centered = envelope.map { $0 - mean }
        let minLag = max(2, Int((60 * frameRate / 240).rounded(.down)))
        let maxLag = min(centered.count / 2, Int((60 * frameRate / 40).rounded(.up)))
        guard maxLag > minLag + 2 else {
            return result(onsetCount: peaks.count, notes: ["The window is too short for tempo candidates."])
        }
        var correlations = [Double](repeating: 0, count: maxLag + 1)
        for lag in minLag...maxLag {
            var cross = 0.0
            var aEnergy = 0.0
            var bEnergy = 0.0
            for i in lag..<centered.count {
                let a = centered[i]
                let b = centered[i - lag]
                cross += a * b
                aEnergy += a * a
                bEnergy += b * b
            }
            if aEnergy > 1e-18, bEnergy > 1e-18 {
                correlations[lag] = max(0, min(1, cross / sqrt(aEnergy * bEnergy)))
            }
        }
        let intervals = zip(peaks.dropFirst(), peaks).map { pair in
            Double(pair.0 - pair.1) / frameRate
        }.sorted()
        let trim = intervals.count >= 10 ? intervals.count / 10 : 0
        let centralIntervals = intervals[trim..<(intervals.count - trim)]
        let meanInterval = centralIntervals.reduce(0, +) / Double(centralIntervals.count)
        let intervalDeviation = sqrt(centralIntervals.reduce(0) { $0 + pow($1 - meanInterval, 2) }
                                     / Double(centralIntervals.count))
        let stableEvents = meanInterval > 0 && intervalDeviation / meanInterval < 0.15
        var eventBPM = 60 / meanInterval
        while eventBPM > 240 { eventBPM /= 2 }
        while eventBPM < 40 { eventBPM *= 2 }

        var candidates: [MusicTempoCandidate] = []
        for lag in minLag...maxLag {
            let score = correlations[lag]
            guard score >= 0.25,
                  lag == minLag || score >= correlations[lag - 1],
                  lag == maxLag || score > correlations[lag + 1] else { continue }
            var fractionalLag = Double(lag)
            if lag > minLag, lag < maxLag {
                let left = correlations[lag - 1]
                let right = correlations[lag + 1]
                let denominator = left - 2 * score + right
                if abs(denominator) > 1e-12 {
                    fractionalLag += max(-0.5, min(0.5, 0.5 * (left - right) / denominator))
                }
            }
            let bpm = 60 * frameRate / fractionalLag
            guard (40...240).contains(bpm) else { continue }
            candidates.append(MusicTempoCandidate(bpm: bpm, fitScore: score))
        }
        guard !candidates.isEmpty else {
            return result(onsetCount: peaks.count, notes: ["Onsets were present, but no sufficiently periodic pulse was found."])
        }
        candidates.sort {
            let a = ranking($0, eventBPM: stableEvents ? eventBPM : nil)
            let b = ranking($1, eventBPM: stableEvents ? eventBPM : nil)
            return a == b ? $0.bpm < $1.bpm : a > b
        }
        if stableEvents, abs(log2(candidates[0].bpm / eventBPM)) < 0.08 {
            // Average event spacing avoids quantizing a regular click track to
            // an integer number of FFT hops.
            candidates[0] = MusicTempoCandidate(bpm: eventBPM, fitScore: candidates[0].fitScore)
        }
        candidates = Array(candidates.prefix(5))
        let best = candidates[0]
        let ambiguous = candidates.dropFirst().contains {
            let ratio = $0.bpm / best.bpm
            return (abs(ratio - 0.5) < 0.04 || abs(ratio - 2) < 0.08)
                && $0.fitScore >= best.fitScore * 0.80
        }
        var notes = ["The estimate describes a periodic onset pulse, not a beat-grid or time-signature annotation."]
        if ambiguous {
            notes.append("Half/double tempo candidates have similar support; choose the musical beat by ear.")
        }
        if !stableEvents {
            notes.append("Onset spacing varies; syncopation, rubato or tempo changes can affect this estimate.")
        }
        return result(best.bpm, candidates: candidates, onsetCount: peaks.count, ambiguous: ambiguous, notes: notes)
    }

    private static func ranking(_ candidate: MusicTempoCandidate, eventBPM: Double?) -> Double {
        guard let eventBPM else { return candidate.fitScore }
        return candidate.fitScore - min(0.35, abs(log2(candidate.bpm / eventBPM)) * 0.25)
    }
}

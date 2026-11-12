import Foundation

public struct MusicKeyCandidate: Codable, Sendable, Equatable {
    public let name: String
    public let tonic: String
    public let mode: String
    public let fitScore: Double
}

public struct MusicKeyAnalysis: Codable, Sendable, Equatable {
    public let estimatedKey: String?
    public let candidates: [MusicKeyCandidate]
    public let pitchedFrameCount: Int
    public let ambiguous: Bool
    public let method: String
    public let scoreMeaning: String
    public let notes: [String]

    enum CodingKeys: String, CodingKey {
        case estimatedKey, candidates, pitchedFrameCount, ambiguous, method, scoreMeaning, notes
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(estimatedKey, forKey: .estimatedKey)
        try container.encode(candidates, forKey: .candidates)
        try container.encode(pitchedFrameCount, forKey: .pitchedFrameCount)
        try container.encode(ambiguous, forKey: .ambiguous)
        try container.encode(method, forKey: .method)
        try container.encode(scoreMeaning, forKey: .scoreMeaning)
        try container.encode(notes, forKey: .notes)
    }
}

enum KeyEstimator {
    // Krumhansl–Kessler major/minor pitch-class profiles.
    private static let major = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    private static let minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    private static let names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

    static func estimate(chroma: [Double], pitchedFrameCount: Int) -> MusicKeyAnalysis {
        func result(
            _ estimated: String? = nil, candidates: [MusicKeyCandidate] = [],
            ambiguous: Bool = false, notes: [String]
        ) -> MusicKeyAnalysis {
            MusicKeyAnalysis(
                estimatedKey: estimated, candidates: candidates,
                pitchedFrameCount: pitchedFrameCount, ambiguous: ambiguous,
                method: "chroma_profile_correlation",
                scoreMeaning: "Correlation with major/minor pitch-class templates; not a probability.",
                notes: notes
            )
        }
        guard chroma.count == 12, let peak = chroma.max(), peak > 1e-12,
              pitchedFrameCount >= 3 else {
            return result(notes: ["No sustained pitched evidence was found."])
        }
        let active = chroma.filter { $0 >= peak * 0.15 }.count
        guard active >= 3 else {
            return result(notes: ["Too few pitch classes were found to distinguish a major/minor key."])
        }
        var candidates: [MusicKeyCandidate] = []
        for tonic in 0..<12 {
            for (mode, profile) in [("major", major), ("minor", minor)] {
                let rotated = (0..<12).map { profile[($0 - tonic + 12) % 12] }
                let score = correlation(chroma, rotated)
                if score > 0 {
                    candidates.append(MusicKeyCandidate(
                        name: "\(names[tonic]) \(mode)", tonic: names[tonic],
                        mode: mode, fitScore: score
                    ))
                }
            }
        }
        candidates.sort { $0.fitScore == $1.fitScore ? $0.name < $1.name : $0.fitScore > $1.fitScore }
        candidates = Array(candidates.prefix(3))
        guard let first = candidates.first, first.fitScore >= 0.60 else {
            return result(candidates: candidates, notes: ["Pitch-class evidence does not fit a major/minor template strongly enough."])
        }
        let margin = candidates.count > 1 ? first.fitScore - candidates[1].fitScore : 1
        let ambiguous = margin < 0.08
        var notes = ["A window-level major/minor estimate; not chord detection, note transcription or tuning analysis."]
        if ambiguous {
            notes.append("The leading templates are close, including possible relative keys; no single key is selected.")
        }
        return result(
            ambiguous ? nil : first.name, candidates: candidates, ambiguous: ambiguous,
            notes: notes
        )
    }

    private static func correlation(_ a: [Double], _ b: [Double]) -> Double {
        let meanA = a.reduce(0, +) / 12
        let meanB = b.reduce(0, +) / 12
        var cross = 0.0
        var energyA = 0.0
        var energyB = 0.0
        for i in 0..<12 {
            let x = a[i] - meanA
            let y = b[i] - meanB
            cross += x * y
            energyA += x * x
            energyB += y * y
        }
        guard energyA > 1e-18, energyB > 1e-18 else { return 0 }
        return max(-1, min(1, cross / sqrt(energyA * energyB)))
    }
}

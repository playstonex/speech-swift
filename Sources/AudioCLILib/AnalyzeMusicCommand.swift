import ArgumentParser
import Foundation
import MusicTools

public struct AnalyzeMusicCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "analyze-music",
        abstract: "Estimate tempo and major/minor key over a bounded audio window"
    )

    @Argument(help: "Mono or stereo audio file supported by the system decoder")
    public var input: String

    @Option(name: .long, help: "Start of the analysis window in seconds")
    public var start: Double = 0

    @Option(name: .long, help: "Maximum seconds to analyze, greater than zero and at most 120")
    public var maxSeconds: Double = 120

    @Flag(name: .long, help: "Write deterministic JSON to stdout, including null estimates when evidence is weak")
    public var json: Bool = false

    public init() {}

    public func validate() throws {
        do {
            try MusicAnalysisOptions(startSeconds: start, maxDurationSeconds: maxSeconds).validate()
        } catch {
            throw ValidationError(error.localizedDescription)
        }
    }

    public func run() throws {
        let analysis = try MusicAnalyzer.analyze(
            url: URL(fileURLWithPath: input),
            options: MusicAnalysisOptions(startSeconds: start, maxDurationSeconds: maxSeconds)
        )
        if json {
            var data = try analysis.jsonData()
            data.append(0x0A)
            FileHandle.standardOutput.write(data)
            return
        }
        print("Analyzed \(String(format: "%.2f", analysis.analyzedRange.durationSeconds))s"
              + " from \(String(format: "%.2f", analysis.analyzedRange.startSeconds))s"
              + " of \(String(format: "%.2f", analysis.source.durationSeconds))s")
        if analysis.analyzedRange.isPartial {
            print("Window only: the rest of the recording was not analyzed.")
        }
        if let bpm = analysis.tempo.estimatedBPM {
            print("Tempo estimate: \(String(format: "%.1f", bpm)) BPM"
                  + (analysis.tempo.ambiguous ? " (half/double tempo is ambiguous)" : ""))
        } else {
            print("Tempo estimate: unavailable")
        }
        for candidate in analysis.tempo.candidates {
            print("  \(String(format: "%.1f", candidate.bpm)) BPM"
                  + " — periodicity fit \(String(format: "%.3f", candidate.fitScore))")
        }
        print("Key estimate: \(analysis.key.estimatedKey ?? "unavailable")"
              + (analysis.key.ambiguous ? " (ambiguous templates)" : ""))
        for candidate in analysis.key.candidates {
            print("  \(candidate.name) — profile fit \(String(format: "%.3f", candidate.fitScore))")
        }
        for note in analysis.tempo.notes + analysis.key.notes + analysis.notes {
            print(note)
        }
    }
}

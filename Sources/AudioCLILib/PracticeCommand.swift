import ArgumentParser
import Foundation
import MusicTools

public struct PracticeCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "practice",
        abstract: "Export a trimmed, repeated music passage with independent speed and pitch"
    )

    @Argument(help: "Input audio file supported by AVFoundation")
    public var input: String

    @Option(name: .long, parsing: .unconditional, help: "Passage start in seconds")
    public var start: Double = 0

    @Option(name: .long, parsing: .unconditional, help: "Passage duration in seconds, at most 600 (default: remaining input)")
    public var duration: Double?

    @Option(name: .long, parsing: .unconditional, help: "Playback speed multiplier, 0.25 to 2.0; pitch is preserved")
    public var speed: Double = 1

    @Option(name: .long, parsing: .unconditional, help: "Independent pitch shift, -12 to 12 semitones")
    public var semitones: Double = 0

    @Option(name: .long, parsing: .unconditional, help: "Number of repetitions, 1 to 100; completed output is at most one hour")
    public var loops: Int = 1

    @Option(name: .long, parsing: .unconditional, help: "Fade at each passage edge, 0 to 50 milliseconds")
    public var fadeMs: Double = 5

    @Option(name: .shortAndLong, help: "Output .wav path (required)")
    public var output: String

    @Flag(name: .long, help: "Replace an existing output file after successful processing")
    public var overwrite = false

    @Flag(name: .long, help: "Write the completed export report as JSON to stdout")
    public var json = false

    public init() {}

    public func validate() throws {
        do { try options.validate() } catch { throw ValidationError(error.localizedDescription) }
        guard URL(fileURLWithPath: output).pathExtension.lowercased() == "wav" else {
            throw ValidationError("The output filename must end in .wav.")
        }
    }

    public func run() throws {
        let result = try MusicPracticeExporter.export(
            input: URL(fileURLWithPath: input), output: URL(fileURLWithPath: output), options: options)
        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            print(String(decoding: try encoder.encode(result), as: UTF8.self))
        } else {
            print("Saved \(result.outputPath)")
            print("\(String(format: "%.3f", result.outputDurationSeconds))s · \(result.channels) channel(s) · \(Int(result.sampleRate)) Hz · \(result.loops) repetition(s)")
            print("Speed \(result.speed)× · Pitch \(result.semitones) semitone(s) · Gain \(String(format: "%.4f", result.appliedGain))")
        }
    }

    private var options: MusicPracticeOptions {
        MusicPracticeOptions(
            startSeconds: start, durationSeconds: duration, speed: speed,
            semitones: semitones, loops: loops, fadeMilliseconds: fadeMs, overwrite: overwrite)
    }
}

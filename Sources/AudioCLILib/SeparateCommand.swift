import Foundation
import ArgumentParser
import SourceSeparation
import AudioCommon
import MLX

public struct SeparateCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "separate",
        abstract: "Separate a music track into stems (vocals, drums, bass, other)"
    )

    @Argument(help: "Input audio file (WAV, stereo 44.1kHz)")
    public var input: String

    @Option(name: .long, help: "Output directory (default: <input>_stems/)")
    public var outputDir: String?

    @Option(name: .long, help: "Stems to extract: vocals,drums,bass,other (default: all)")
    public var stems: String?

    @Flag(name: .long, help: "Also save instrumental.wav from drums+bass+other; attenuate only if the mix peak exceeds 1")
    public var instrumental: Bool = false

    @Option(name: .long, help: "Engine: umx (default) or htdemucs (Demucs v4, higher quality)")
    public var engine: String = "umx"

    @Option(name: .long, help: "Model variant: hq (default, 8.9M/stem) or l (28.3M/stem)")
    public var model: String = "hq"

    @Option(name: .long, help: "Local htdemucs weights dir (htdemucs_ft.safetensors + _config.json)")
    public var htdemucsDir: String?

    @Option(name: .long, help: "HTDemucs precision: fp16 (default) or int8 (smaller download)")
    public var htdemucsPrecision: String = "fp16"

    @Flag(name: .long, help: "Show timing info")
    public var verbose: Bool = false

    public init() {}

    public func validate() throws {
        _ = try outputTargets()
    }

    func outputTargets() throws -> [SeparationTarget] {
        guard let stems else { return SeparationTarget.allCases }
        var targets: [SeparationTarget] = []
        for value in stems.split(separator: ",", omittingEmptySubsequences: false) {
            let name = value.trimmingCharacters(in: .whitespaces)
            guard let target = SeparationTarget(rawValue: name) else {
                throw ValidationError("Invalid stem '\(name)'. Choose vocals, drums, bass or other.")
            }
            if !targets.contains(target) { targets.append(target) }
        }
        return targets
    }

    func inferenceTargets() throws -> [SeparationTarget] {
        let selected = try outputTargets()
        // Keep vocals in UMX's Wiener denominator when producing a backing mix.
        return instrumental ? SeparationTarget.allCases : selected
    }

    public func run() throws {
        let inputURL = URL(fileURLWithPath: input)
        let baseName = inputURL.deletingPathExtension().lastPathComponent
        let outDir = URL(fileURLWithPath: outputDir ?? "\(baseName)_stems")

        let targets = try outputTargets()
        let extractionTargets = try inferenceTargets()

        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        if engine.lowercased() == "htdemucs" {
            try runHTDemucs(inputURL: inputURL, outDir: outDir, targets: targets)
            return
        }

        try runAsync {
            let modelId = model.lowercased() == "l" ? SourceSeparator.largeModelId : SourceSeparator.defaultModelId
            print("Loading model (\(model.lowercased() == "l" ? "UMX-L" : "UMX-HQ"))...")
            let separator = try await SourceSeparator.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            print("Loading audio: \(input)")
            let audio = try AudioFileLoader.loadStereo(url: inputURL, targetSampleRate: 44100, quality: .mastering)
            let duration = Double(audio[0].count) / 44100.0
            print("  Duration: \(String(format: "%.1f", duration))s")

            print("Separating into \(targets.map(\.rawValue).joined(separator: ", "))...")
            let startTime = CFAbsoluteTimeGetCurrent()

            let results = separator.separate(audio: audio, targets: extractionTargets)

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / duration

            try writeOutputs(results, targets: targets, to: outDir)

            print("Done in \(String(format: "%.1f", elapsed))s (RTF: \(String(format: "%.2f", rtf)))")
        }
    }

    private func runHTDemucs(inputURL: URL, outDir: URL, targets: [SeparationTarget]) throws {
        try runAsync {
            let precision = HTDemucsSeparator.Precision(rawValue: htdemucsPrecision.lowercased()) ?? .fp16
            let separator: HTDemucsSeparator
            if let dir = htdemucsDir {
                print("Loading HTDemucs (Demucs v4, \(precision.rawValue)) from \(dir)...")
                separator = try HTDemucsSeparator.fromLocal(
                    directory: URL(fileURLWithPath: dir), modelName: precision.modelName)
            } else {
                print("Loading HTDemucs (Demucs v4, \(precision.rawValue))...")
                separator = try await HTDemucsSeparator.fromPretrained(
                    precision: precision, progressHandler: reportProgress)
            }

            print("Loading audio: \(input)")
            let audio = try AudioFileLoader.loadStereo(url: inputURL, targetSampleRate: 44100, quality: .mastering)
            let L = audio[0].count
            let duration = Double(L) / 44100.0
            print("  Duration: \(String(format: "%.1f", duration))s")
            let mix = MLXArray(audio[0] + audio[1], [1, 2, L])

            let wanted = Set(try inferenceTargets().map(\.rawValue))
            print("Separating into \(targets.map(\.rawValue).joined(separator: ", "))...")
            let start = CFAbsoluteTimeGetCurrent()
            let stems = separator.separate(mix)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            var results: [SeparationTarget: [[Float]]] = [:]
            for (name, arr) in stems where wanted.contains(name) {
                guard let target = SeparationTarget(rawValue: name) else { continue }
                let flat = arr.asArray(Float.self)             // [1, 2, L] → 2L
                let left = Array(flat[0..<L])
                let right = Array(flat[L..<(2 * L)])
                results[target] = [left, right]
            }
            try writeOutputs(results, targets: targets, to: outDir)
            let rtf = elapsed / duration
            print("Done in \(String(format: "%.1f", elapsed))s (RTF: \(String(format: "%.2f", rtf)))")
        }
    }

    private func writeOutputs(
        _ results: [SeparationTarget: [[Float]]],
        targets: [SeparationTarget],
        to directory: URL
    ) throws {
        let backing = instrumental ? try StemMixer.instrumental(from: results) : nil
        for target in targets {
            guard let audio = results[target] else {
                throw ValidationError("The separator did not return the \(target.rawValue) stem.")
            }
            let outputURL = directory.appendingPathComponent("\(target.rawValue).wav")
            try WAVWriter.writeStereo(left: audio[0], right: audio[1], sampleRate: 44100, to: outputURL)
            print("  Saved: \(outputURL.lastPathComponent)")
        }
        if let backing {
            let outputURL = directory.appendingPathComponent("instrumental.wav")
            try WAVWriter.writeStereo(left: backing[0], right: backing[1], sampleRate: 44100, to: outputURL)
            print("  Saved: \(outputURL.lastPathComponent)")
        }
    }
}

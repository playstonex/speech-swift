// Built by profile_fish_drama3.py as a temporary release target.
import AudioCommon
import Darwin
import FishAudioTTS
import Foundation
import MLX
import Qwen3ASR

private func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

private func memory(includeMLX: Bool = true) -> [String: UInt64] {
    let usage = currentProcessMemoryUsage()
    precondition(usage.residentBytes > 0 && usage.physicalFootprintBytes > 0, "Memory accounting failed")
    var resource = rusage()
    guard getrusage(RUSAGE_SELF, &resource) == 0 else { fatalError("getrusage failed") }
    var result: [String: UInt64] = [
        "rss_bytes": usage.residentBytes,
        "peak_rss_bytes": max(usage.peakResidentBytes, UInt64(max(0, resource.ru_maxrss))),
        "physical_bytes": usage.physicalFootprintBytes,
        "peak_physical_bytes": usage.peakPhysicalFootprintBytes,
    ]
    if includeMLX {
        result["mlx_active_bytes"] = UInt64(MLX.Memory.activeMemory)
        result["mlx_cache_bytes"] = UInt64(MLX.Memory.cacheMemory)
        result["mlx_peak_active_bytes"] = UInt64(MLX.Memory.peakMemory)
    }
    return result
}

private func write(_ object: Any, to url: URL) throws {
    try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        .write(to: url, options: .atomic)
}

private enum ProfileError: Error { case invalidInput(String) }

@main
struct FishDrama3Benchmark {
    static func main() async throws {
        try await Task.detached(priority: .userInitiated) { try await run() }.value
    }

    private static func run() async throws {
        #if DEBUG
        throw ProfileError.invalidInput("Build this profiler in release mode")
        #endif
        setlinebuf(stdout)
        let args = CommandLine.arguments
        func option(_ flag: String, _ fallback: String = "") -> String {
            guard let index = args.firstIndex(of: flag), index + 1 < args.count else { return fallback }
            return args[index + 1]
        }
        let output = URL(fileURLWithPath: option("--output"))
        if args.contains("--asr-manifest") {
            try await verify(manifest: URL(fileURLWithPath: option("--asr-manifest")), output: output)
            return
        }
        let mode = option("--mode")
        let streaming = args.contains("--streaming")
        guard ["text-only", "voice-clone"].contains(mode),
              let repeats = Int(option("--warm-repeats", "3")), (1...10).contains(repeats),
              let seed = UInt64(option("--seed", "42")) else {
            throw ProfileError.invalidInput("Invalid benchmark mode, repeat count or seed")
        }
        let text = option("--text")
        let memoryMode = option("--memory-mode", "balanced")
        guard ["balanced", "unrestricted"].contains(memoryMode) else { throw ProfileError.invalidInput("Invalid memory mode") }
        let memoryOptions: FishAudioMemoryOptions = memoryMode == "unrestricted" ? .unrestricted : .balanced
        let baseline = memory(includeMLX: false)
        print("Loading Drama 3 for \(mode)")
        let loadStart = now()
        let model = try await FishAudioDrama3Model.fromBundle(URL(fileURLWithPath: option("--bundle")), memoryOptions: memoryOptions)
        let loadSeconds = now() - loadStart
        let loaded = memory()
        print(String(format: "Loaded in %.3fs", loadSeconds))
        let referenceStart = now()
        let reference: [Float]? = mode == "voice-clone"
            ? try AudioFileLoader.load(url: URL(fileURLWithPath: option("--reference")), targetSampleRate: model.sampleRate)
            : nil
        let referenceReadSeconds = mode == "voice-clone" ? now() - referenceStart : 0
        let chunkFrames = Int(option("--chunk-frames", "3")) ?? 0
        let options = FishAudioDrama3Options(maxFrames: 512, seed: seed, streamingChunkFrames: chunkFrames)
        try options.validate()
        var rows: [[String: Any]] = []
        for index in 0...repeats {
            let start = now()
            var audio: [Float] = []
            var firstAudio: Double?
            var chunks = 0, firstChunkSamples = 0
            if streaming {
                let deliver: (AudioChunk) async throws -> Void = { chunk in
                    if !chunk.samples.isEmpty {
                        if firstAudio == nil { firstAudio = now() - start; firstChunkSamples = chunk.samples.count }
                        chunks += 1
                        audio += chunk.samples
                    }
                }
                if let reference {
                    try await model.generateStreaming(text: text, referenceAudio: reference,
                        referenceSampleRate: model.sampleRate, referenceText: option("--reference-text"),
                        options: options, onChunk: deliver)
                } else { try await model.generateStreaming(text: text, options: options, onChunk: deliver) }
            } else if let reference {
                audio = try await model.generate(text: text, referenceAudio: reference,
                    referenceSampleRate: model.sampleRate, referenceText: option("--reference-text"), options: options)
            } else {
                audio = try await model.generate(text: text, options: options)
            }
            let seconds = now() - start
            guard audio.count > model.sampleRate, audio.allSatisfy(\.isFinite),
                  (audio.map(abs).max() ?? 0) > 0.01 else {
                throw ProfileError.invalidInput("Invalid generated waveform")
            }
            let snapshot = memory()
            let filename = "\(mode)-\(index).wav"
            try WAVWriter.write(samples: audio, sampleRate: model.sampleRate,
                                to: output.deletingLastPathComponent().appendingPathComponent(filename))
            let duration = Double(audio.count) / Double(model.sampleRate)
            var row: [String: Any] = ["index": index, "phase": index == 0 ? "first" : "warm",
                         "text": text, "audio_file": filename, "samples": audio.count,
                         "sample_rate": model.sampleRate, "audio_seconds": duration,
                         "generation_seconds": seconds, "rtf": seconds / duration, "memory": snapshot]
            if streaming {
                guard let firstAudio, chunks > 0 else { throw ProfileError.invalidInput("Stream produced no PCM") }
                row["first_audio_seconds"] = firstAudio
                row["chunk_count"] = chunks
                row["first_chunk_samples"] = firstChunkSamples
                print(String(format: "First PCM: %.1fms, %d chunks", firstAudio * 1000, chunks))
            }
            rows.append(row)
            print(String(format: "%@ %@ %d: %.3fs synthesis, %.3fs audio, RTF %.3f",
                         mode, index == 0 ? "first" : "warm", index, seconds, duration, seconds / duration))
        }
        try write(["schema_version": 1, "build_configuration": "release", "mode": mode + (streaming ? "-stream" : ""),
                   "delivery": streaming ? "streaming" : "buffered", "streaming_chunk_frames": chunkFrames,
                   "memory_mode": memoryMode,
                   "load_seconds": loadSeconds, "reference_read_seconds": referenceReadSeconds,
                   "reference_encoding_in_generation": mode == "voice-clone",
                   "baseline_memory": baseline, "loaded_memory": loaded, "final_memory": memory(),
                   "warm_repeats": repeats, "seed": seed, "max_frames": options.maxFrames,
                   "temperature": options.temperature, "top_k": options.topK, "top_p": options.topP,
                   "rows": rows], to: output)
    }

    private static func verify(manifest: URL, output: URL) async throws {
        guard let inputs = try JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as? [[String: String]],
              !inputs.isEmpty else { throw ProfileError.invalidInput("Invalid ASR manifest") }
        // This mode never loads the TTS model; it runs after both TTS processes exit.
        let model = try await Qwen3ASRModel.fromPretrained(offlineMode: true)
        var results: [[String: Any]] = []
        for input in inputs {
            guard let path = input["path"], let reference = input["text"], let id = input["id"] else {
                throw ProfileError.invalidInput("Incomplete ASR manifest row")
            }
            let audio = try AudioFileLoader.load(url: URL(fileURLWithPath: path), targetSampleRate: 16000)
            let hypothesis = model.transcribe(audio: audio, sampleRate: 16000, language: "english")
            let errors = WER.compute(reference: Normalizer.normalize(reference), hypothesis: Normalizer.normalize(hypothesis))
            results.append(["id": id, "expected": reference, "transcribed": hypothesis,
                            "word_errors": errors.totalErrors, "reference_words": errors.referenceWords,
                            "wer": errors.wer, "passed": errors.totalErrors == 0])
            print("ASR \(id): \(errors.totalErrors) word errors — \(hypothesis)")
        }
        try write(["model_id": Qwen3ASRModel.defaultModelId, "rows": results], to: output)
    }
}

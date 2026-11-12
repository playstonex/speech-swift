import AudioCommon
import Darwin
import Foundation
import PocketTTS
import KokoroTTS

@main struct PocketBenchmark {
    static func main() async {
        do { try await run() }
        catch {
            FileHandle.standardError.write(Data("Pocket benchmark failed: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }
    static func run() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        if args.contains("--help") {
            print("pocket-tts-bench --bundle-dir PATH [--text TEXT] [--runs 5] [--warmup 1] [--seed 42] [--engine pocket|kokoro] [--output-dir PATH]")
            return
        }
        let keys = ["--bundle-dir", "--text", "--runs", "--warmup", "--seed", "--output-dir", "--engine"]
        guard args.count % 2 == 0 else { throw BenchError.arguments }
        var values: [String:String] = [:]
        for index in stride(from: 0, to: args.count, by: 2) {
            guard keys.contains(args[index]), values[args[index]] == nil else { throw BenchError.arguments }
            values[args[index]] = args[index+1]
        }
        guard let bundle = values["--bundle-dir"], let runs = Int(values["--runs"] ?? "5"), runs > 0,
              let warmup = Int(values["--warmup"] ?? "1"), warmup >= 0,
              let seed = UInt64(values["--seed"] ?? "42") else { throw BenchError.arguments }
        let text = values["--text"] ?? "The quick brown fox jumps over the lazy dog."
        let output = values["--output-dir"].map { URL(fileURLWithPath: $0) }
        if let output { try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true) }
        func seconds(_ start: ContinuousClock.Instant) -> Double {
            let d = start.duration(to: .now).components
            return Double(d.seconds) + Double(d.attoseconds)/1e18
        }
        // The baseline uses the package compute-unit resolver; pin its override too.
        setenv("SPEECH_COREML_COMPUTE_UNITS", "cpuOnly", 1)
        let engine = values["--engine"] ?? "pocket"
        guard ["pocket", "kokoro"].contains(engine) else { throw BenchError.arguments }
        let load = ContinuousClock.now
        let baselineRSS = rss()
        let model: PocketTTSModel?
        let kokoro: KokoroTTSModel?
        if engine == "pocket" {
            model = try PocketTTSModel.fromBundle(URL(fileURLWithPath: bundle), computeUnits: .cpuOnly)
            kokoro = nil
        } else {
            model = nil
            // The measured loop runs its own warmups, so load_ms stays load only.
            kokoro = try await KokoroTTSModel.fromPretrained(cacheDir: URL(fileURLWithPath: bundle), offlineMode: true, computeUnits: .cpuOnly, warmUp: false)
        }
        let loadMS = seconds(load)*1000
        for run in 0..<(warmup+runs) {
            let start = ContinuousClock.now
            var first: Double?, samples: [Float] = [], frames = 0, finalCount = 0
            if let model {
                try await model.generateStreaming(text: text, options: .init(seed: seed)) { chunk in
                    if !chunk.samples.isEmpty {
                        if first == nil { first = seconds(start)*1000 }
                        samples.append(contentsOf: chunk.samples); frames += 1
                    }
                    if chunk.isFinal { finalCount += 1 }
                }
            } else if let kokoro {
                samples = try kokoro.synthesize(text: text)
                first = seconds(start)*1000; frames = 1; finalCount = 1
            }
            let totalMS = seconds(start)*1000
            guard !samples.isEmpty, samples.allSatisfy(\.isFinite), finalCount == 1 else { throw BenchError.invalidAudio }
            let duration = Double(samples.count)/24000
            if let output {
                try WAVWriter.write(samples: samples, sampleRate: 24000, to: output.appendingPathComponent("run-\(run).wav"))
            }
            let report: [String:Any] = ["run": run, "warmup": run < warmup, "text": text, "seed": seed,
                "engine": engine, "compute_units": "cpuOnly", "load_ms": loadMS, "first_audio_ms": first!, "total_ms": totalMS,
                "audio_seconds": duration, "rtf": totalMS/1000/duration, "frames": frames,
                "samples": samples.count, "rss_bytes": rss(), "baseline_rss_bytes": baselineRSS,
                "peak_rss_bytes": peakRSS(), "physical_footprint_bytes": footprint()]
            let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
            print(String(decoding: data, as: UTF8.self))
            if let output {
                let file = output.appendingPathComponent("metrics.jsonl")
                if run == 0 { try Data().write(to: file) }
                let handle = try FileHandle(forWritingTo: file)
                defer { try? handle.close() }
                try handle.seekToEnd(); try handle.write(contentsOf: data + Data([10]))
            }
        }
    }
    enum BenchError: Error { case arguments, invalidAudio }
    static func rss() -> UInt64 {
        var info = mach_task_basic_info(); var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info)/4)
        let result = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count) }
        }
        return result == KERN_SUCCESS ? UInt64(info.resident_size) : 0
    }
    static func peakRSS() -> UInt64 {
        var usage = rusage(); getrusage(RUSAGE_SELF, &usage); return UInt64(usage.ru_maxrss)
    }
    static func footprint() -> UInt64 {
        var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info)/4)
        let result = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : 0
    }
}

// Built by build_runtime_profile.py; no production package targets are added.
import Foundation
import Darwin
import AudioCommon
import AudioServer
import Qwen3ASR
#if RUNTIME_ADAPTER
import SpeechRuntime
import Qwen3ASRRuntime
#endif

private func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

private func memory() -> [String: UInt64] {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    precondition(status == KERN_SUCCESS)
    return ["physical_bytes": UInt64(info.phys_footprint),
            "peak_physical_bytes": UInt64(max(0, info.ledger_phys_footprint_peak)),
            "resident_bytes": UInt64(info.resident_size)]
}

private func writeJSON(_ value: [String: Any], to path: String) throws {
    let data = try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: path), options: .atomic)
}

private func distribution(_ values: [Double]) -> [String: Double] {
    let sorted = values.sorted()
    func percentile(_ p: Double) -> Double { sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * p))] }
    return ["median": percentile(0.5), "p95": percentile(0.95), "p99": percentile(0.99),
            "maximum": sorted.last!, "mean": values.reduce(0, +) / Double(values.count)]
}

@main
struct RuntimeOwnershipProfile {
    static func main() async throws {
        // Keep native and managed calls off the implicit main-actor entry point.
        try await Task.detached(priority: .userInitiated) { try await run() }.value
    }

    private static func run() async throws {
        setlinebuf(stdout)
        let arguments = CommandLine.arguments
        func option(_ name: String, default fallback: String = "") -> String {
            guard let index = arguments.firstIndex(of: name), index + 1 < arguments.count else { return fallback }
            return arguments[index + 1]
        }
        if arguments.contains("--serve") {
            try await AudioServer(host: "127.0.0.1", port: Int(option("--port", default: "19440"))!).run()
            return
        }
        if arguments.contains("--micro") {
            try await micro(output: option("--output"))
            return
        }

        let backend = option("--backend", default: "mlx")
        let mode = option("--mode", default: "native")
        let manifest = URL(fileURLWithPath: option("--manifest"))
        let repeats = Int(option("--repeats", default: "2"))!
        let base = manifest.deletingLastPathComponent()
        let lines = try String(contentsOf: manifest, encoding: .utf8).split(separator: "\n")
        let clips = try lines.filter { !$0.hasPrefix("#") }.map { line -> (String, String, [Float]) in
            let fields = line.split(separator: "\t", maxSplits: 1)
            precondition(fields.count == 2)
            let path = String(fields[0])
            let url = path.hasPrefix("/") ? URL(fileURLWithPath: path) : base.appendingPathComponent(path)
            return (url.path, String(fields[1]), try AudioFileLoader.load(url: url, targetSampleRate: 16000))
        }
        precondition(!clips.isEmpty && repeats > 0)
        var snapshots: [[String: Any]] = [["phase": "before_load", "memory": memory()]]
        let loadStart = now()
        let transcribe: ([Float]) async throws -> String
        let unload: () async throws -> Void
        if mode == "managed" {
            #if RUNTIME_ADAPTER
            let descriptor = backend == "coreml" ? Qwen3ASRDescriptors.coreML : Qwen3ASRDescriptors.small
            let runtime = Qwen3ASRRuntime(descriptor: descriptor, offlineMode: true)
            try await runtime.prepare()
            transcribe = { try await runtime.transcribe(audio: $0, sampleRate: 16000) }
            unload = { try await runtime.unload() }
            #else
            fatalError("The baseline has no runtime adapter")
            #endif
        } else if backend == "coreml" {
            let model = try await CoreMLASRModel.fromPretrained(offlineMode: true)
            // Match EngineQwen3CoreML's per-utterance scratch-buffer lifetime.
            transcribe = { audio in
                try autoreleasepool {
                    try model.transcribe(audio: audio, sampleRate: 16000, language: nil, maxTokens: 448)
                }
            }
            unload = { }
        } else {
            let model = try await Qwen3ASRModel.fromPretrained(offlineMode: true)
            transcribe = { model.transcribe(audio: $0, sampleRate: 16000, language: nil) }
            unload = { model.unload() }
        }
        let loadSeconds = now() - loadStart
        snapshots.append(["phase": "after_load", "memory": memory()])
        for _ in 0..<2 { _ = try await transcribe(clips[0].2) }
        snapshots.append(["phase": "after_warmup", "memory": memory()])
        var rows: [[String: Any]] = []
        for pass in 0..<repeats {
            for (index, clip) in clips.enumerated() {
                let start = now()
                let text = try await transcribe(clip.2)
                let seconds = now() - start
                let reference = Normalizer.normalize(clip.1)
                let hypothesis = Normalizer.normalize(text)
                let errors = WER.compute(reference: reference, hypothesis: hypothesis)
                rows.append(["pass": pass, "index": index, "path": clip.0, "reference": clip.1,
                             "text": text, "seconds": seconds, "audio_seconds": Double(clip.2.count) / 16000,
                             "word_errors": errors.totalErrors, "reference_words": errors.referenceWords,
                             "memory": memory()])
                print("PROFILE \(backend) \(mode) pass=\(pass) clip=\(index) seconds=\(seconds)")
            }
            snapshots.append(["phase": "after_pass_\(pass)", "memory": memory()])
        }
        try await unload()
        snapshots.append(["phase": "after_unload", "memory": memory()])
        try writeJSON(["backend": backend, "mode": mode, "load_seconds": loadSeconds,
                       "repeats": repeats, "clips": clips.count, "rows": rows, "snapshots": snapshots],
                      to: option("--output"))
    }

    static func micro(output: String) async throws {
        var result: [String: Any] = [:]
        let chunk = Array(repeating: Float(0.25), count: 960)
        let destination = UnsafeMutablePointer<Float>.allocate(capacity: 512)
        defer { destination.deallocate() }
        let ring = AudioSampleRingBuffer(capacity: 720000)
        var ringReads: [Double] = []
        var ringWrites: [Double] = []
        var checksum: Float = 0
        for _ in 0..<30000 {
            let start = now()
            ring.write(chunk)
            ringWrites.append((now() - start) * 1e6)
            while ring.availableToRead > 0 {
                let start = now()
                let count = ring.read(into: destination, count: 512)
                ringReads.append((now() - start) * 1e6)
                checksum += destination[count - 1]
            }
        }
        result["ring_read_us"] = distribution(ringReads)
        result["ring_write_us"] = distribution(ringWrites)
        #if RUNTIME_ADAPTER
        let playback = PlaybackSampleBuffer(capacity: 720000)
        var playbackReads: [Double] = []
        var playbackWrites: [Double] = []
        for _ in 0..<30000 {
            let start = now()
            playback.write(chunk)
            playbackWrites.append((now() - start) * 1e6)
            while playback.availableToRead > 0 {
                let start = now()
                let count = playback.read(into: destination, count: 512)
                playbackReads.append((now() - start) * 1e6)
                checksum += destination[count - 1]
            }
        }
        result["playback_read_us"] = distribution(playbackReads)
        result["playback_write_us"] = distribution(playbackWrites)
        let overflow = (0..<1440000).map(Float.init)
        playback.write(overflow)
        var samples = 0
        var overflowReads: [Double] = []
        while playback.availableToRead > 0 {
            let start = now()
            let count = playback.read(into: destination, count: 512)
            overflowReads.append((now() - start) * 1e6)
            for index in 0..<count { precondition(destination[index] == Float(samples + index)) }
            samples += count
        }
        precondition(samples == overflow.count)
        playback.reclaimConsumed()
        result["overflow_read_us"] = distribution(overflowReads)
        result["overflow_samples_preserved"] = samples
        let concurrentPlayback = PlaybackSampleBuffer(capacity: 720000)
        let producer = DispatchGroup()
        producer.enter()
        let stressStart = now()
        DispatchQueue.global().async {
            for _ in 0..<30000 { concurrentPlayback.write(chunk) }
            producer.leave()
        }
        var concurrentReads: [Double] = []
        var concurrentSamples = 0
        while concurrentSamples < chunk.count * 30000 {
            precondition(now() - stressStart < 30, "Playback producer/consumer stalled")
            let start = now()
            let count = concurrentPlayback.read(into: destination, count: 512)
            if count > 0 {
                concurrentReads.append((now() - start) * 1e6)
                precondition(destination[0] == 0.25 && destination[count - 1] == 0.25)
                concurrentSamples += count
            }
        }
        await withCheckedContinuation { continuation in
            producer.notify(queue: .global()) { continuation.resume() }
        }
        concurrentPlayback.reclaimConsumed()
        result["concurrent_playback_read_us"] = distribution(concurrentReads)
        result["concurrent_playback_samples"] = concurrentSamples
        result["concurrent_playback_wall_seconds"] = now() - stressStart
        let gate = InferenceGate()
        var admissions: [Double] = []
        for _ in 0..<50000 {
            let start = now()
            let permit = try await gate.acquire()
            permit.release()
            admissions.append((now() - start) * 1e6)
        }
        result["gate_acquire_release_us"] = distribution(admissions)
        let runtime = ModelRuntime<Int> { 1 }
        var operations: [Double] = []
        for _ in 0..<50000 {
            let start = now()
            let value = try await runtime.withModel { $0 }
            operations.append((now() - start) * 1e6)
            precondition(value == 1)
        }
        result["runtime_operation_us"] = distribution(operations)
        let contendedStart = now()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<32 {
                group.addTask {
                    for _ in 0..<2000 {
                        let value = try await runtime.withModel { $0 }
                        precondition(value == 1)
                    }
                }
            }
            try await group.waitForAll()
        }
        result["contended_runtime_operations"] = 64000
        result["contended_runtime_wall_seconds"] = now() - contendedStart
        #endif
        result["checksum"] = checksum
        result["memory"] = memory()
        try writeJSON(result, to: output)
    }
}

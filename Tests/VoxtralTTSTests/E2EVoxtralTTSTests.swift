import AudioCommon
import Foundation
import Darwin
import MLX
import MLXNN
import Qwen3ASR
import XCTest
@testable import VoxtralTTS

final class E2EVoxtralTTSTests: XCTestCase {
    private static var sharedModel: VoxtralTTSModel?

    override class func tearDown() {
        sharedModel = nil
        super.tearDown()
    }

    private func loadModel() async throws -> VoxtralTTSModel {
        if let model = Self.sharedModel { return model }
        let model: VoxtralTTSModel
        if let directory = ProcessInfo.processInfo.environment["VOXTRAL_TTS_BUNDLE_DIR"] {
            model = try VoxtralTTSModel.fromBundle(URL(fileURLWithPath: directory))
        } else {
            model = try await VoxtralTTSModel.fromPretrained()
        }
        Self.sharedModel = model
        return model
    }

    func testTekkenMatchesMistralPromptsAcrossNineLanguages() async throws {
        let directory: URL
        if let path = ProcessInfo.processInfo.environment["VOXTRAL_TTS_BUNDLE_DIR"] {
            directory = URL(fileURLWithPath: path)
        } else {
            _ = try await loadModel()
            directory = try HuggingFaceDownloader.getCacheDirectory(for: VoxtralTTSModel.defaultModelId)
        }
        let tokenizer = try VoxtralTTSTokenizer(tekkenURL: directory.appendingPathComponent("tekken.json"))
        struct Fixture: Decodable {
            struct Case: Decodable {
                let text: String
                let voice: String
                let textTokens: [Int]
                let prompt: [Int]
                enum CodingKeys: String, CodingKey { case text, voice, prompt; case textTokens = "text_tokens" }
            }
            let cases: [Case]
        }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "tekken_expected", withExtension: "json", subdirectory: "Resources"))
        let expected = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        for item in expected.cases {
            XCTAssertEqual(try tokenizer.encode(item.text), item.textTokens, item.text)
            XCTAssertEqual(try tokenizer.speechPrompt(text: item.text, voice: item.voice), item.prompt, item.voice)
        }
    }

    func testEnglishAndFrenchRoundTripsPreserveOpeningWords() async throws {
        let tts = try await loadModel()
        let asr = try await Qwen3ASRModel.fromPretrained()
        let cases = [
            ("english-opening", "Please open the window and let the fresh air in.", "casual_male", "english"),
            ("english-fox", "The quick brown fox jumps over the lazy dog.", "neutral_female", "english"),
            ("french", "Bonjour, nous testons la synthèse vocale.", "fr_female", "french"),
        ]
        for (name, text, voice, language) in cases {
            let start = CFAbsoluteTimeGetCurrent()
            let samples = try await tts.generate(text: text, voice: voice)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            XCTAssertFalse(samples.isEmpty)
            XCTAssertTrue(samples.allSatisfy(\.isFinite))
            XCTAssertGreaterThan(samples.map { abs($0) }.max() ?? 0, 0.01)
            var decoding = Qwen3DecodingOptions()
            decoding.language = language
            let transcript = try asr.transcribeThrowing(audio: samples, sampleRate: tts.sampleRate, options: decoding)
            let words: (String) -> String = { value in
                value.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).joined(separator: " ")
            }
            XCTAssertEqual(words(transcript), words(text), "\(name): \(transcript)")
            print("Voxtral \(name): \(samples.count) samples, \(elapsed)s, transcript=\(transcript)")
            if let output = ProcessInfo.processInfo.environment["VOXTRAL_VALIDATION_OUTPUT"] {
                let directory = URL(fileURLWithPath: output)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try WAVWriter.write(samples: samples, sampleRate: tts.sampleRate,
                                    to: directory.appendingPathComponent("\(name).wav"))
                let record: [String: Any] = ["text": text, "voice": voice, "transcript": transcript,
                    "samples": samples.count, "sample_rate": tts.sampleRate, "synthesis_seconds": elapsed,
                    "audio_seconds": Double(samples.count) / Double(tts.sampleRate),
                    "rtf": elapsed / (Double(samples.count) / Double(tts.sampleRate))]
                try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
                    .write(to: directory.appendingPathComponent("\(name).json"))
            }
        }
    }

    func testRepeatedAndConcurrentCallsHaveIndependentState() async throws {
        let model = try await loadModel()
        let text = "Hello, this is a short speech test."
        let reference = try await model.generate(text: text)
        async let first = model.generate(text: text)
        async let second = model.generate(text: text)
        let (a, b) = try await (first, second)
        XCTAssertEqual(a, reference)
        XCTAssertEqual(b, reference)
    }

    func testCancellationAndFrameLimitDoNotPoisonTheNextRequest() async throws {
        let model = try await loadModel()
        let task = Task { try await model.generate(text: String(repeating: "This is a long cancellation test. ", count: 30)) }
        try await Task.sleep(for: .milliseconds(100))
        let queued = Task { try await model.generate(text: "This queued request should be cancelled.") }
        try await Task.sleep(for: .milliseconds(100))
        queued.cancel()
        do { _ = try await queued.value; XCTFail("Expected queued cancellation") }
        catch is CancellationError {}
        task.cancel()
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch is CancellationError {}
        do {
            _ = try await model.generate(text: "Please finish this sentence.", options: .init(maxFrames: 1))
            XCTFail("Expected an explicit frame-limit error")
        } catch VoxtralTTSError.generationLimit(let frames) {
            XCTAssertEqual(frames, 1)
        }
        let recovered = try await model.generate(text: "The speech test is complete.")
        XCTAssertGreaterThan(recovered.count, model.sampleRate)
        XCTAssertTrue(recovered.allSatisfy(\.isFinite))
    }

    func testRepeatedSynthesisDoesNotRetainKVOrVoiceBuffers() async throws {
        let model = try await loadModel()
        _ = try await model.generate(text: "Hello.")
        let baseline = Memory.activeMemory
        for _ in 0..<8 { _ = try await model.generate(text: "Hello.") }
        XCTAssertLessThanOrEqual(Memory.activeMemory, baseline + 1024 * 1024)
    }

    func testPerformanceProfile() async throws {
        let model = try await loadModel()
        _ = try await model.generate(text: "Hello.")
        let cases = [
            ("english-opening", "Please open the window and let the fresh air in.", "casual_male"),
            ("english-fox", "The quick brown fox jumps over the lazy dog.", "neutral_female"),
            ("french", "Bonjour, nous testons la synthèse vocale.", "fr_female"),
        ]
        var records: [[String: Any]] = []
        for (name, text, voice) in cases {
            for repetition in 0..<3 {
                let start = CFAbsoluteTimeGetCurrent()
                let samples = try await model.generate(text: text, voice: voice)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                let duration = Double(samples.count) / Double(model.sampleRate)
                XCTAssertGreaterThan(duration, 0)
                records.append(["name": name, "repeat": repetition, "voice": voice, "text": text,
                    "samples": samples.count, "audio_seconds": duration, "synthesis_seconds": elapsed, "rtf": elapsed / duration])
            }
        }
        if let output = ProcessInfo.processInfo.environment["VOXTRAL_VALIDATION_OUTPUT"] {
            let directory = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let report: [String: Any] = ["implementation": "MLX Swift", "peak_mlx_bytes": Memory.peakMemory, "records": records]
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: directory.appendingPathComponent("performance.json"))
        }
    }

    func testStreamingCodecPreservesChunkBoundariesAndFinalSamples() async throws {
        struct Fixture: Decodable { let codes: [[Int32]] }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "decoder_expected", withExtension: "json", subdirectory: "Resources"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        let model = try await loadModel()
        let codes = MLXArray(fixture.codes.flatMap { $0 }).reshaped(1, fixture.codes.count, 37)
        let full = try model.network.decoder.decode(codes).asType(.float32).reshaped(-1).asArray(Float.self)
        for size in [1, 3, 7, 16, fixture.codes.count] {
            var state = VoxtralAudioDecoder.State(model.config.codec)
            var samples: [Float] = []
            for start in stride(from: 0, to: fixture.codes.count, by: size) {
                let end = min(start + size, fixture.codes.count)
                let audio = try model.network.decoder.decode(codes[0..., start..<end, 0...], state: &state)
                samples += audio.asType(.float32).reshaped(-1).asArray(Float.self)
                for (stage, layers) in state.attention.enumerated() {
                    for cache in layers { XCTAssertLessThanOrEqual(try XCTUnwrap(cache).0.dim(2), 2 << stage) }
                }
            }
            // Single-frame matrices can select different approximate M5 kernels.
            // The separate full-FP32 run uses the strict 1e-4 bound for every size.
            assertWaveformsMatch(samples, full, context: "codec chunk size \(size)", maximumAbsoluteError: size == 1 ? 0.025 : 0.01)
        }
    }

    func testStreamingRoundTripsPreserveWordsAndSampleCounts() async throws {
        let model = try await loadModel()
        let asr = try await Qwen3ASRModel.fromPretrained()
        for (name, text, voice, language) in [
            ("english", "Please open the window and let the fresh air in.", "casual_male", "english"),
            ("french", "Bonjour, nous testons la synthèse vocale.", "fr_female", "french"),
        ] {
            let buffered = try await model.generate(text: text, voice: voice)
            var samples: [Float] = []
            var frames = 0
            var finalCount = 0
            try await model.generateStreaming(text: text, voice: voice) { chunk in
                XCTAssertEqual(chunk.frameIndex, frames)
                XCTAssertEqual(chunk.sampleRate, 24000)
                frames += chunk.samples.count / 1920
                samples += chunk.samples
                if chunk.isFinal { finalCount += 1 }
            }
            XCTAssertEqual(finalCount, 1)
            assertWaveformsMatch(samples, buffered, context: name)
            var options = Qwen3DecodingOptions()
            options.language = language
            let transcript = try asr.transcribeThrowing(audio: samples, sampleRate: model.sampleRate, options: options)
            let normalize: (String) -> [String] = {
                $0.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init)
            }
            XCTAssertEqual(normalize(transcript), normalize(text))
            print("Voxtral streaming \(name) round trip: \(transcript)")
            if let output = ProcessInfo.processInfo.environment["VOXTRAL_VALIDATION_OUTPUT"] {
                let directory = URL(fileURLWithPath: output)
                try WAVWriter.write(samples: samples, sampleRate: 24000, to: directory.appendingPathComponent("stream-\(name).wav"))
            }
        }
    }

    func testStreamingDeliveryCancellationAndErrorsReleaseAdmission() async throws {
        let model = try await loadModel()
        let firstChunk = expectation(description: "first chunk delivered before utterance completion")
        let task = Task {
            try await model.generateStreaming(text: String(repeating: "This is a long cancellation test. ", count: 20)) { chunk in
                XCTAssertFalse(chunk.isFinal)
                firstChunk.fulfill()
                try await Task.sleep(for: .seconds(30))
            }
        }
        await fulfillment(of: [firstChunk], timeout: 10)
        let queued = Task { try await model.generate(text: "The next request is complete.") }
        task.cancel()
        do { try await task.value; XCTFail("Expected cancellation") } catch is CancellationError {}
        let recovered = try await queued.value
        XCTAssertFalse(recovered.isEmpty)

        let finalChunk = expectation(description: "final chunk is awaiting delivery")
        let resumeDelivery = AsyncStream<Void>.makeStream()
        let finalDelivery = Task {
            try await model.generateStreaming(text: "Hello.", options: .init(streamingChunkFrames: 125)) { chunk in
                XCTAssertTrue(chunk.isFinal)
                finalChunk.fulfill()
                for await _ in resumeDelivery.stream { break }
            }
        }
        await fulfillment(of: [finalChunk], timeout: 10)
        let waitingForDelivery = Task { try await model.generate(text: "Hello.") }
        for _ in 0..<100 {
            if model.inference.pendingCount == 1 { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(model.inference.pendingCount, 1, "The next request must wait through final chunk delivery")
        resumeDelivery.continuation.yield(())
        resumeDelivery.continuation.finish()
        try await finalDelivery.value
        _ = try await waitingForDelivery.value

        enum DeliveryFailure: Error { case disconnected }
        do {
            try await model.generateStreaming(text: "This request must stop after its first audio chunk.") { _ in
                throw DeliveryFailure.disconnected
            }
            XCTFail("Expected delivery failure")
        } catch DeliveryFailure.disconnected {}
        let recovery = try await model.generate(text: "Hello.")
        XCTAssertFalse(recovery.isEmpty)
    }

    func testStreamingFrameLimitDoesNotEmitSuccessfulFinalChunk() async throws {
        let model = try await loadModel()
        var chunks = 0
        do {
            try await model.generateStreaming(text: "Please finish this complete sentence.",
                options: .init(maxFrames: 3, streamingChunkFrames: 1)) { chunk in
                XCTAssertFalse(chunk.isFinal)
                chunks += 1
            }
            XCTFail("Expected frame-limit failure")
        } catch VoxtralTTSError.generationLimit(let frames) { XCTAssertEqual(frames, 3) }
        XCTAssertEqual(chunks, 3)
        let recovery = try await model.generate(text: "Hello.")
        XCTAssertFalse(recovery.isEmpty)
    }

    func testStreamingProtocolAndMemoryRecovery() async throws {
        let model = try await loadModel()
        let speech: any SpeechGenerationModel = model
        var chunks = 0
        var finals = 0
        for try await chunk in speech.generateStream(text: "Hello, this is a short speech test.", language: "english") {
            if !chunk.samples.isEmpty { chunks += 1 }
            if chunk.isFinal { finals += 1 }
        }
        XCTAssertGreaterThan(chunks, 1)
        XCTAssertEqual(finals, 1)
        let firstAudio = expectation(description: "protocol stream yields audio before completion")
        let consumer = Task {
            var first = true
            for try await chunk in model.generateStream(text: String(repeating: "This is a long streaming test. ", count: 20)) {
                if first && !chunk.samples.isEmpty { first = false; firstAudio.fulfill() }
                try Task.checkCancellation()
            }
        }
        await fulfillment(of: [firstAudio], timeout: 10)
        let cancelledAt = ProcessInfo.processInfo.systemUptime
        consumer.cancel()
        do { try await consumer.value } catch is CancellationError {}
        try await model.generateStreaming(text: "Hello.") { _ in }
        print("Voxtral async stream cancellation and recovery: \(ProcessInfo.processInfo.systemUptime - cancelledAt)s")
        let baseline = Memory.activeMemory
        for _ in 0..<8 { try await model.generateStreaming(text: "Hello.") { _ in } }
        XCTAssertLessThanOrEqual(Memory.activeMemory, baseline + 1024 * 1024)
    }

    func testStreamingPerformanceProfile() async throws {
        let beforeLoad = ProcessInfo.processInfo.systemUptime
        let model = try await loadModel()
        let loadSeconds = ProcessInfo.processInfo.systemUptime - beforeLoad
        var records: [[String: Any]] = []
        let cases = [
            ("english-opening", "Please open the window and let the fresh air in.", "casual_male"),
            ("english-fox", "The quick brown fox jumps over the lazy dog.", "neutral_female"),
            ("french", "Bonjour, nous testons la synthèse vocale.", "fr_female"),
        ]
        for repetition in -1..<3 {
            for (name, text, voice) in (repetition == -1 ? [cases[0]] : cases) {
                var samples = 0
                var chunks = 0
                var firstAudio: Double?
                var lastAudio: Double?
                var largestGap = 0.0
                let start = ProcessInfo.processInfo.systemUptime
                try await model.generateStreaming(text: text, voice: voice) { chunk in
                    if !chunk.samples.isEmpty {
                        let arrival = ProcessInfo.processInfo.systemUptime - start
                        if firstAudio == nil { firstAudio = arrival }
                        if let lastAudio { largestGap = max(largestGap, arrival - lastAudio) }
                        lastAudio = arrival
                        samples += chunk.samples.count
                        chunks += 1
                    }
                }
                let elapsed = ProcessInfo.processInfo.systemUptime - start
                let duration = Double(samples) / 24000
                records.append(["name": name, "repeat": repetition, "cold": repetition == -1,
                    "samples": samples, "chunks": chunks, "audio_seconds": duration,
                    "first_audio_seconds": try XCTUnwrap(firstAudio), "synthesis_seconds": elapsed,
                    "largest_chunk_gap_seconds": largestGap, "rtf": elapsed / duration])
            }
        }
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        if let output = ProcessInfo.processInfo.environment["VOXTRAL_VALIDATION_OUTPUT"] {
            let report: [String: Any] = ["records": records, "load_seconds": loadSeconds,
                "peak_mlx_bytes": Memory.peakMemory, "peak_rss_bytes": usage.ru_maxrss, "chunk_frames": 3]
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: URL(fileURLWithPath: output).appendingPathComponent("streaming-performance.json"))
        }
    }

    private func assertWaveformsMatch(_ actual: [Float], _ expected: [Float], context: String, maximumAbsoluteError: Float = 0.01,
                                      file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, context, file: file, line: line)
        guard actual.count == expected.count else { return }
        let errors = zip(actual, expected).map { $0 - $1 }
        let rms = sqrt(errors.reduce(0.0) { $0 + Double($1 * $1) } / Double(errors.count))
        let signal = sqrt(expected.reduce(0.0) { $0 + Double($1 * $1) } / Double(expected.count))
        let strict = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0"
        XCTAssertLessThan(errors.map { abs($0) }.max() ?? 0, strict ? 1e-4 : maximumAbsoluteError, context, file: file, line: line)
        XCTAssertLessThan(rms / max(signal, 1e-8), strict ? 1e-4 : 0.01, context, file: file, line: line)
        print("Voxtral streaming parity \(context): max=\(errors.map { abs($0) }.max() ?? 0), relative_rms=\(rms / max(signal, 1e-8))")
    }

    func testDecoderMatchesReferenceAcrossAttentionTilesAndFinalSamples() async throws {
        struct Fixture: Decodable {
            let codes: [[Int32]]
            let samples: Int
            let indices: [Int]
            let values: [Float]
        }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "decoder_expected", withExtension: "json", subdirectory: "Resources"))
        let expected = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        let model = try await loadModel()
        let codes = MLXArray(expected.codes.flatMap { $0 }).reshaped(1, expected.codes.count, 37)
        let audio = try model.network.decoder.decode(codes).reshaped(-1).asType(.float32)
        eval(audio)
        let actual = audio.asArray(Float.self)
        XCTAssertEqual(actual.count, expected.samples)
        let errors = zip(expected.indices, expected.values).map { abs(actual[$0.0] - $0.1) }
        let rmsError = sqrt(errors.reduce(0) { $0 + $1 * $1 } / Float(errors.count))
        print("Voxtral decoder: max absolute error=\(errors.max() ?? 0), RMS error=\(rmsError)")
        let referenceRMS = sqrt(expected.values.reduce(0) { $0 + $1 * $1 } / Float(expected.values.count))
        let relativeRMS = rmsError / max(referenceRMS, 1e-8)
        let strict = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0"
        // The fixture uses full FP32 matmuls. M5's default TF32 kernels have
        // lower precision; verify a bounded error there and tight parity when
        // both reference and runtime use MLX_ENABLE_TF32=0 at process launch.
        if let output = ProcessInfo.processInfo.environment["VOXTRAL_VALIDATION_OUTPUT"] {
            let report: [String: Any] = ["samples": actual.count, "max_absolute_error": errors.max() ?? 0,
                "rms_error": rmsError, "relative_rms_error": relativeRMS, "precision": strict ? "full_fp32" : "mlx_default"]
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: URL(fileURLWithPath: output).appendingPathComponent(strict ? "decoder-strict.json" : "decoder-default.json"))
        }
        XCTAssertLessThanOrEqual(relativeRMS, strict ? 1e-4 : 1e-2)
        XCTAssertLessThanOrEqual(errors.max() ?? 0, strict ? 1e-4 : 1e-2)
    }
}

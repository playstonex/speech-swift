import AudioCommon
import Foundation
import MLX
import MLXNN
import Qwen3ASR
import XCTest
@testable import FishAudioTTS

final class E2EFishAudioDrama3MathTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        // Compare FP32 reference math independently of M5's reduced-precision matmul default.
        setenv("MLX_ENABLE_TF32", "0", 1)
    }

    func testUpstreamIntermediateLayers() throws {
        let (fixture, config, network) = try fixtureNetwork()
        let rows = try XCTUnwrap(fixture["frames"] as? [[Int32]])
        let frames = MLXArray(rows.flatMap { $0 }).reshaped(1, rows.count, rows[0].count)
        let trace = try XCTUnwrap(fixture["trace"] as? [String: Any])
        let embeddings = network.tower.embedBackbone(frames, text: network.backbone.model.embeddings(frames[.ellipsis, 0]))
        assertClose(embeddings, try floats(trace["embedding"]), accuracy: 1e-6)
        let first = network.backbone.model.layers[0]
        let normalized = first.inputNorm(embeddings)
        assertClose(normalized, try floats(trace["model.layers.0.input_layernorm"]), accuracy: 1e-6)
        let delta = first.delta!
        let projected = delta.qkv(normalized)
        assertClose(projected, try floats(trace["model.layers.0.linear_attn.in_proj_qkv"]), accuracy: 1e-6)
        assertClose(delta.z(normalized), try floats(trace["model.layers.0.linear_attn.in_proj_z"]), accuracy: 1e-6)
        let padded = concatenated([MLXArray.zeros([1, config.text.convKernel - 1, projected.dim(-1)]), projected], axis: 1)
        assertClose(silu(MLX.conv1d(padded, delta.conv1d.weight, groups: projected.dim(-1))), try floats(trace["convolution"]), accuracy: 1e-6)
        let expectedNormalized = MLXArray(try floats(trace["model.layers.0.linear_attn.norm"])).reshaped(1, rows.count, -1)
        assertClose(delta.output(expectedNormalized), try floats(trace["model.layers.0.linear_attn"]), accuracy: 1e-6)
        assertClose(first.delta!(normalized, cache: Drama3Cache()), try floats(trace["model.layers.0.linear_attn"]), accuracy: 1e-6)
        var hidden = embeddings
        for i in 0..<config.text.layerCount {
            hidden = network.backbone.model.layers[i](hidden, cache: Drama3Cache())
            assertClose(hidden, try floats(trace["model.layers.\(i)"]), accuracy: 1e-6)
        }
    }

    func testUpstreamBackboneAndTemporalTowerParity() throws {
        let (fixture, config, network) = try fixtureNetwork()
        let rows = try XCTUnwrap(fixture["frames"] as? [[Int32]])
        let frames = MLXArray(rows.flatMap { $0 }).reshaped(1, rows.count, rows[0].count)
        let backbone = (0..<config.text.layerCount).map { _ in Drama3Cache() }
        let tower = (0..<(config.tower.sharedLayerCount + config.tower.codebookCount - 1)).map { _ in Drama3Cache() }
        let hidden = network.hidden(frames, caches: backbone)
        assertClose(hidden, try floats(fixture["hidden"]))
        let last = hidden[0..., (rows.count - 1)..., 0...]
        assertClose(network.semanticHead(last), try floats(fixture["semantic_logits"]))
        network.tower.prefill(hidden[0..., 0..<(rows.count - 1), 0...], frames: frames[0..., 0..<(rows.count - 1), 0...],
                              next: frames[0..., 1..., 0...], caches: tower)
        XCTAssertTrue(tower.allSatisfy { $0.keys?.dim(2) == rows.count - 1 })
        let logits = try XCTUnwrap(fixture["code_logits"] as? [[NSNumber]]).map { $0.map(\.floatValue) }
        var codebook = 0
        let codes = network.tower.decode(last, code: 2, frame: frames[0..., (rows.count - 1)..., 0...], caches: tower) { actual in
            assertClose(actual, logits[codebook]); codebook += 1
            return actual.argMax(axis: -1).item(Int.self)
        }
        XCTAssertEqual(codes, fixture["codes"] as? [Int])
        XCTAssertTrue(tower.allSatisfy { $0.keys?.dim(2) == rows.count })
        let feedback = MLXArray(([config.semanticStartTokenId + codes[0]] + codes).map(Int32.init)).reshaped(1, 1, rows[0].count)
        assertClose(network.hidden(feedback, caches: backbone), try floats(fixture["next_hidden"]))
        let steppedCaches = (0..<config.text.layerCount).map { _ in Drama3Cache() }
        let stepped = (0..<rows.count).map { network.hidden(frames[0..., $0..<($0 + 1), 0...], caches: steppedCaches) }
        assertClose(concatenated(stepped, axis: 1), try floats(fixture["hidden"]))
    }

    func testGenerationLimitDoesNotReturnPartialCodebooks() throws {
        let (fixture, config, network) = try fixtureNetwork()
        try network.semanticHead.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.zeros([config.tower.vocabularySize + 1, config.text.hiddenSize]),
        ]), verify: .all)
        let frames = try XCTUnwrap(fixture["frames"] as? [[Int32]])
        let input = try FishAudioModelInput(rows: frames[0].indices.map { column in frames.map { $0[column] } })
        XCTAssertThrowsError(try network.generate(input, options: .init(maxFrames: 1, temperature: 0, topK: 1))) {
            XCTAssertEqual($0 as? FishAudioError, .generationLimit(1))
        }
    }

    private func fixtureNetwork() throws -> ([String: Any], FishAudioDrama3Config, Drama3Network) {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Resources/drama3_reference.json")
        let fixture = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let config = try FishAudioDrama3Config.parse(XCTUnwrap(fixture["config"] as? [String: Any]))
        let network = Drama3Network(config)
        let raw = try XCTUnwrap(fixture["weights"] as? [String: [String: Any]])
        let weights = try raw.mapValues { record -> MLXArray in
            MLXArray(try floats(record["values"])).reshaped(try XCTUnwrap(record["shape"] as? [Int]))
        }
        try network.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        return (fixture, config, network)
    }

    private func floats(_ value: Any?) throws -> [Float] {
        try XCTUnwrap(value as? [NSNumber]).map(\.floatValue)
    }

    private func assertClose(_ actual: MLXArray, _ expected: [Float], accuracy: Float = 2e-4, file: StaticString = #filePath, line: UInt = #line) {
        let values = actual.asType(.float32).asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, file: file, line: line)
        for (a, b) in zip(values, expected) { XCTAssertEqual(a, b, accuracy: accuracy, file: file, line: line) }
    }
}

final class E2EFishAudioDrama3TokenizerTests: XCTestCase {
    func testReleasedTokenizerMatchesUpstreamUnicodeAndSpecialTokens() async throws {
        guard let path = ProcessInfo.processInfo.environment["FISH_DRAMA3_BUNDLE"] else {
            throw XCTSkip("Set FISH_DRAMA3_BUNDLE to a local bundle with tokenizer metadata")
        }
        let tokenizer = try await FishAudioTokenizer.load(from: URL(fileURLWithPath: path))
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Resources/drama3_tokenizer_reference.json")
        let cases = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]])
        for record in cases {
            let text = try XCTUnwrap(record["text"] as? String)
            XCTAssertEqual(tokenizer.encode(text), record["ids"] as? [Int], text)
        }
    }
}

final class E2EFishAudioDrama3Tests: XCTestCase {
    private let target = "The quick brown fox jumps over the lazy dog."

    private func localBundle() throws -> URL {
        let env = ProcessInfo.processInfo.environment
        guard env["FISH_DRAMA3_E2E"] == "1", let path = env["FISH_DRAMA3_BUNDLE"] else {
            throw XCTSkip("Set FISH_DRAMA3_E2E=1 and FISH_DRAMA3_BUNDLE to run real-model tests")
        }
        return URL(fileURLWithPath: path)
    }

    // This scope releases the TTS model before the independent ASR model is loaded.
    private func render(reference: (audio: [Float], text: String)? = nil) async throws -> [Float] {
        let model = try await FishAudioDrama3Model.fromBundle(localBundle())
        let options = FishAudioDrama3Options(maxFrames: 512, seed: 42)
        if let reference {
            return try await model.generate(text: target, referenceAudio: reference.audio, referenceSampleRate: 44100,
                                             referenceText: reference.text, options: options)
        }
        return try await model.generate(text: target, options: options)
    }

    func testTextOnlyASRRoundTrip() async throws {
        _ = try localBundle()
        let audio = try await render()
        Memory.clearCache()
        try await assertContent(audio, name: "text-only")
    }

    func testReferenceCloningASRRoundTrip() async throws {
        _ = try localBundle()
        let env = ProcessInfo.processInfo.environment
        guard let file = env["FISH_DRAMA3_REFERENCE_AUDIO"], let transcript = env["FISH_DRAMA3_REFERENCE_TEXT"] else {
            throw XCTSkip("Set FISH_DRAMA3_REFERENCE_AUDIO and FISH_DRAMA3_REFERENCE_TEXT for cloning")
        }
        let reference = try AudioFileLoader.load(url: URL(fileURLWithPath: file), targetSampleRate: 44100)
        let audio = try await render(reference: (reference, transcript))
        Memory.clearCache()
        try await assertContent(audio, name: "voice-clone")
    }

    func testCLIProducesWAVWithCorrectContent() async throws {
        try await checkCLI(streaming: false)
    }

    func testStreamingCLIProducesWAVWithCorrectContent() async throws {
        try await checkCLI(streaming: true)
    }

    private func checkCLI(streaming: Bool) async throws {
        let bundle = try localBundle()
        guard let binary = ProcessInfo.processInfo.environment["FISH_DRAMA3_CLI"] else {
            throw XCTSkip("Set FISH_DRAMA3_CLI to the built speech executable")
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("output.wav")
        let log = directory.appendingPathComponent("speech.log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let handle = try FileHandle(forWritingTo: log)
        defer { try? handle.close() }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: binary)
        process.arguments = ["speak", target, "--engine", "fish-drama3", "--fish-drama3-bundle-dir", bundle.path,
                             "--fish-drama3-max-frames", "512", "--seed", "42", "-o", output.path]
        if streaming { process.arguments! += ["--stream", "--fish-drama3-chunk-frames", "3"] }
        process.standardOutput = handle; process.standardError = handle
        try process.run(); process.waitUntilExit()
        let logText = String(data: try Data(contentsOf: log), encoding: .utf8) ?? ""
        XCTAssertEqual(process.terminationStatus, 0, logText)
        let audio = try AudioFileLoader.load(url: output, targetSampleRate: 44100)
        if streaming { XCTAssertTrue(logText.contains("First audio:"), logText) }
        try await assertContent(audio, name: streaming ? "cli-stream" : "cli")
    }

    private func renderStreaming(reference: (audio: [Float], text: String)? = nil) async throws -> [Float] {
        let model = try await FishAudioDrama3Model.fromBundle(localBundle())
        let options = FishAudioDrama3Options(maxFrames: 512, seed: 42)
        let expected: [Float]
        if let reference {
            expected = try await model.generate(text: target, referenceAudio: reference.audio,
                referenceSampleRate: 44100, referenceText: reference.text, options: options)
        } else { expected = try await model.generate(text: target, options: options) }
        var audio: [Float] = []
        var finalCount = 0, chunks = 0
        let deliver: (AudioChunk) async throws -> Void = { chunk in
            XCTAssertEqual(chunk.frameIndex * 2048, audio.count)
            XCTAssertEqual(chunk.sampleRate, 44100)
            XCTAssertLessThanOrEqual(chunk.samples.count, 3 * 2048)
            XCTAssertEqual(finalCount, 0)
            if chunk.isFinal { finalCount += 1 }
            else { XCTAssertFalse(chunk.samples.isEmpty) }
            if !chunk.samples.isEmpty { chunks += 1 }
            audio += chunk.samples
        }
        if let reference {
            try await model.generateStreaming(text: target, referenceAudio: reference.audio,
                referenceSampleRate: 44100, referenceText: reference.text, options: options, onChunk: deliver)
        } else { try await model.generateStreaming(text: target, options: options, onChunk: deliver) }
        XCTAssertGreaterThan(chunks, 1)
        XCTAssertEqual(finalCount, 1)
        XCTAssertEqual(audio.count, expected.count)
        let squaredError = zip(audio, expected).reduce(Float(0)) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }
        let energy = expected.reduce(Float(0)) { $0 + $1 * $1 }
        let relativeRMS = sqrt(squaredError / max(energy, 1e-12))
        let maximum = zip(audio, expected).map { abs($0 - $1) }.max() ?? .infinity
        print("Drama 3 streamed/full: max error \(maximum), relative RMS \(relativeRMS)")
        XCTAssertLessThan(maximum, 0.01)
        XCTAssertLessThan(relativeRMS, 0.01)
        return audio
    }

    func testStreamingTextOnlyASRRoundTripAndBufferedParity() async throws {
        _ = try localBundle()
        let audio = try await renderStreaming()
        Memory.clearCache()
        try await assertContent(audio, name: "text-only-stream")
    }

    func testStreamingReferenceCloningASRRoundTripAndBufferedParity() async throws {
        _ = try localBundle()
        let env = ProcessInfo.processInfo.environment
        guard let file = env["FISH_DRAMA3_REFERENCE_AUDIO"], let text = env["FISH_DRAMA3_REFERENCE_TEXT"] else {
            throw XCTSkip("Set the reference audio and transcript for cloning")
        }
        let reference = try AudioFileLoader.load(url: URL(fileURLWithPath: file), targetSampleRate: 44100)
        let audio = try await renderStreaming(reference: (reference, text))
        Memory.clearCache()
        try await assertContent(audio, name: "voice-clone-stream")
    }

    private func assertContent(_ audio: [Float], name: String) async throws {
        XCTAssertGreaterThan(audio.count, 44100)
        XCTAssertTrue(audio.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(audio.map(abs).max() ?? 0, 0.01)
        let artifacts = ProcessInfo.processInfo.environment["FISH_DRAMA3_ARTIFACT_DIR"].map { URL(fileURLWithPath: $0) }
        if let artifacts {
            try FileManager.default.createDirectory(at: artifacts, withIntermediateDirectories: true)
            try WAVWriter.write(samples: audio, sampleRate: 44100, to: artifacts.appendingPathComponent(name + ".wav"))
        }
        let asr = try await Qwen3ASRModel.fromPretrained()
        let text = asr.transcribe(audio: audio, sampleRate: 44100, language: "english").lowercased()
        if let artifacts {
            let report: [String: Any] = ["expected": target, "transcribed": text, "sample_rate": 44100, "samples": audio.count]
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: artifacts.appendingPathComponent(name + ".json"))
        }
        let actualWords = text.split { !$0.isLetter }.map(String.init)
        let expectedWords = target.lowercased().split { !$0.isLetter }.map(String.init)
        XCTAssertEqual(actualWords, expectedWords, text)
    }
}

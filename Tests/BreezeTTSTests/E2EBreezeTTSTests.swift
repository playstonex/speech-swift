import AudioCommon
@testable import BreezeTTS
import MLX
import Qwen3ASR
import XCTest

final class E2EBreezeTTSTests: XCTestCase {
    private func loadModel() async throws -> BreezeTTSModel {
        if let path = ProcessInfo.processInfo.environment["BREEZE_E2E_BUNDLE"] {
            return try await BreezeTTSModel.fromBundle(URL(fileURLWithPath: path))
        }
        return try await BreezeTTSModel.fromPretrained()
    }

    func testFullModelAgainstOfficialPyTorch() async throws {
        let model = try await loadModel()
        let url = try XCTUnwrap(Bundle.module.url(forResource: "model-fixture", withExtension: "json", subdirectory: "Resources"))
        struct Fixture: Decodable {
            let ids: [Int32]
            let text_features: [[Float]]
            let backbone_hidden: [Float]
            let backbone_logits: [Float]
            let first_codebook: Int
            let depth_logits: [Float]
        }
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        let encoded = model.network.textEncoder(MLXArray(fixture.ids).reshaped([1, -1]))
        eval(encoded)
        let embeddings = model.network.textProjection(encoded)
        let hidden = model.network.backbone(embeddings, cache: model.network.backbone.makeCache())[0..., -1, 0...]
        eval(hidden)
        let logits = model.network.lmHead(hidden)
        let depthCache = model.network.depth.model.layers.map { _ in BreezeKVCache() }
        let depthHidden = model.network.depth.model.prefill(first: fixture.first_codebook, backbone: hidden, cache: depthCache)
        let depthLogits = model.network.depth.logits(depthHidden, codebook: 1)
        func cosine(_ actual: MLXArray, _ values: [Float]) -> Float {
            let a = actual.reshaped([-1]).asType(.float32), b = MLXArray(values)
            return ((a * b).sum() / sqrt((a * a).sum() * (b * b).sum())).item(Float.self)
        }
        let textCosine = cosine(encoded[0, 0..., 0..<8], fixture.text_features.flatMap { $0 })
        let backboneCosine = cosine(hidden, fixture.backbone_hidden)
        let logitsCosine = cosine(logits, fixture.backbone_logits)
        let depthCosine = cosine(depthLogits, fixture.depth_logits)
        print("Breeze reference cosine: text=\(textCosine), backbone=\(backboneCosine), logits=\(logitsCosine), depth=\(depthCosine)")
        let threshold: Float = model.config.quantization?.bits == 4 ? 0.97 : 0.99
        XCTAssertGreaterThan(textCosine, threshold)
        XCTAssertGreaterThan(backboneCosine, threshold)
        XCTAssertGreaterThan(logitsCosine, threshold)
        XCTAssertGreaterThan(depthCosine, threshold)
    }

    func testStreamingAndIndependentASRRoundTrip() async throws {
        try await assertStreamingRoundTrip(text: "The quick brown fox jumps over the lazy dog.",
            instruction: "A clear, calm English narrator.", language: "English")
    }

    func testChineseStreamingAndIndependentASRRoundTrip() async throws {
        try await assertStreamingRoundTrip(text: "你好，欢迎使用语音合成。今天天气很好，我们一起去散步吧。",
            instruction: "一位声音清晰、语气亲切的普通话女声。", language: "Chinese")
    }

    private func assertStreamingRoundTrip(text: String, instruction: String, language: String) async throws {
        var model: BreezeTTSModel? = try await loadModel()
        let fixture = try XCTUnwrap(Bundle.module.url(forResource: "tokenizer-fixtures", withExtension: "json", subdirectory: "Resources"))
        struct Row: Decodable { let text: String; let ids: [Int] }
        for row in try JSONDecoder().decode([Row].self, from: Data(contentsOf: fixture)) {
            XCTAssertEqual(try model!.tokenIDs(row.text), row.ids, row.text)
        }
        var samples: [Float] = []
        var chunks = 0, finalChunks = 0
        try await model!.generateStreaming(text: text, instruction: instruction,
            options: BreezeTTSOptions(maxFrames: 200)) { chunk in
            XCTAssertEqual(chunk.sampleRate, 24_000)
            XCTAssertEqual(chunk.frameIndex, samples.count / 1920)
            XCTAssertTrue(chunk.samples.count.isMultiple(of: 1920))
            XCTAssertTrue(chunk.samples.allSatisfy(\.isFinite))
            if !chunk.samples.isEmpty { chunks += 1 }
            if chunk.isFinal { finalChunks += 1 }
            samples += chunk.samples
        }
        XCTAssertGreaterThan(chunks, 1)
        XCTAssertEqual(finalChunks, 1)
        XCTAssertGreaterThan(samples.count, 24_000)
        XCTAssertLessThan(samples.count, 200 * 1920)
        model = nil
        Memory.clearCache()
        let asr = try await Qwen3ASRModel.fromPretrained()
        let actual = asr.transcribe(audio: samples, sampleRate: 24_000, language: language)
        func words(_ text: String) -> [String] {
            text.lowercased().split(whereSeparator: { !$0.isLetter }).map(String.init)
        }
        if language == "Chinese" {
            XCTAssertEqual(actual.filter { $0.isLetter || $0.isNumber },
                           text.filter { $0.isLetter || $0.isNumber }, actual)
        } else {
            XCTAssertEqual(words(actual), words(text), actual)
        }
    }

    func testFrameLimitThrowsInsteadOfReturningTruncatedAudio() async throws {
        let model = try await loadModel()
        do {
            _ = try await model.generate(text: "The quick brown fox jumps over the lazy dog.",
                instruction: "A clear English voice.", options: BreezeTTSOptions(maxFrames: 1))
            XCTFail("Expected frame limit")
        } catch BreezeTTSError.frameLimit(let limit) { XCTAssertEqual(limit, 1) }
    }

    func testReferenceCloneAndDirection() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["BREEZE_REFERENCE_WAV"], let transcript = env["BREEZE_REFERENCE_TEXT"] else {
            throw XCTSkip("Set BREEZE_REFERENCE_WAV and BREEZE_REFERENCE_TEXT")
        }
        let model = try await loadModel()
        let reference = try AudioFileLoader.load(url: URL(fileURLWithPath: path), targetSampleRate: 24_000)
        for instruction in [nil, "Speak clearly and warmly."] as [String?] {
            let samples = try await model.generate(text: "The weather is beautiful today.", instruction: instruction,
                referenceAudio: reference, referenceText: transcript, options: BreezeTTSOptions(maxFrames: 200))
            XCTAssertGreaterThan(samples.count, 24_000)
            XCTAssertTrue(samples.allSatisfy(\.isFinite))
            XCTAssertGreaterThan(samples.map { abs($0) }.max() ?? 0, 0.01)
        }
    }

    func testOverlappingGenerationIsRejectedAndLockRecovers() async throws {
        let model = try await loadModel()
        var checkedOverlap = false
        do {
            try await model.generateStreaming(text: "The weather is beautiful today.",
                instruction: "A clear English narrator.", options: BreezeTTSOptions(maxFrames: 200)) { chunk in
                guard !chunk.samples.isEmpty else { return }
                checkedOverlap = true
                do {
                    _ = try await model.generate(text: "Hello.", instruction: nil)
                    XCTFail("Expected overlapping generation to fail")
                } catch BreezeTTSError.invalidInput(let message) {
                    XCTAssertTrue(message.contains("already generating"), message)
                }
                throw CancellationError()
            }
            XCTFail("Expected the chunk callback cancellation to propagate")
        } catch is CancellationError {}
        XCTAssertTrue(checkedOverlap)

        do {
            _ = try await model.generate(text: "The weather is beautiful today.", instruction: nil,
                options: BreezeTTSOptions(maxFrames: 1))
            XCTFail("Expected frame limit after the generation lock was released")
        } catch BreezeTTSError.frameLimit(let limit) {
            XCTAssertEqual(limit, 1)
        }
    }
}

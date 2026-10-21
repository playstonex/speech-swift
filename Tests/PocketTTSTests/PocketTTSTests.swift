import Foundation
import XCTest
@testable import PocketTTS

final class PocketTTSTests: XCTestCase {
    private func config(_ replacements: [String: Any] = [:]) throws -> PocketTTSConfig {
        var json: [String: Any] = ["model_type":"pocket_tts", "format_version":1, "sample_rate":24000,
            "samples_per_frame":1920, "language":"en", "voice":"alba", "layers":6,
            "cache_capacity":512, "voice_position":126, "flow_steps":8]
        json.merge(replacements) { _, new in new }
        return try JSONDecoder().decode(PocketTTSConfig.self, from: JSONSerialization.data(withJSONObject: json))
    }
    func testTokenizerMatchesSentencePieceReference() throws {
        struct Fixture: Decodable { let text: String; let ids: [Int] }
        let tokenizer = try PocketTTSTokenizer(data: Data(contentsOf: Bundle.module.url(forResource: "tokenizer", withExtension: "model")!))
        let cases = try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: Bundle.module.url(forResource: "tokenizer-fixtures", withExtension: "json")!))
        for fixture in cases { XCTAssertEqual(try tokenizer.encode(fixture.text), fixture.ids, fixture.text) }
        let long = String(repeating: "Please set a timer for five minutes. ", count: 30)
        let chunks = try tokenizer.chunks(long)
        XCTAssertEqual(chunks.joined(separator: " "), long.trimmingCharacters(in: .whitespaces))
        XCTAssertTrue(try chunks.allSatisfy { try tokenizer.encode($0).count <= 50 })
    }
    func testBundleContract() throws {
        try config().validate()
        for invalid: [String:Any] in [["format_version":2],["sample_rate":16000],["layers":24],
            ["voice":"unknown"],["mimi_static_inputs":["attn0_cache"]],["mimi_static_inputs":["conv0_first","conv0_first"]],["voice_position":512],["flow_steps":4],["language":"fr"]] {
            XCTAssertThrowsError(try config(invalid).validate())
        }
    }
    func testInvalidOptions() throws {
        try PocketTTSOptions(seed: 0).validate()
        for option in [PocketTTSOptions(temperature: .nan), .init(temperature: -.infinity),
                       .init(temperature: -1), .init(maxFrames: 0), .init(maxFrames: 501)] {
            XCTAssertThrowsError(try option.validate())
        }
    }
    func testExplicitVoiceAndLanguageContract() throws {
        try PocketTTSModel.validate(language: "English")
        try PocketTTSModel.validate(language: nil)
        XCTAssertThrowsError(try PocketTTSModel.validate(language: "fr"))
        XCTAssertThrowsError(try PocketTTSModel.validate(language: "en", voice: "../alba"))
    }
    func testTextNormalizationAndLimits() throws {
        XCTAssertEqual(try PocketTTSTokenizer.normalized("  It’s\n\ta test.  "), "It's a test.")
        XCTAssertThrowsError(try PocketTTSTokenizer.normalized(" \n"))
        XCTAssertThrowsError(try PocketTTSTokenizer.normalized(String(repeating: "x", count: 4097)))
    }
    func testNoiseSeedIsRequestLocalAndFinite() {
        var first = PocketRNG(state: 0), second = PocketRNG(state: 0), other = PocketRNG(state: 1)
        let a = (0..<1000).map { _ in first.gaussian() }
        XCTAssertEqual(a, (0..<1000).map { _ in second.gaussian() })
        XCTAssertNotEqual(a, (0..<1000).map { _ in other.gaussian() })
        XCTAssertTrue(a.allSatisfy(\.isFinite))
    }
    func testConstantFileValidation() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: path) }
        try Data([0,0,0]).write(to: path)
        XCTAssertThrowsError(try PocketTTSModel.readFloats(path, count: 1))
        var nonfinite = Float.infinity.bitPattern.littleEndian
        try withUnsafeBytes(of: &nonfinite) { try Data($0).write(to: path) }
        XCTAssertThrowsError(try PocketTTSModel.readFloats(path, count: 1))
    }
}

final class E2EPocketTTSTests: XCTestCase {
    func model() throws -> PocketTTSModel {
        guard let path = ProcessInfo.processInfo.environment["POCKET_TTS_BUNDLE"] else {
            throw XCTSkip("Set POCKET_TTS_BUNDLE to a verified local CoreML bundle")
        }
        return try .fromBundle(URL(fileURLWithPath: path))
    }
    func testStreamingStateIsolationAndCompletion() async throws {
        let model = try model()
        XCTAssertEqual(model.config.mimiStaticInputs?.count, 11)
        XCTAssertTrue(Set(model.mimi.modelDescription.inputDescriptionsByName.keys)
            .isDisjoint(with: Set(model.mimi.modelDescription.outputDescriptionsByName.keys)))
        var first: [Float] = [], index = 0, finals = 0
        try await model.generateStreaming(text: "Hello world.") { chunk in
            XCTAssertEqual(chunk.frameIndex, index)
            if chunk.isFinal { finals += 1; XCTAssertTrue(chunk.samples.isEmpty) }
            else { XCTAssertEqual(chunk.samples.count,1920); index += 1 }
            first.append(contentsOf: chunk.samples)
        }
        XCTAssertEqual(finals, 1); XCTAssertGreaterThan(first.count, 1920)
        XCTAssertTrue(first.allSatisfy(\.isFinite))
        let second = try await model.generate(text: "Hello world.", options: .init())
        XCTAssertEqual(first,second, "Recurrent state must reset for every request")
        XCTAssertGreaterThan(first.map { abs($0) }.max() ?? 0, 0.01)
    }
    func testFailureAndCancellationReleaseAdmission() async throws {
        let model = try model()
        do {
            try await model.generateStreaming(text: "This sentence requires several frames.", options: .init(maxFrames: 1)) { _ in }
            XCTFail("Frame limit must fail")
        } catch PocketTTSError.generationLimit { }
        let task = Task {
            try await model.generateStreaming(text: "Hello world.") { _ in
                throw CancellationError()
            }
        }
        do { try await task.value; XCTFail("Callback failure must propagate") }
        catch is CancellationError { }
        let recovered = try await model.generate(text: "Hello world.", options: .init())
        XCTAssertFalse(recovered.isEmpty)
    }
}

import Foundation
import XCTest
@testable import MossTTSNano

final class MossTTSNanoTests: XCTestCase {
    private func fixture(_ name: String) throws -> Data {
        try Data(contentsOf: XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json")))
    }

    func testConfigurationAndUnsupportedDimensions() throws {
        let data = try fixture("config")
        let config = try JSONDecoder().decode(MossTTSNanoConfiguration.self, from: data)
        try config.validate()
        XCTAssertEqual(config.contextLength, 1024)
        var fields = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for (key, value) in [("channels", 1), ("sample_rate", 24000), ("codebooks", 8), ("prefill_limit", 0)] {
            var invalid = fields; invalid[key] = value
            let result = try JSONDecoder().decode(MossTTSNanoConfiguration.self, from: JSONSerialization.data(withJSONObject: invalid))
            XCTAssertThrowsError(try result.validate(), key)
        }
        fields["model_type"] = "another_model"
        XCTAssertThrowsError(try JSONDecoder().decode(MossTTSNanoConfiguration.self,
            from: JSONSerialization.data(withJSONObject: fields)).validate())
    }

    func testUpstreamTokenizerAndPromptParity() throws {
        struct Fixture: Decodable {
            struct Encoding: Decodable { let text: String; let ids: [Int32] }
            let encodings: [Encoding]; let prompt: [[Int32]]
        }
        let tokenizer = try MossTTSNanoTokenizer(data: fixture("tokenizer_swift"))
        let expected = try JSONDecoder().decode(Fixture.self, from: fixture("tokenizer_fixtures"))
        for example in expected.encodings { XCTAssertEqual(try tokenizer.encode(example.text), example.ids, example.text.debugDescription) }
        XCTAssertEqual(try tokenizer.prompt(text: "Hello from Core ML."), expected.prompt)
        let reference = try MossTTSNanoReference(codes: [[Int32](repeating: 17, count: 16)])
        let rows = try tokenizer.prompt(text: "Hello", reference: reference)
        let index = try XCTUnwrap(rows.firstIndex { $0[0] == 8 })
        XCTAssertEqual(rows[index - 1][0], 6)
        XCTAssertEqual(rows[index], [8] + [Int32](repeating: 17, count: 16))
        XCTAssertEqual(rows[index + 1][0], 7)
        XCTAssertTrue(rows.filter { $0[0] != 8 }.allSatisfy { $0.count == 17 && $0.dropFirst().allSatisfy { $0 == 1024 } })
        XCTAssertThrowsError(try tokenizer.prompt(text: "  \n"))
        XCTAssertThrowsError(try tokenizer.encode(String(repeating: "a", count: 32769)))
    }

    func testRejectsInvalidTokenizerArchive() throws {
        XCTAssertThrowsError(try MossTTSNanoTokenizer(data: Data("{}".utf8)))
        var fields = try XCTUnwrap(JSONSerialization.jsonObject(with: fixture("tokenizer_swift")) as? [String: Any])
        fields["precompiled_charsmap"] = Data([0, 0, 0, 0, 0]).base64EncodedString()
        XCTAssertThrowsError(try MossTTSNanoTokenizer(data: JSONSerialization.data(withJSONObject: fields)))
    }

    func testReferenceAndGenerationValidation() throws {
        XCTAssertThrowsError(try MossTTSNanoReference(codes: []))
        XCTAssertThrowsError(try MossTTSNanoReference(codes: [[1, 2]]))
        XCTAssertThrowsError(try MossTTSNanoReference(codes: [[Int32](repeating: 1024, count: 16)]))
        try MossTTSNanoOptions().validate()
        for options in [MossTTSNanoOptions(maxFrames: 0), MossTTSNanoOptions(audioTemperature: .nan),
                        MossTTSNanoOptions(topK: -1), MossTTSNanoOptions(topP: 0), MossTTSNanoOptions(repetitionPenalty: 0)] {
            XCTAssertThrowsError(try options.validate())
        }
    }

    func testSamplerGreedyPenaltyNucleusAndDeterminism() throws {
        var sampler = MossTTSNanoSampler(state: 0)
        XCTAssertEqual(try sampler.sample([1, 2, 2], greedy: true), 1)
        XCTAssertEqual(try sampler.sample([1.9, 2], seen: [1], penalty: 2, greedy: true), 0)
        XCTAssertEqual(try sampler.sample([-2, -1.9], seen: [1], penalty: 2, greedy: true), 0)
        for _ in 0..<100 {
            XCTAssertEqual(try sampler.sample([0, 10, -1], topK: 1), 1)
            XCTAssertEqual(try sampler.sample([0, 10, -1], topP: 0.01), 1)
        }
        var a = MossTTSNanoSampler(state: 42), b = a
        let first = try (0..<100).map { _ in try a.sample([1, 1, 1]) }
        let second = try (0..<100).map { _ in try b.sample([1, 1, 1]) }
        XCTAssertEqual(first, second)
        XCTAssertEqual(Set(first), [0, 1, 2])
        XCTAssertThrowsError(try sampler.sample([.nan]))
        XCTAssertThrowsError(try sampler.sample([]))
        XCTAssertThrowsError(try sampler.sample([1], seen: [2]))
    }

    func testPCG64MatchesNumPyReference() {
        let fixtures: [(UInt64, [Double])] = [
            (0, [0.6369616873214543, 0.2697867137638703, 0.04097352393619469, 0.016527635528529094]),
            (1, [0.5118216247002567, 0.9504636963259353, 0.14415961271963373, 0.9486494471372439]),
            (42, [0.7739560485559633, 0.4388784397520523, 0.8585979199113825, 0.6973680290593639]),
            (4294967296, [0.8897387912781343, 0.5571380502062263, 0.8009080868919721, 0.9565138174753386]),
            (.max, [0.6800266789616931, 0.8453117585624743, 0.007403081599260064, 0.8945681264391473]),
        ]
        for (seed, expected) in fixtures {
            var sampler = MossTTSNanoSampler(state: seed)
            XCTAssertEqual(expected.map { _ in sampler.uniform() }, expected)
        }
    }

    func testSamplerMatchesUpstreamSequences() throws {
        struct Fixture: Decodable {
            struct Sequence: Decodable { let seed: UInt64; let ids: [Int] }
            let logits: [Float]; let sequences: [Sequence]
        }
        let expected = try JSONDecoder().decode(Fixture.self, from: fixture("sampling_fixtures"))
        for sequence in expected.sequences {
            var sampler = MossTTSNanoSampler(state: sequence.seed), seen = Set<Int>()
            for token in sequence.ids {
                let actual = try sampler.sample(expected.logits, temperature: 0.8, topK: 25, topP: 0.95, seen: seen, penalty: 1.2)
                XCTAssertEqual(actual, token)
                seen.insert(actual)
            }
        }
    }

    func testStereoDownmix() {
        let chunk = MossTTSNanoChunk(left: [1, -1, 0.5], right: [-1, 1, 0.5], frameIndex: 0, isFinal: false, elapsedTime: 0)
        XCTAssertEqual(chunk.mono, [0, 0, 0.5])
    }
}

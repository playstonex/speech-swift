@testable import BreezeTTS
import MLX
import XCTest

final class BreezeSamplingTests: XCTestCase {
    func testGreedyMasksReservedIDsAndMapsBackboneEOS() {
        Device.withDefaultDevice(.cpu) {
            let logits = MLXArray([Float(0), 1, 2, 100, 100, 100, 4]).reshaped([1, 7])
            let options = BreezeTTSOptions(temperature: 0)
            let random = MLXRandom.RandomState(seed: 42)
            XCTAssertEqual(breezeSampleCodebook(logits, codecVocabSize: 3, audioVocabSize: 6,
                allowsEOS: false, options: options, random: random).item(Int.self), 2)
            XCTAssertEqual(breezeSampleCodebook(logits, codecVocabSize: 3, audioVocabSize: 6,
                allowsEOS: true, options: options, random: random).item(Int.self), 6)
        }
    }

    func testTopKAndNucleusKeepTheOnlyEligibleCode() {
        Device.withDefaultDevice(.cpu) {
            let logits = MLXArray([Float(-20), -20, 10, 100, 100, 100, -20]).reshaped([1, 7])
            for options in [BreezeTTSOptions(topK: 1), BreezeTTSOptions(topK: 0, topP: 0.01)] {
                XCTAssertEqual(breezeSampleCodebook(logits, codecVocabSize: 3, audioVocabSize: 6,
                    allowsEOS: false, options: options, random: MLXRandom.RandomState(seed: 42)).item(Int.self), 2)
            }
        }
    }

    func testDeferredEvaluationPreservesSeededTokenSequence() {
        Device.withDefaultDevice(.cpu) {
            let logits = MLXArray([Float(1), 0.5, 2, -1, -2, -3, 0.8]).reshaped([1, 7])
            for topP in [Float(1), 0.8] {
                let options = BreezeTTSOptions(topK: 3, topP: topP)
                let eager = MLXRandom.RandomState(seed: 42), deferred = MLXRandom.RandomState(seed: 42)
                let expected = (0..<16).map { _ in
                    breezeSampleCodebook(logits, codecVocabSize: 3, audioVocabSize: 6,
                        allowsEOS: true, options: options, random: eager).item(Int32.self)
                }
                let tokens = (0..<16).map { _ in
                    breezeSampleCodebook(logits, codecVocabSize: 3, audioVocabSize: 6,
                        allowsEOS: true, options: options, random: deferred)
                }
                XCTAssertEqual(stacked(tokens).asArray(Int32.self), expected)
            }
        }
    }

    func testDeviceTokenStepMatchesHostTokensForGuidanceBranches() throws {
        try Device.withDefaultDevice(.cpu) {
            let data = Data("""
            {"hidden_size":16,"intermediate_size":32,"num_hidden_layers":2,
             "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
             "audio_embed_size":16,"backbone_hidden_size":16,"vocab_size":32}
            """.utf8)
            let config = try JSONDecoder().decode(BreezeDepthDecoderConfig.self, from: data)
            let model = BreezeDepthModel(config: config)
            let backbone = MLXArray((0..<32).map { Float($0) / 32 }).reshaped([2, 16])
            let hostCache = model.layers.map { _ in BreezeKVCache() }
            let deviceCache = model.layers.map { _ in BreezeKVCache() }
            eval(model.prefill(first: 3, backbone: backbone, cache: hostCache),
                 model.prefill(first: 3, backbone: backbone, cache: deviceCache))
            for (codebook, token) in [5, 7, 9].enumerated() {
                let expected = model.step(token: token, codebook: codebook + 1, batchSize: 2, cache: hostCache)
                let deviceToken = MLXArray([Int32(token - 1), 1]).sum()
                let actual = model.step(token: deviceToken, codebook: codebook + 1, batchSize: 2, cache: deviceCache)
                XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 0.0001)
            }
        }
    }
}

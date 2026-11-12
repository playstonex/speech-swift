@testable import BreezeTTS
import MLX
import MLXNN
import XCTest

/// CPU reference checks avoid the reduced TF32 precision used by M5 Metal matmuls.
final class BreezeMathTests: XCTestCase {
    func testCodecUpsamplingPreservesEveryFrame() throws {
        try Device.withDefaultDevice(.cpu) {
            let first = BreezeTTSCodec.makeUpsampler(channels: 1)
            let second = BreezeTTSCodec.makeUpsampler(channels: 1)
            for layer in [first, second] {
                try layer.update(parameters: ModuleParameters.unflattened([
                    ("conv.weight", MLXArray.ones([1, 2, 1])), ("conv.bias", MLXArray.zeros([1]))
                ]), verify: .all)
            }
            for count in [1, 3, 25, 50] {
                let input = MLXArray.ones([1, count, 1])
                let output = second(first(input))
                XCTAssertEqual(output.shape, [1, count * 4, 1])
                XCTAssertEqual(output.reshaped([-1]).asArray(Float.self), Array(repeating: 1, count: count * 4))
            }
        }
    }

    func testTextEncoderMatchesOfficialPyTorchFixture() throws {
        try Device.withDefaultDevice(.cpu) {
            let url = try XCTUnwrap(Bundle.module.url(forResource: "text-encoder-fixture", withExtension: "json", subdirectory: "Resources"))
            let data = try Data(contentsOf: url)
            struct Fixture: Decodable {
                let config: BreezeTextEncoderConfig
                let ids: [Int32]
                let expected: [[Float]]
                let parameter_names: [String]
            }
            let fixture = try JSONDecoder().decode(Fixture.self, from: data)
            let encoder = BreezeTTSTextEncoder(config: fixture.config)
            let parameters = encoder.parameters().flattened().sorted { $0.0 < $1.0 }
            XCTAssertEqual(parameters.map(\.0), fixture.parameter_names)
            let weights = parameters.enumerated().map { ordinal, entry -> (String, MLXArray) in
                let values = (0..<entry.1.size).map { Float(($0 + ordinal * 3) % 17 - 8) / 64 }
                return (entry.0, MLXArray(values).reshaped(entry.1.shape))
            }
            try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            let actual = encoder(MLXArray(fixture.ids).reshaped([1, -1])).reshaped([-1])
            let expected = MLXArray(fixture.expected.flatMap { $0 })
            XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 0.0001)
        }
    }

    func testDepthCacheMatchesFullCausalForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let data = Data("""
            {"hidden_size":16,"intermediate_size":32,"num_hidden_layers":2,
             "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
             "audio_embed_size":16,"backbone_hidden_size":16,"vocab_size":32,
             "rope_scaling":{"rope_type":"llama3","factor":32,"low_freq_factor":0.001953125,
                             "high_freq_factor":0.0078125,"original_max_position_embeddings":16}}
            """.utf8)
            let config = try JSONDecoder().decode(BreezeDepthDecoderConfig.self, from: data)
            let model = BreezeDepthModel(config: config)
            XCTAssertFalse(model.parameters().flattened().contains { $0.0.contains("frequencies") })
            eval(model.parameters())
            let backbone = MLXArray((0..<16).map { Float($0) / 16 }).reshaped([1, 16])
            let ids = MLXArray([Int32(3), 5 + 32, 7 + 64]).reshaped([1, 3])
            let embeddings = concatenated([backbone.expandedDimensions(axis: 1), model.embedTokens(ids)], axis: 1)
            let full = model.forward(embeddings, cache: model.layers.map { _ in BreezeKVCache() })
            eval(full)
            let cache = model.layers.map { _ in BreezeKVCache() }
            let prefill = model.prefill(first: 3, backbone: backbone, cache: cache)
            eval(prefill)
            cache.forEach { $0.materialize() }
            let second = model.step(token: 5, codebook: 1, batchSize: 1, cache: cache)
            eval(second)
            cache.forEach { $0.materialize() }
            let last = model.step(token: 7, codebook: 2, batchSize: 1, cache: cache)
            eval(last)
            XCTAssertLessThan(abs(full[0, -1] - last[0, -1]).max().item(Float.self), 0.0001)
            XCTAssertEqual(cache.first?.offset, 4)
        }
    }

    func testBidirectionalWindowMatchesUpstreamEdges() throws {
        try Device.withDefaultDevice(.cpu) {
            let mask = try XCTUnwrap(breezeTextAttentionMask(length: 7, layerType: "sliding_attention", slidingWindow: 4))
            let rows = mask.reshaped([7, 7]).asArray(Bool.self)
            func row(_ index: Int) -> [Int] { (0..<7).filter { rows[index * 7 + $0] } }
            XCTAssertEqual(row(0), [0, 1, 2])
            XCTAssertEqual(row(3), [2, 3, 4, 5])
            XCTAssertEqual(row(6), [5, 6])
            XCTAssertNil(breezeTextAttentionMask(length: 7, layerType: "full_attention", slidingWindow: 4))
        }
    }
}

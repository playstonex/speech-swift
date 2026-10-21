import AudioCommon
import Foundation
import MLX
import MLXNN
import PersonaPlex
import XCTest
@testable import KyutaiSTT

func configurationData(_ changes: [String: Any] = [:]) throws -> Data {
    var config: [String: Any] = [
        "model_type": "stt", "dim": 2048, "num_heads": 16, "num_layers": 16,
        "text_card": 8000, "card": 2048, "n_q": 32, "dep_q": 0,
        "context": 750, "max_period": 100000.0, "hidden_scale": 4.125,
        "delays": Array(repeating: 0, count: 33), "causal": true,
        "norm": "rms_norm_f32", "gating": "silu", "positional_embedding": "rope",
        "layer_scale": NSNull(), "conditioners": [:],
        "mimi_name": "mimi.safetensors", "tokenizer_name": "tokenizer.model",
        "stt_config": ["audio_delay_seconds": 0.5, "audio_silence_prefix_seconds": 0.0],
    ]
    config.merge(changes) { _, new in new }
    return try JSONSerialization.data(withJSONObject: config)
}

final class KyutaiSTTTests: XCTestCase {
    func testPublishedVariants() {
        XCTAssertEqual(KyutaiSTTVariant.allCases.count, 3)
        XCTAssertEqual(KyutaiSTTModel.defaultModelId, KyutaiSTTVariant.int8.modelId)
        XCTAssertEqual(Set(KyutaiSTTVariant.allCases.map(\.modelId)).count, 3)
        XCTAssertEqual(KyutaiSTTModel.upstreamModelId, "kyutai/stt-1b-en_fr")
    }

    func testCodecWeightMappingOmitsOnlyAudioDecoding() throws {
        let weights = try KyutaiSTTCodec.sanitize([
            "encoder.model.0.conv.conv.weight": MLXArray.ones([64, 1, 7]),
            "encoder_transformer.transformer.layers.0.self_attn.in_proj_weight": MLXArray.ones([12, 4]),
            "quantizer.rvq_first.input_proj.weight": MLXArray.ones([8, 4, 1]),
            "decoder.model.0.conv.conv.weight": MLXArray.ones([64, 1, 7]),
            "decoder_transformer.transformer.layers.0.self_attn.in_proj_weight": MLXArray.ones([12, 4]),
            "upsample.convtr.convtr.weight": MLXArray.ones([4, 1, 4]),
        ])
        XCTAssertEqual(weights.count, 3)
        XCTAssertEqual(weights["encoder.init_conv1d.conv.conv.weight"]?.shape, [64, 7, 1])
        XCTAssertEqual(weights["quantizer.rvq_first.input_proj.weight"]?.shape, [8, 1, 4])
        XCTAssertNotNil(weights["encoder_transformer.transformer.layers.0.self_attn.in_proj.weight"])
        XCTAssertThrowsError(try KyutaiSTTCodec.sanitize(["unknown.weight": MLXArray.zeros([1])]))
    }

    func testMimiPaddingCanMatchKyutaiWithoutChangingDefault() {
        let legacyFactory: (Int) -> MimiConfig = MimiConfig.moshiko
        XCTAssertEqual(legacyFactory(16).numCodebooks, 16)
        if case .edge = MimiConfig.moshiko().seanet.padMode {} else { XCTFail("Default padding changed") }
        if case .constant = MimiConfig.moshiko(numCodebooks: 32, padMode: .constant).seanet.padMode {} else {
            XCTFail("Kyutai requires constant zero padding")
        }
    }

    func testOriginalEnglishFrenchConfiguration() throws {
        let config = try JSONDecoder().decode(KyutaiSTTConfig.self, from: configurationData())
        XCTAssertEqual(config.intermediateSize, 5632)
        XCTAssertEqual(config.prefixFrames, 0)
        XCTAssertEqual(config.flushFrames, 20)
        XCTAssertEqual(config.weightLayout, "pytorch")
        XCTAssertNil(config.quantization)
    }

    func testEnglishTimingRoundsPrefixAndFlushUp() throws {
        let config = try JSONDecoder().decode(KyutaiSTTConfig.self, from: configurationData([
            "num_heads": 32, "num_layers": 48, "text_card": 4000, "context": 375,
            "stt_config": ["audio_delay_seconds": 2.5, "audio_silence_prefix_seconds": 1.0],
        ]))
        XCTAssertEqual(config.prefixFrames, 13)
        XCTAssertEqual(config.flushFrames, 45)
    }

    func testAcceptsBothQuantizedLayouts() throws {
        for bits in [4, 8] {
            let config = try JSONDecoder().decode(KyutaiSTTConfig.self, from: configurationData([
                "format_version": 1, "weight_layout": "speech_swift_v1",
                "quantization": ["bits": bits, "group_size": 64, "mode": "affine"],
            ]))
            XCTAssertEqual(config.quantization?.bits, bits)
        }
    }

    func testRejectsIncompatibleArchitecturesBeforeAllocation() throws {
        let invalid: [[String: Any]] = [
            ["dim": 0], ["num_heads": 3], ["n_q": 16], ["dep_q": 8],
            ["delays": Array(repeating: 1, count: 33)], ["model_type": "moshi"],
            ["hidden_scale": 4], ["norm": "layer_norm"], ["causal": false],
            ["layer_scale": 0.01], ["conditioners": ["speaker": [:]]],
            ["extra_heads_num_heads": 3], ["existing_text_padding_id": 0],
            ["sample_rate": 16000], ["frame_rate": 25],
            ["stt_config": ["audio_delay_seconds": -1, "audio_silence_prefix_seconds": 0]],
            ["tokenizer_name": "../tokenizer.model"], ["mimi_name": ".."],
            ["weight_layout": "unknown"], ["weight_layout": "speech_swift_v1", "format_version": 2],
            ["quantization": ["bits": 7, "group_size": 64, "mode": "affine"]],
            ["quantization": ["bits": 4, "group_size": 0, "mode": "affine"]],
        ]
        for changes in invalid {
            XCTAssertThrowsError(try JSONDecoder().decode(KyutaiSTTConfig.self, from: configurationData(changes)), "\(changes)")
        }
    }

    func testTensorRemappingAndNormFlattening() throws {
        let weights = try KyutaiSTTDecoder.sanitize([
            "transformer.layers.0.self_attn.in_proj_weight": MLXArray.zeros([24, 8]),
            "transformer.layers.0.norm1.alpha": MLXArray.ones([1, 1, 8]),
            "out_norm.alpha": MLXArray.ones([1, 1, 8]),
        ])
        XCTAssertEqual(weights["layers.0.self_attn.in_proj.weight"]?.shape, [24, 8])
        XCTAssertEqual(weights["layers.0.norm1.weight"]?.shape, [8])
        XCTAssertEqual(weights["out_norm.weight"]?.shape, [8])
        XCTAssertThrowsError(try KyutaiSTTDecoder.sanitize([
            "out_norm.alpha": MLXArray.ones([1, 1, 8]), "out_norm.weight": MLXArray.ones([8]),
        ]))
    }

    func testSlidingCacheRetainsRecentFramesAndAbsolutePosition() {
        let cache = KyutaiSTTCache(capacity: 3)
        for i in 0..<8 {
            let value = MLXArray([Float(i)]).reshaped(1, 1, 1, 1)
            _ = cache.update(keys: value, values: value)
        }
        XCTAssertEqual(cache.offset, 8)
        XCTAssertEqual(cache.keysArray?.reshaped(-1).asArray(Float.self), [5, 6, 7])
        cache.trim(cache.offset)
        XCTAssertEqual(cache.offset, 0)
        XCTAssertNil(cache.keysArray)
    }

    func testTokenizerSkipsPaddingAndDecodesSplitUTF8() throws {
        let pieces: [(String, Int)] = [
            ("<unk>", 2), ("<s>", 3), ("</s>", 3), ("<pad>", 3),
            ("▁Hello", 1), (",", 1), ("▁world", 1), ("!", 1),
            ("<0xC3>", 6), ("<0xA9>", 6),
        ]
        var data = Data()
        for (text, type) in pieces {
            let utf8 = Array(text.utf8)
            let message = [UInt8(10), UInt8(utf8.count)] + utf8 + [24, UInt8(type)]
            data.append(contentsOf: [10, UInt8(message.count)] + message)
        }
        var tokenizer = KyutaiSTTTokenizer(vocabulary: try SentencePieceModel(data: data))
        let result = [0, 3, 4, 5, 6, 7, 8, 9].map { tokenizer.append($0) }.joined()
        XCTAssertEqual(result, " Hello, world!é")
    }

    func testTinyDecoderCheckpointRequiresAllTensors() throws {
        let config = try JSONDecoder().decode(KyutaiSTTConfig.self, from: configurationData([
            "dim": 8, "num_heads": 2, "num_layers": 1, "text_card": 8,
        ]))
        let decoder = KyutaiSTTDecoder(config: config)
        let weights = Dictionary(uniqueKeysWithValues: decoder.parameters().flattened())
        XCTAssertEqual(weights["emb.31.weight"]?.shape, [2049, 8])
        XCTAssertEqual(weights["layers.0.gating.linear_in.weight"]?.shape, [44, 8])
        XCTAssertNoThrow(try decoder.update(parameters: .unflattened(weights), verify: .all))
        var incomplete = weights
        incomplete.removeValue(forKey: "text_linear.weight")
        XCTAssertThrowsError(try decoder.update(parameters: .unflattened(incomplete), verify: .all))
    }
}

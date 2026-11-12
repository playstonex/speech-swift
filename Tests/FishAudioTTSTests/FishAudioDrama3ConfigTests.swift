import Foundation
import XCTest
@testable import FishAudioTTS

final class FishAudioDrama3ConfigTests: XCTestCase {
    func fixture() throws -> [String: Any] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Resources/drama3_config.json")
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    func testReleasedShapeAndMixedQuantization() throws {
        let c = try FishAudioDrama3Config.parse(fixture())
        XCTAssertEqual(c.text.layerCount, 40)
        XCTAssertEqual(c.text.layerTypes.filter { $0 == "linear_attention" }.count, 30)
        XCTAssertEqual(c.text.expertCount, 256)
        XCTAssertEqual(c.text.expertsPerToken, 8)
        XCTAssertEqual(c.text.hiddenSize, 2048)
        XCTAssertEqual(c.text.rotaryDim, 64)
        XCTAssertEqual(c.tower.codebookCount, 10)
        XCTAssertEqual(c.tower.sharedLayerCount, 4)
        XCTAssertEqual(c.tower.backboneCodebooks, 1)
        XCTAssertEqual(c.quantization["backbone.model.layers.0.mlp.switch_mlp.gate_proj"]?.bits, 4)
        XCTAssertEqual(c.quantization["semantic_head"]?.bits, 8)
        XCTAssertEqual(c.semanticStartTokenId, 248099)
        XCTAssertEqual(c.eosTokenId, 248046)
    }

    func testRejectsRawSourceAndIncompatiblePrompt() throws {
        for (key, value) in [("model_type", "fish_omni" as Any), ("speech_template_version", "unknown" as Any),
                             ("quantization_config", ["quant_method": "bsgemm-moe"] as Any),
                             ("eos_token_id", 248099 as Any), ("semantic_end_token_id", 248100 as Any)] {
            var root = try fixture(); root[key] = value
            XCTAssertThrowsError(try FishAudioDrama3Config.parse(root), key)
        }
    }

    func testRejectsInconsistentGeometryBeforeAllocatingWeights() throws {
        for (key, value) in [("num_key_value_heads", 3 as Any), ("linear_num_key_heads", 3 as Any),
                             ("num_experts_per_tok", 257 as Any), ("conv_kernel_dim", 0 as Any),
                             ("linear_key_head_dim", 4 as Any), ("linear_value_head_dim", 3 as Any),
                             ("use_qk_norm", false as Any), ("rotary_interleaved", true as Any)] {
            var root = try fixture(), t = try XCTUnwrap(root["text_config"] as? [String: Any])
            t[key == "conv_kernel_dim" ? "linear_conv_kernel_dim" : key] = value
            root["text_config"] = t
            XCTAssertThrowsError(try FishAudioDrama3Config.parse(root), key)
        }
        var root = try fixture(), a = try XCTUnwrap(root["audio_decoder_config"] as? [String: Any])
        a["vq_backbone_codebooks"] = 10; root["audio_decoder_config"] = a
        XCTAssertThrowsError(try FishAudioDrama3Config.parse(root))
    }

    func testRejectsOversizeRotaryFactorsWithoutIntegerOverflow() throws {
        for factor in [1.1, 1e20, Double(Float.greatestFiniteMagnitude)] {
            var root = try fixture()
            var text = try XCTUnwrap(root["text_config"] as? [String: Any])
            var rope = try XCTUnwrap(text["rope_parameters"] as? [String: Any])
            rope["partial_rotary_factor"] = factor
            text["rope_parameters"] = rope
            root["text_config"] = text
            XCTAssertThrowsError(try FishAudioDrama3Config.parse(root), "Text factor: \(factor)")

            root = try fixture()
            var tower = try XCTUnwrap(root["audio_decoder_config"] as? [String: Any])
            tower["rotary_percent"] = factor
            root["audio_decoder_config"] = tower
            XCTAssertThrowsError(try FishAudioDrama3Config.parse(root), "Tower factor: \(factor)")
        }
    }

    func testTowerQKNormConfigDefaultRemainsAccepted() throws {
        var root = try fixture(), a = try XCTUnwrap(root["audio_decoder_config"] as? [String: Any])
        a["attention_qk_norm"] = false; root["audio_decoder_config"] = a
        XCTAssertEqual(try FishAudioDrama3Config.parse(root).tower.headDim, 256)
    }

    func testSamplingOptionsRejectInvalidAndUnboundedWork() throws {
        try FishAudioDrama3Options().validate()
        for options in [FishAudioDrama3Options(maxFrames: 0), .init(maxFrames: 8193), .init(minFrames: -1),
                        .init(temperature: .nan), .init(temperature: .infinity), .init(topP: 0), .init(topP: 1.1),
                        .init(topK: 0), .init(topK: 4098), .init(streamingChunkFrames: 0), .init(streamingChunkFrames: 129)] {
            XCTAssertThrowsError(try options.validate())
        }
    }

    func testStreamingDefaultsAndBoundaries() throws {
        XCTAssertEqual(FishAudioDrama3Options().streamingChunkFrames, 3)
        try FishAudioDrama3Options(streamingChunkFrames: 1).validate()
        try FishAudioDrama3Options(streamingChunkFrames: 128).validate()
    }

    func testSeedIsLocalAndRepeatable() throws {
        var a = Drama3Sampler(state: 42), b = Drama3Sampler(state: 42)
        let logits: [Float] = [0, 0.1, 0.2, 0.3]
        let first = try (0..<30).map { _ in try a.sample(logits, temperature: 1, topK: 4, topP: 1) }
        let second = try (0..<30).map { _ in try b.sample(logits, temperature: 1, topK: 4, topP: 1) }
        XCTAssertEqual(first, second)
        XCTAssertGreaterThan(Set(first).count, 1)
    }

    func testSamplerDropsNucleusCrossingTokenAndHonorsSuppression() throws {
        var sampler = Drama3Sampler(state: 7)
        let logits: [Float] = [log(0.6), log(0.3), log(0.1)]
        for _ in 0..<30 {
            XCTAssertEqual(try sampler.sample(logits, temperature: 1, topK: 3, topP: 0.8), 0)
        }
        XCTAssertEqual(try sampler.sample(logits, temperature: 0, topK: 3, topP: 1, suppressed: 0), 1)
        XCTAssertThrowsError(try sampler.sample([.nan, 1], temperature: 1, topK: 2, topP: 1))
        XCTAssertThrowsError(try sampler.sample([.infinity], temperature: 0, topK: 1, topP: 1))
    }
}

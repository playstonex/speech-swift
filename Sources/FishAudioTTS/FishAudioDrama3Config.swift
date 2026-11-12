import Foundation

/// Configuration of the MLX Drama 3 format, with shifted FP32 normalization weights.
public struct FishAudioDrama3Config: Sendable {
    public struct Quantization: Sendable, Equatable {
        public let bits: Int
        public let groupSize: Int
    }

    public struct Text: Sendable {
        public let vocabularySize, hiddenSize, layerCount: Int
        public let layerTypes: [String]
        public let attentionHeads, keyValueHeads, headDim, rotaryDim: Int
        public let ropeTheta, normEps: Float
        public let maxSequenceLength, convKernel, keyHeads, valueHeads, keyDim, valueDim: Int
        public let expertCount, expertsPerToken, expertSize, sharedExpertSize: Int
        public let normalizeExpertWeights: Bool
    }

    public struct Tower: Sendable {
        public let hiddenSize, intermediateSize, attentionHeads, keyValueHeads, headDim, rotaryDim: Int
        public let ropeTheta, normEps: Float
        public let vocabularySize, codebookCount, sharedLayerCount, backboneCodebooks: Int
    }

    public let text: Text
    public let tower: Tower
    public let semanticStartTokenId, semanticEndTokenId, eosTokenId: Int
    public let quantization: [String: Quantization]
    public let defaultQuantization: Quantization?

    public static func load(from url: URL) throws -> Self {
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw FishAudioError.invalidConfig("Drama 3 config must be an object")
        }
        return try parse(root)
    }

    static func parse(_ root: [String: Any]) throws -> Self {
        func invalid(_ message: String) -> FishAudioError { .invalidConfig("Drama 3: \(message)") }
        func integer(_ object: [String: Any], _ key: String) throws -> Int {
            guard let value = object[key] as? Int, value > 0 else { throw invalid("invalid \(key)") }
            return value
        }
        func scalar(_ object: [String: Any], _ key: String) throws -> Float {
            guard let value = object[key] as? NSNumber, value.floatValue.isFinite, value.floatValue > 0 else {
                throw invalid("invalid \(key)")
            }
            return value.floatValue
        }
        func rotaryDimensions(_ headDim: Int, factor: Float) throws -> Int {
            let width = (Double(headDim) * Double(factor)).rounded(.towardZero)
            guard factor <= 1, let dimensions = Int(exactly: width),
                  dimensions > 0, dimensions <= headDim, dimensions % 2 == 0 else {
                throw invalid("invalid rotary dimensions")
            }
            return dimensions
        }
        func flags(_ object: [String: Any], _ required: [String: Bool]) throws {
            for (key, expected) in required where (object[key] as? Bool ?? expected) != expected {
                throw invalid("unsupported \(key)")
            }
        }
        guard root["model_type"] as? String == "fish_drama3", root["quantization_config"] == nil,
              root["speech_template_version"] as? String == "reference-bracket-v1",
              root["audio_encoder_config"] == nil || root["audio_encoder_config"] is NSNull,
              let t = root["text_config"] as? [String: Any],
              let a = root["audio_decoder_config"] as? [String: Any],
              let rope = t["rope_parameters"] as? [String: Any] else {
            throw invalid("expected an exported fish_drama3 bundle (raw FP8 weights are unsupported)")
        }
        try flags(t, ["rotary_interleaved": false, "use_qk_norm": true, "attn_output_gate": true,
                      "layernorm_zero_centered_gamma": true, "shared_expert_gate": true,
                      "tie_word_embeddings": false, "attention_bias": false])
        try flags(a, ["rotary_interleaved": true, "attention_output_gate": true,
                      "layernorm_zero_centered_gamma": true, "use_moe": false,
                      "mtp_use_swa": false, "attention_qkv_bias": false, "attention_o_bias": false])
        guard a["decoder_type"] as? String == "codebook_mtp_shared",
              ["backbone_partial", "mtp_shared_pre"].contains(a["vq_feature_placement"] as? String ?? ""),
              (rope["rope_type"] as? String ?? "default") == "default",
              (t["hidden_act"] as? String ?? "silu") == "silu" else { throw invalid("unsupported architecture") }
        let layers = try integer(t, "num_hidden_layers")
        guard let types = t["layer_types"] as? [String], types.count == layers,
              types.allSatisfy({ ["linear_attention", "full_attention"].contains($0) }),
              (t["mlp_layer_types"] as? [String] ?? Array(repeating: "moe", count: layers)) == Array(repeating: "moe", count: layers)
        else { throw invalid("invalid layer types") }
        let textHeadDim = try integer(t, "head_dim")
        let textRotary = try rotaryDimensions(textHeadDim, factor: scalar(rope, "partial_rotary_factor"))
        let text = try Text(
            vocabularySize: integer(t, "vocab_size"), hiddenSize: integer(t, "hidden_size"), layerCount: layers,
            layerTypes: types, attentionHeads: integer(t, "num_attention_heads"),
            keyValueHeads: integer(t, "num_key_value_heads"), headDim: textHeadDim, rotaryDim: textRotary,
            ropeTheta: scalar(rope, "rope_theta"), normEps: scalar(t, "rms_norm_eps"),
            maxSequenceLength: integer(t, "max_position_embeddings"), convKernel: integer(t, "linear_conv_kernel_dim"),
            keyHeads: integer(t, "linear_num_key_heads"), valueHeads: integer(t, "linear_num_value_heads"),
            keyDim: integer(t, "linear_key_head_dim"), valueDim: integer(t, "linear_value_head_dim"),
            expertCount: integer(t, "num_experts"), expertsPerToken: integer(t, "num_experts_per_tok"),
            expertSize: integer(t, "moe_intermediate_size"), sharedExpertSize: integer(t, "shared_expert_intermediate_size"),
            normalizeExpertWeights: t["norm_topk_prob"] as? Bool ?? true)
        let headDim = try integer(a, "head_dim")
        guard let count = a["vq_backbone_codebooks"] as? Int, count >= 0 else { throw invalid("invalid backbone codebooks") }
        let tower = try Tower(
            hiddenSize: integer(a, "dim"), intermediateSize: integer(a, "intermediate_size"),
            attentionHeads: integer(a, "n_head"), keyValueHeads: integer(a, "n_local_heads"),
            headDim: headDim, rotaryDim: rotaryDimensions(headDim, factor: scalar(a, "rotary_percent")),
            ropeTheta: scalar(a, "rope_base"), normEps: scalar(a, "norm_eps"),
            vocabularySize: integer(a, "vocab_size"), codebookCount: integer(a, "num_codebooks"),
            sharedLayerCount: integer(a, "shared_enrichment_layers"), backboneCodebooks: count)
        let start = try integer(root, "semantic_start_token_id")
        let end = try integer(root, "semantic_end_token_id")
        let eos = try integer(root, "eos_token_id")
        guard text.attentionHeads % text.keyValueHeads == 0, text.valueHeads % text.keyHeads == 0,
              text.keyDim % 32 == 0, text.valueDim % 4 == 0,
              text.expertsPerToken <= text.expertCount, text.convKernel >= 2,
              textRotary > 0, textRotary <= textHeadDim, textRotary % 2 == 0,
              tower.attentionHeads % tower.keyValueHeads == 0,
              tower.rotaryDim > 0, tower.rotaryDim <= headDim, tower.rotaryDim % 2 == 0,
              tower.hiddenSize == text.hiddenSize, tower.codebookCount >= 2,
              count < tower.codebookCount, end >= start, end < text.vocabularySize,
              eos < text.vocabularySize, !(start...end).contains(eos),
              end - start + 1 == tower.vocabularySize,
              (a["text_dim"] as? Int ?? tower.hiddenSize) == tower.hiddenSize,
              (a["n_layer"] as? Int) == tower.sharedLayerCount,
              a["vq_feature_placement"] as? String != "mtp_shared_pre" || count == 0
        else { throw invalid("inconsistent dimensions or semantic token band") }
        func quant(_ value: [String: Any]) throws -> Quantization {
            let bits = try integer(value, "bits"), group = try integer(value, "group_size")
            guard [3, 4, 6, 8].contains(bits), [32, 64, 128].contains(group),
                  (value["mode"] as? String ?? "affine") == "affine" else { throw invalid("unsupported quantization") }
            return Quantization(bits: bits, groupSize: group)
        }
        let q = root["quantization"] as? [String: Any]
        var overrides: [String: Quantization] = [:]
        for (key, value) in q ?? [:] where value is [String: Any] {
            overrides[key] = try quant(value as! [String: Any])
        }
        return Self(text: text, tower: tower, semanticStartTokenId: start, semanticEndTokenId: end,
                    eosTokenId: eos, quantization: overrides, defaultQuantization: try q.map(quant))
    }
}

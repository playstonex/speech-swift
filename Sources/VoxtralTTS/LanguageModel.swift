import Foundation
import MLX
import MLXCommon
import MLXFast
import MLXNN

struct VoxtralTTSLanguageState {
    var caches: [(MLXArray, MLXArray)?]
    var position: Int

    static func initial(layerCount: Int) -> VoxtralTTSLanguageState {
        VoxtralTTSLanguageState(
            caches: Array(repeating: nil, count: layerCount),
            position: 0)
    }
}

final class VoxtralTTSLlamaAttention: Module {
    let queryHeads: Int
    let keyValueHeads: Int
    let headDimension: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear

    init(_ config: VoxtralTTSConfig) {
        queryHeads = config.transformer.heads
        keyValueHeads = config.transformer.kvHeads
        headDimension = config.transformer.headDim
        scale = 1 / sqrt(Float(config.transformer.headDim))
        _queryProjection.wrappedValue = Linear(
            config.transformer.dim, queryHeads * headDimension, bias: false)
        _keyProjection.wrappedValue = Linear(
            config.transformer.dim, keyValueHeads * headDimension, bias: false)
        _valueProjection.wrappedValue = Linear(
            config.transformer.dim, keyValueHeads * headDimension, bias: false)
        _outputProjection.wrappedValue = Linear(
            queryHeads * headDimension, config.transformer.dim, bias: false)
        rope = RoPE(dimensions: config.transformer.headDim, traditional: true, base: config.transformer.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: (MLXArray, MLXArray)?,
        position: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let batch = input.dim(0)
        let length = input.dim(1)
        var query = queryProjection(input)
            .reshaped(batch, length, queryHeads, headDimension)
            .transposed(0, 2, 1, 3)
        var key = keyProjection(input)
            .reshaped(batch, length, keyValueHeads, headDimension)
            .transposed(0, 2, 1, 3)
        let value = valueProjection(input)
            .reshaped(batch, length, keyValueHeads, headDimension)
            .transposed(0, 2, 1, 3)
        let offset = cache?.0.dim(2) ?? position
        query = rope(query, offset: offset)
        key = rope(key, offset: offset)
        let cachedKey = cache.map { MLX.concatenated([$0.0, key], axis: 2) } ?? key
        let cachedValue = cache.map { MLX.concatenated([$0.1, value], axis: 2) } ?? value

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if length == 1 && (cache != nil || position > 0) {
            mask = .none
        } else {
            let keyLength = cachedKey.dim(2)
            let previousLength = keyLength - length
            let causal = MLXArray.tri(
                length, m: keyLength, k: previousLength, type: Float.self) - 1
            mask = .array(
                (causal * Float.greatestFiniteMagnitude)
                    .reshaped(1, 1, length, keyLength)
                    .asType(query.dtype))
        }
        let attended = SDPA.attendAndMerge(
            qHeads: query,
            kHeads: cachedKey,
            vHeads: cachedValue,
            scale: scale,
            mask: mask)
        return (outputProjection(attended), (cachedKey, cachedValue))
    }
}

final class VoxtralTTSLlamaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(_ config: VoxtralTTSConfig) {
        _gateProjection.wrappedValue = Linear(
            config.transformer.dim, config.transformer.hiddenDim, bias: false)
        _upProjection.wrappedValue = Linear(
            config.transformer.dim, config.transformer.hiddenDim, bias: false)
        _downProjection.wrappedValue = Linear(
            config.transformer.hiddenDim, config.transformer.dim, bias: false)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

final class VoxtralTTSLlamaLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: VoxtralTTSLlamaAttention
    @ModuleInfo var mlp: VoxtralTTSLlamaMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: VoxtralTTSConfig) {
        _attention.wrappedValue = VoxtralTTSLlamaAttention(config)
        _mlp.wrappedValue = VoxtralTTSLlamaMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.transformer.dim, eps: config.transformer.normEps!)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.transformer.dim, eps: config.transformer.normEps!)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: (MLXArray, MLXArray)?,
        position: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let attentionResult = attention(
            inputLayerNorm(input), cache: cache, position: position)
        let hidden = input + attentionResult.0
        return (
            hidden + mlp(postAttentionLayerNorm(hidden)),
            attentionResult.1)
    }
}

final class VoxtralTTSLlamaModel: Module {
    let config: VoxtralTTSConfig

    @ModuleInfo(key: "embed_tokens") var tokenEmbedding: Embedding
    @ModuleInfo var layers: [VoxtralTTSLlamaLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: VoxtralTTSConfig) {
        self.config = config
        _tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.transformer.dim)
        _layers.wrappedValue = (0..<config.transformer.layers).map { _ in
            VoxtralTTSLlamaLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.transformer.dim, eps: config.transformer.normEps!)
        super.init()
    }

    func embed(_ tokenIDs: MLXArray) -> MLXArray { tokenEmbedding(tokenIDs) }

    func forward(
        embeddings: MLXArray,
        state: VoxtralTTSLanguageState
    ) -> (MLXArray, VoxtralTTSLanguageState) {
        var hidden = embeddings
        var caches = state.caches
        for index in layers.indices {
            let result = layers[index](
                hidden,
                cache: state.caches[index],
                position: state.position)
            hidden = result.0
            caches[index] = result.1
        }
        return (
            norm(hidden),
            VoxtralTTSLanguageState(
                caches: caches,
                position: state.position + embeddings.dim(1)))
    }
}

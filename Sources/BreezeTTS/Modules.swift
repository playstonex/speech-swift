import Foundation
import MLX
import MLXCommon
import MLXNN

func breezeLinear(_ input: Int, _ output: Int, quantization: BreezeQuantization?) -> Linear {
    if quantization == nil {
        return BreezeSmallBatchLinear(input, output, bias: false)
    }
    return makeMaybeQuantizedLinear(input, output, bias: false,
                                    groupSize: quantization?.groupSize ?? 64, bits: quantization?.bits ?? 0)
}

final class BreezeKVCache {
    var keys: MLXArray?
    var values: MLXArray?
    var offset: Int { keys?.dim(2) ?? 0 }
    var arrays: [MLXArray] { [keys, values].compactMap { $0 } }

    func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let nextK = keys.map { concatenated([$0, k], axis: 2) } ?? k
        let nextV = values.map { concatenated([$0, v], axis: 2) } ?? v
        keys = nextK; values = nextV
        return (nextK, nextV)
    }

    func materialize() {
        if let keys, let values { eval(keys, values) }
    }
}

/// Reciprocal angular frequencies for the depth decoder's Llama 3 scaling.
func breezeRoPEFrequencies(dimensions: Int, base: Float, scaling: BreezeRoPEScaling?) -> [Float] {
    (0..<(dimensions / 2)).map { i in
        let frequency = pow(base, Float(2 * i) / Float(dimensions))
        guard let scaling, scaling.ropeType == "llama3" else { return frequency }
        let wavelength = 2 * Float.pi * frequency
        let oldLength = Float(scaling.originalMaxPositionEmbeddings ?? 16)
        let low = scaling.lowFreqFactor ?? 1, high = scaling.highFreqFactor ?? 4
        let factor = scaling.factor ?? 1
        if wavelength > oldLength / low { return frequency * factor }
        if wavelength < oldLength / high { return frequency }
        let smooth = (oldLength / wavelength - low) / (high - low)
        return frequency / ((1 - smooth) / factor + smooth)
    }
}

private struct BreezeRotaryBuffer {
    let array: MLXArray
}

final class BreezeAttention: Module {
    let queryHeads: Int, kvHeads: Int, headDim: Int
    let scale: Float, ropeTheta: Float
    private let frequencies: BreezeRotaryBuffer?
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm?

    init(hiddenSize: Int, queryHeads: Int, kvHeads: Int, headDim: Int, eps: Float,
         ropeTheta: Float, scaling: BreezeRoPEScaling?, qkNorm: Bool, quantization: BreezeQuantization?) {
        self.queryHeads = queryHeads; self.kvHeads = kvHeads; self.headDim = headDim
        self.ropeTheta = ropeTheta; scale = 1 / sqrt(Float(headDim))
        frequencies = scaling?.ropeType == "llama3"
            ? BreezeRotaryBuffer(array: MLXArray(breezeRoPEFrequencies(dimensions: headDim, base: ropeTheta, scaling: scaling))) : nil
        _query.wrappedValue = breezeLinear(hiddenSize, queryHeads * headDim, quantization: quantization)
        _key.wrappedValue = breezeLinear(hiddenSize, kvHeads * headDim, quantization: quantization)
        _value.wrappedValue = breezeLinear(hiddenSize, kvHeads * headDim, quantization: quantization)
        _output.wrappedValue = breezeLinear(queryHeads * headDim, hiddenSize, quantization: quantization)
        _queryNorm.wrappedValue = qkNorm ? RMSNorm(dimensions: headDim, eps: eps) : nil
        _keyNorm.wrappedValue = qkNorm ? RMSNorm(dimensions: headDim, eps: eps) : nil
    }

    func callAsFunction(_ x: MLXArray, cache: BreezeKVCache) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1), offset = cache.offset
        var q = query(x).reshaped([batch, length, queryHeads, headDim])
        var k = key(x).reshaped([batch, length, kvHeads, headDim])
        if let queryNorm { q = queryNorm(q) }
        if let keyNorm { k = keyNorm(k) }
        q = MLXFast.RoPE(q.transposed(0, 2, 1, 3), dimensions: headDim, traditional: false,
                         base: frequencies == nil ? ropeTheta : nil, scale: 1, offset: offset, freqs: frequencies?.array)
        k = MLXFast.RoPE(k.transposed(0, 2, 1, 3), dimensions: headDim, traditional: false,
                         base: frequencies == nil ? ropeTheta : nil, scale: 1, offset: offset, freqs: frequencies?.array)
        let v = value(x).reshaped([batch, length, kvHeads, headDim]).transposed(0, 2, 1, 3)
        let (keys, values) = cache.update(k, v)
        let attended = SDPA.attendAndMerge(qHeads: q, kHeads: keys, vHeads: values, scale: scale,
                                           mask: length == 1 ? .none : .causal)
        return output(attended)
    }
}

final class BreezeMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(hidden: Int, intermediate: Int, quantization: BreezeQuantization?) {
        _gate.wrappedValue = breezeLinear(hidden, intermediate, quantization: quantization)
        _up.wrappedValue = breezeLinear(hidden, intermediate, quantization: quantization)
        _down.wrappedValue = breezeLinear(intermediate, hidden, quantization: quantization)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class BreezeLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: BreezeAttention
    @ModuleInfo var mlp: BreezeMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(hidden: Int, intermediate: Int, queryHeads: Int, kvHeads: Int, headDim: Int,
         eps: Float, theta: Float, scaling: BreezeRoPEScaling?, qkNorm: Bool, quantization: BreezeQuantization?) {
        _attention.wrappedValue = BreezeAttention(hiddenSize: hidden, queryHeads: queryHeads, kvHeads: kvHeads,
            headDim: headDim, eps: eps, ropeTheta: theta, scaling: scaling, qkNorm: qkNorm, quantization: quantization)
        _mlp.wrappedValue = BreezeMLP(hidden: hidden, intermediate: intermediate, quantization: quantization)
        _inputNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, cache: BreezeKVCache) -> MLXArray {
        let h = x + attention(inputNorm(x), cache: cache)
        return h + mlp(postAttentionNorm(h))
    }
}

final class BreezeAudioEmbedding: Module {
    let numCodebooks: Int, vocabSize: Int
    @ModuleInfo(key: "embed_audio_tokens") var embedAudioTokens: Embedding
    @ModuleInfo(key: "audio_embeds_projector") var audioEmbedsProjector: Linear?

    init(config: BreezeTTSConfig) {
        numCodebooks = config.numCodebooks; vocabSize = config.audioVocabSize
        _embedAudioTokens.wrappedValue = Embedding(embeddingCount: numCodebooks * vocabSize,
                                                   dimensions: config.audioEmbedSize)
        _audioEmbedsProjector.wrappedValue = config.audioEmbedSize == config.backboneConfig.hiddenSize ? nil
            : breezeLinear(config.audioEmbedSize, config.backboneConfig.hiddenSize, quantization: config.quantization)
    }

    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        let offsets = MLX.arange(numCodebooks).reshaped([1, 1, numCodebooks]) * vocabSize
        var hidden = embedAudioTokens(codes + offsets)
        if let audioEmbedsProjector { hidden = audioEmbedsProjector(hidden) }
        return hidden.sum(axis: -2)
    }
}

final class BreezeBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: BreezeAudioEmbedding
    @ModuleInfo var layers: [BreezeLayer]
    @ModuleInfo var norm: RMSNorm
    init(config: BreezeTTSConfig) {
        let c = config.backboneConfig
        _embedTokens.wrappedValue = BreezeAudioEmbedding(config: config)
        _layers.wrappedValue = (0..<c.numHiddenLayers).map { _ in
            BreezeLayer(hidden: c.hiddenSize, intermediate: c.intermediateSize, queryHeads: c.numAttentionHeads,
                kvHeads: c.numKeyValueHeads, headDim: c.headDim, eps: c.rmsNormEps, theta: c.ropeTheta,
                scaling: c.ropeScaling, qkNorm: true, quantization: c.quantization)
        }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
    func makeCache() -> [BreezeKVCache] { layers.map { _ in BreezeKVCache() } }
    func callAsFunction(_ embeddings: MLXArray, cache: [BreezeKVCache]) -> MLXArray {
        var hidden = embeddings
        for (index, layer) in layers.enumerated() { hidden = layer(hidden, cache: cache[index]) }
        return norm(hidden)
    }
}

final class BreezeDepthModel: Module {
    let config: BreezeDepthDecoderConfig
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "inputs_embeds_projector") var inputProjector: Linear
    @ModuleInfo(key: "backbone_hidden_state_projector") var backboneProjector: Linear?
    @ModuleInfo var layers: [BreezeLayer]
    @ModuleInfo var norm: RMSNorm

    init(config c: BreezeDepthDecoderConfig) {
        config = c
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.numCodebooks * c.vocabSize, dimensions: c.audioEmbedSize)
        _inputProjector.wrappedValue = breezeLinear(c.audioEmbedSize, c.hiddenSize, quantization: c.quantization)
        _backboneProjector.wrappedValue = c.backboneHiddenSize == c.audioEmbedSize ? nil
            : breezeLinear(c.backboneHiddenSize, c.audioEmbedSize, quantization: c.quantization)
        _layers.wrappedValue = (0..<c.numHiddenLayers).map { _ in
            BreezeLayer(hidden: c.hiddenSize, intermediate: c.intermediateSize, queryHeads: c.numAttentionHeads,
                kvHeads: c.numKeyValueHeads, headDim: c.headDim, eps: c.rmsNormEps, theta: c.ropeTheta,
                scaling: c.ropeScaling, qkNorm: false, quantization: c.quantization)
        }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }

    func forward(_ embeddings: MLXArray, cache: [BreezeKVCache]) -> MLXArray {
        var hidden = inputProjector(embeddings)
        for (index, layer) in layers.enumerated() { hidden = layer(hidden, cache: cache[index]) }
        return norm(hidden)
    }

    func prefill(first: Int, backbone: MLXArray, cache: [BreezeKVCache]) -> MLXArray {
        let hidden = backboneProjector?(backbone) ?? backbone
        let ids = MLXArray(Array(repeating: Int32(first), count: backbone.dim(0))).reshaped([backbone.dim(0), 1])
        return forward(concatenated([hidden.expandedDimensions(axis: 1), embedTokens(ids)], axis: 1), cache: cache)
    }

    func step(token: MLXArray, codebook: Int, batchSize: Int, cache: [BreezeKVCache]) -> MLXArray {
        let ids = broadcast(token.asType(.int32).reshaped([1, 1]), to: [batchSize, 1])
            + Int32(codebook * config.vocabSize)
        return forward(embedTokens(ids), cache: cache)
    }

    func step(token: Int, codebook: Int, batchSize: Int, cache: [BreezeKVCache]) -> MLXArray {
        let ids = MLXArray(Array(repeating: Int32(token + codebook * config.vocabSize), count: batchSize))
            .reshaped([batchSize, 1])
        return forward(embedTokens(ids), cache: cache)
    }
}

final class BreezeCodebooksHead: Module {
    @ParameterInfo var weight: MLXArray
    init(config: BreezeDepthDecoderConfig) {
        _weight.wrappedValue = MLXArray.zeros([config.numCodebooks - 1, config.hiddenSize, config.vocabSize])
    }
}

final class BreezeDepthDecoder: Module {
    @ModuleInfo var model: BreezeDepthModel
    @ModuleInfo(key: "codebooks_head") var codebooksHead: BreezeCodebooksHead
    init(config: BreezeDepthDecoderConfig) {
        _model.wrappedValue = BreezeDepthModel(config: config)
        _codebooksHead.wrappedValue = BreezeCodebooksHead(config: config)
    }
    func logits(_ hidden: MLXArray, codebook: Int) -> MLXArray {
        matmul(hidden[0..., -1, 0...], codebooksHead.weight[codebook - 1])
    }
}

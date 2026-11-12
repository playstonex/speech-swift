import Foundation
import MLX
import MLXNN

final class Drama3TowerAttention: Module {
    @ModuleInfo var wqkv: Drama3Linear
    @ModuleInfo var wo: Drama3Linear
    @ModuleInfo(key: "q_norm") var qNorm: Drama3Norm
    @ModuleInfo(key: "k_norm") var kNorm: Drama3Norm
    let heads, kvHeads, headDim: Int
    let rope: RoPE
    init(_ c: FishAudioDrama3Config, _ path: String) {
        let t = c.tower
        heads = t.attentionHeads; kvHeads = t.keyValueHeads; headDim = t.headDim
        _wqkv.wrappedValue = Drama3Linear(t.hiddenSize, 2 * (heads + kvHeads) * headDim, c, path + ".wqkv")
        _wo.wrappedValue = Drama3Linear(heads * headDim, t.hiddenSize, c, path + ".wo")
        // Tower Q/K norms are required even when attention_qk_norm is false in the source config.
        _qNorm.wrappedValue = Drama3Norm(headDim, t.normEps)
        _kNorm.wrappedValue = Drama3Norm(headDim, t.normEps)
        rope = RoPE(dimensions: t.rotaryDim, traditional: true, base: t.ropeTheta)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, cache: Drama3Cache) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), qw = heads * headDim, kw = kvHeads * headDim
        let parts = split(wqkv(x), indices: [qw, 2 * qw, 2 * qw + kw], axis: -1)
        let q = qNorm(parts[0].reshaped(b, n, heads, headDim)).transposed(0, 2, 1, 3)
        let k = kNorm(parts[2].reshaped(b, n, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = parts[3].reshaped(b, n, kvHeads, headDim).transposed(0, 2, 1, 3)
        let offset = cache.keys?.dim(2) ?? 0
        let qr = rope(q.asType(.float32), offset: offset).asType(q.dtype)
        let kr = rope(k.asType(.float32), offset: offset).asType(k.dtype)
        return wo(drama3Attention(qr, kr, v, cache: cache, headDim: headDim) * sigmoid(parts[1]))
    }
}

final class Drama3TowerMLP: Module {
    @ModuleInfo var w1: Drama3Linear
    @ModuleInfo var w2: Drama3Linear
    @ModuleInfo var w3: Drama3Linear
    init(_ c: FishAudioDrama3Config, _ path: String) {
        let t = c.tower
        _w1.wrappedValue = Drama3Linear(t.hiddenSize, t.intermediateSize, c, path + ".w1")
        _w2.wrappedValue = Drama3Linear(t.intermediateSize, t.hiddenSize, c, path + ".w2")
        _w3.wrappedValue = Drama3Linear(t.hiddenSize, t.intermediateSize, c, path + ".w3")
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

final class Drama3TowerBlock: Module {
    @ModuleInfo var attention: Drama3TowerAttention
    @ModuleInfo(key: "attention_norm") var attentionNorm: Drama3Norm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: Drama3Norm
    @ModuleInfo(key: "feed_forward") var feedForward: Drama3TowerMLP
    init(_ c: FishAudioDrama3Config, _ path: String) {
        _attention.wrappedValue = Drama3TowerAttention(c, path + ".attention")
        _attentionNorm.wrappedValue = Drama3Norm(c.tower.hiddenSize, c.tower.normEps)
        _ffnNorm.wrappedValue = Drama3Norm(c.tower.hiddenSize, c.tower.normEps)
        _feedForward.wrappedValue = Drama3TowerMLP(c, path + ".feed_forward")
        super.init()
    }
    func callAsFunction(_ x: MLXArray, cache: Drama3Cache) -> MLXArray {
        let h = x + attention(attentionNorm(x), cache: cache)
        return h + feedForward(ffnNorm(h))
    }
}

final class Drama3TowerLayer: Module {
    @ModuleInfo(key: "semantic_embedding") var semanticEmbedding: Embedding?
    @ModuleInfo(key: "codebook_embedding") var codebookEmbedding: Embedding?
    @ModuleInfo var enorm: Drama3Norm
    @ModuleInfo var hnorm: Drama3Norm
    @ModuleInfo(key: "eh_proj") var projection: Drama3Linear
    @ModuleInfo(key: "transformer_layer") var transformer: Drama3TowerBlock
    @ModuleInfo(key: "final_layernorm") var finalNorm: Drama3Norm
    let vocabSize: Int

    init(_ c: FishAudioDrama3Config, _ path: String, shared: Bool) {
        let t = c.tower
        vocabSize = t.vocabularySize
        let embedding = Embedding(weight: .zeros([vocabSize, t.hiddenSize], dtype: .bfloat16))
        if shared { _semanticEmbedding.wrappedValue = embedding }
        else { _codebookEmbedding.wrappedValue = embedding }
        _enorm.wrappedValue = Drama3Norm(t.hiddenSize, t.normEps)
        _hnorm.wrappedValue = Drama3Norm(t.hiddenSize, t.normEps)
        _projection.wrappedValue = Drama3Linear(2 * t.hiddenSize, t.hiddenSize, c, path + ".eh_proj")
        _transformer.wrappedValue = Drama3TowerBlock(c, path + ".transformer_layer")
        _finalNorm.wrappedValue = Drama3Norm(t.hiddenSize, t.normEps)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, codes: MLXArray, semantic: MLXArray,
                        cache: Drama3Cache, extra: MLXArray? = nil) -> MLXArray {
        let embedding = semanticEmbedding ?? codebookEmbedding!
        var e = MLX.where(semantic.expandedDimensions(axis: -1), embedding(clip(codes, min: 0, max: vocabSize - 1)), 0)
        if let extra { e = e + extra }
        let mixed = projection(concatenated([enorm(e), hnorm(hidden)], axis: -1))
        return finalNorm(transformer(mixed, cache: cache)).asType(mixed.dtype)
    }
}

final class Drama3Tower: Module {
    @ModuleInfo(key: "codebook_embeddings") var embeddings: Embedding
    @ModuleInfo(key: "shared_layers") var sharedLayers: [Drama3TowerLayer]
    @ModuleInfo(key: "residual_layers") var residualLayers: [Drama3TowerLayer]
    @ModuleInfo(key: "codebook_heads") var heads: [Drama3Linear]
    let config: FishAudioDrama3Config

    init(_ c: FishAudioDrama3Config) {
        config = c
        let t = c.tower
        _embeddings.wrappedValue = Embedding(weight: .zeros([t.vocabularySize * t.codebookCount, t.hiddenSize], dtype: .bfloat16))
        _sharedLayers.wrappedValue = (0..<t.sharedLayerCount).map { Drama3TowerLayer(c, "tower.shared_layers.\($0)", shared: true) }
        _residualLayers.wrappedValue = (0..<(t.codebookCount - 1)).map { Drama3TowerLayer(c, "tower.residual_layers.\($0)", shared: false) }
        _heads.wrappedValue = (0..<(t.codebookCount - 1)).map { Drama3Linear(t.hiddenSize, t.vocabularySize, c, "tower.codebook_heads.\($0)") }
        super.init()
    }

    func isSemantic(_ tokens: MLXArray) -> MLXArray {
        (tokens .>= config.semanticStartTokenId) .&& (tokens .<= config.semanticEndTokenId)
    }

    func embeddingSum(_ frames: MLXArray, first: Int, end: Int) -> MLXArray {
        if first == end {
            return .zeros([frames.dim(0), frames.dim(1), config.tower.hiddenSize], dtype: embeddings.weight.dtype)
        }
        let indices = frames[.ellipsis, (1 + first)..<(1 + end)] + MLXArray(first..<end) * config.tower.vocabularySize
        return embeddings(indices).sum(axis: -2)
    }

    func embedBackbone(_ frames: MLXArray, text: MLXArray) -> MLXArray {
        let count = config.tower.backboneCodebooks
        let mixed = (text + embeddingSum(frames, first: 1, end: 1 + count)) / sqrt(Float(1 + count))
        return MLX.where(isSemantic(frames[.ellipsis, 0]).expandedDimensions(axis: -1), mixed, text)
    }

    func enrich(_ hidden: MLXArray, code: MLXArray, semantic: MLXArray,
                frames: MLXArray, caches: [Drama3Cache]) -> MLXArray {
        let unseen = embeddingSum(frames, first: 1 + config.tower.backboneCodebooks, end: config.tower.codebookCount)
        let extra = MLX.where(isSemantic(frames[.ellipsis, 0]).expandedDimensions(axis: -1), unseen, 0)
        var h = hidden
        for (index, layer) in sharedLayers.enumerated() {
            h = layer(h, codes: code, semantic: semantic, cache: caches[index], extra: index == 0 ? extra : nil)
        }
        return h
    }

    func prefill(_ hidden: MLXArray, frames: MLXArray, next: MLXArray, caches: [Drama3Cache]) {
        let semantic = isSemantic(next[.ellipsis, 0])
        var h = enrich(hidden, code: next[.ellipsis, 1], semantic: semantic, frames: frames, caches: caches)
        for (index, layer) in residualLayers.enumerated() {
            h = layer(h, codes: next[.ellipsis, index + 1], semantic: semantic, cache: caches[sharedLayers.count + index])
        }
        // Teacher forcing builds a lazy graph whose cache must be evaluated by the caller.
    }

    func decode(_ hidden: MLXArray, code: Int, frame: MLXArray, caches: [Drama3Cache],
                sample: (MLXArray) throws -> Int) rethrows -> [Int] {
        let semantic = MLXArray.ones([1, 1], dtype: .bool)
        var codes = [code]
        var h = enrich(hidden, code: MLXArray([Int32(code)]).reshaped(1, 1), semantic: semantic,
                       frames: frame, caches: caches)
        for (index, layer) in residualLayers.enumerated() {
            h = layer(h, codes: MLXArray([Int32(codes.last!)]).reshaped(1, 1), semantic: semantic,
                      cache: caches[sharedLayers.count + index])
            codes.append(try sample(heads[index](h)))
        }
        return codes
    }
}

final class Drama3Network: Module {
    @ModuleInfo var backbone: Drama3Backbone
    @ModuleInfo(key: "semantic_head") var semanticHead: Drama3Linear
    @ModuleInfo var tower: Drama3Tower
    let config: FishAudioDrama3Config

    init(_ config: FishAudioDrama3Config) {
        self.config = config
        _backbone.wrappedValue = Drama3Backbone(config)
        _semanticHead.wrappedValue = Drama3Linear(config.text.hiddenSize, config.tower.vocabularySize + 1, config, "semantic_head")
        _tower.wrappedValue = Drama3Tower(config)
        super.init()
    }

    func hidden(_ frames: MLXArray, caches: [Drama3Cache]) -> MLXArray {
        let tokens = backbone.model.embeddings(frames[.ellipsis, 0])
        return backbone.model(tower.embedBackbone(frames, text: tokens), caches: caches)
    }
}

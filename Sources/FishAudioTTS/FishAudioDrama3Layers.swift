import Foundation
import MLX
import MLXCommon
import MLXFast
import MLXLMCommon
import MLXNN

/// Packed placeholders avoid allocating a floating-point copy of the MoE weights during loading.
final class Drama3Linear: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo var scales: MLXArray?
    @ParameterInfo var biases: MLXArray?
    let quantization: FishAudioDrama3Config.Quantization?

    init(_ input: Int, _ output: Int, _ config: FishAudioDrama3Config, _ path: String, experts: Int? = nil) {
        let q = config.quantization[path]
        quantization = q
        let leading = experts.map { [$0] } ?? []
        _weight.wrappedValue = .zeros(leading + [output, q.map { input / 32 * $0.bits } ?? input],
                                     dtype: q == nil ? .bfloat16 : .uint32)
        if let q {
            _scales.wrappedValue = .zeros(leading + [output, input / q.groupSize], dtype: .bfloat16)
            _biases.wrappedValue = .zeros(leading + [output, input / q.groupSize], dtype: .bfloat16)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let q = quantization, let scales {
            return quantizedMM(x, weight, scales: scales, biases: biases,
                                   transpose: true, groupSize: q.groupSize, bits: q.bits)
        }
        return matmul(x.asType(weight.dtype), weight.swappedAxes(-1, -2))
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        if let q = quantization, let scales {
            return gatherQuantizedMM(x, weight, scales: scales, biases: biases, rhsIndices: indices,
                                     transpose: true, groupSize: q.groupSize, bits: q.bits)
        }
        return gatherMM(x.asType(weight.dtype), weight.swappedAxes(-1, -2), rhsIndices: indices)
    }
}

/// Converted norms already contain 1 + gamma, kept in FP32 to preserve small offsets.
final class Drama3Norm: Module {
    @ParameterInfo var weight: MLXArray
    let eps: Float
    init(_ dimensions: Int, _ eps: Float) {
        self.eps = eps
        _weight.wrappedValue = .ones([dimensions])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps).asType(x.dtype)
    }
}

final class Drama3Cache {
    var keys: MLXArray?
    var values: MLXArray?
    var convolution: MLXArray?
    var recurrent: MLXArray?
    var arrays: [MLXArray] { [keys, values, convolution, recurrent].compactMap { $0 } }
}

func drama3Attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                     cache: Drama3Cache, headDim: Int) -> MLXArray {
    let past = cache.keys?.dim(2) ?? 0
    let keys = cache.keys.map { concatenated([$0, k], axis: 2) } ?? k
    let values = cache.values.map { concatenated([$0, v], axis: 2) } ?? v
    cache.keys = keys
    cache.values = values
    let length = q.dim(2)
    let mask: MLXFast.ScaledDotProductAttentionMaskMode
    if length == 1 { mask = .none }
    else {
        let causal = MLXArray.tri(length, m: keys.dim(2), k: past, type: Float.self) - 1
        mask = .array((causal * Float.greatestFiniteMagnitude).asType(q.dtype))
    }
    return SDPA.attendAndMerge(qHeads: q, kHeads: keys, vHeads: values,
                              scale: 1 / sqrt(Float(headDim)), mask: mask)
}

final class Drama3Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Drama3Linear
    @ModuleInfo(key: "k_proj") var kProj: Drama3Linear
    @ModuleInfo(key: "v_proj") var vProj: Drama3Linear
    @ModuleInfo(key: "o_proj") var oProj: Drama3Linear
    @ModuleInfo(key: "q_norm") var qNorm: Drama3Norm
    @ModuleInfo(key: "k_norm") var kNorm: Drama3Norm
    let heads, kvHeads, headDim: Int
    let rope: RoPE

    init(_ config: FishAudioDrama3Config, _ path: String) {
        let t = config.text
        heads = t.attentionHeads; kvHeads = t.keyValueHeads; headDim = t.headDim
        _qProj.wrappedValue = Drama3Linear(t.hiddenSize, 2 * heads * headDim, config, path + ".q_proj")
        _kProj.wrappedValue = Drama3Linear(t.hiddenSize, kvHeads * headDim, config, path + ".k_proj")
        _vProj.wrappedValue = Drama3Linear(t.hiddenSize, kvHeads * headDim, config, path + ".v_proj")
        _oProj.wrappedValue = Drama3Linear(heads * headDim, t.hiddenSize, config, path + ".o_proj")
        _qNorm.wrappedValue = Drama3Norm(headDim, t.normEps)
        _kNorm.wrappedValue = Drama3Norm(headDim, t.normEps)
        rope = RoPE(dimensions: t.rotaryDim, traditional: false, base: t.ropeTheta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Drama3Cache) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), offset = cache.keys?.dim(2) ?? 0
        let pair = split(qProj(x).reshaped(b, n, heads, 2 * headDim), parts: 2, axis: -1)
        let q = qNorm(pair[0]).transposed(0, 2, 1, 3)
        let k = kNorm(kProj(x).reshaped(b, n, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, n, kvHeads, headDim).transposed(0, 2, 1, 3)
        let qr = rope(q.asType(.float32), offset: offset).asType(q.dtype)
        let kr = rope(k.asType(.float32), offset: offset).asType(k.dtype)
        let attended = drama3Attention(qr, kr, v, cache: cache, headDim: headDim)
        return oProj(attended * sigmoid(pair[1].reshaped(b, n, heads * headDim)))
    }
}

final class Drama3Convolution: Module {
    @ParameterInfo var weight: MLXArray
    init(channels: Int, kernel: Int) {
        _weight.wrappedValue = .zeros([channels, kernel, 1], dtype: .bfloat16)
        super.init()
    }
}

final class Drama3DeltaNet: Module {
    @ModuleInfo(key: "in_proj_qkv") var qkv: Drama3Linear
    @ModuleInfo(key: "in_proj_z") var z: Drama3Linear
    @ModuleInfo(key: "in_proj_b") var beta: Drama3Linear
    @ModuleInfo(key: "in_proj_a") var decay: Drama3Linear
    @ModuleInfo(key: "out_proj") var output: Drama3Linear
    @ModuleInfo var conv1d: Drama3Convolution
    @ModuleInfo var norm: Drama3Norm
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    let config: FishAudioDrama3Config.Text

    init(_ c: FishAudioDrama3Config, _ path: String) {
        config = c.text
        let t = c.text, channels = 2 * t.keyHeads * t.keyDim + t.valueHeads * t.valueDim
        _qkv.wrappedValue = Drama3Linear(t.hiddenSize, channels, c, path + ".in_proj_qkv")
        _z.wrappedValue = Drama3Linear(t.hiddenSize, t.valueHeads * t.valueDim, c, path + ".in_proj_z")
        _beta.wrappedValue = Drama3Linear(t.hiddenSize, t.valueHeads, c, path + ".in_proj_b")
        _decay.wrappedValue = Drama3Linear(t.hiddenSize, t.valueHeads, c, path + ".in_proj_a")
        _output.wrappedValue = Drama3Linear(t.valueHeads * t.valueDim, t.hiddenSize, c, path + ".out_proj")
        _conv1d.wrappedValue = Drama3Convolution(channels: channels, kernel: t.convKernel)
        _norm.wrappedValue = Drama3Norm(t.valueDim, t.normEps)
        _aLog.wrappedValue = .zeros([t.valueHeads], dtype: .bfloat16)
        _dtBias.wrappedValue = .zeros([t.valueHeads], dtype: .bfloat16)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Drama3Cache) -> MLXArray {
        let t = config, b = x.dim(0), n = x.dim(1), keyWidth = t.keyHeads * t.keyDim
        let projected = qkv(x), channels = projected.dim(-1)
        let history = cache.convolution ?? .zeros([b, t.convKernel - 1, channels], dtype: projected.dtype)
        let padded = concatenated([history, projected], axis: 1)
        cache.convolution = padded[0..., (padded.dim(1) - t.convKernel + 1)..., 0...]
        let mixed = silu(MLX.conv1d(padded, conv1d.weight, groups: channels))
        let parts = split(mixed, indices: [keyWidth, 2 * keyWidth], axis: -1)
        // The reference adds epsilon to the sum of squares, rather than its mean.
        func normalize(_ x: MLXArray) -> MLXArray {
            let f = x.asType(.float32)
            return f * rsqrt((f * f).sum(axis: -1, keepDims: true) + 1e-6)
        }
        let q = normalize(parts[0].reshaped(b, n, t.keyHeads, t.keyDim)) / sqrt(Float(t.keyDim))
        let k = normalize(parts[1].reshaped(b, n, t.keyHeads, t.keyDim))
        let v = parts[2].reshaped(b, n, t.valueHeads, t.valueDim)
        let (core, state) = gatedDeltaUpdate(q: q, k: k, v: v.asType(.float32),
                                            a: decay(x).asType(.float32), b: beta(x).asType(.float32),
                                            aLog: aLog.asType(.float32), dtBias: dtBias.asType(.float32), state: cache.recurrent)
        cache.recurrent = state
        let gate = z(x).reshaped(b, n, t.valueHeads, t.valueDim)
        let cf = core.asType(x.dtype).asType(.float32)
        let normalized = (cf * rsqrt((cf * cf).mean(axis: -1, keepDims: true) + t.normEps)).asType(x.dtype)
        let gated = ((normalized * norm.weight).asType(.float32) * silu(gate.asType(.float32))).asType(x.dtype)
        return output(gated.reshaped(b, n, t.valueHeads * t.valueDim))
    }
}

final class Drama3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Drama3Linear
    @ModuleInfo(key: "up_proj") var up: Drama3Linear
    @ModuleInfo(key: "down_proj") var down: Drama3Linear
    init(_ dim: Int, _ intermediate: Int, _ config: FishAudioDrama3Config, _ path: String, experts: Int? = nil) {
        _gate.wrappedValue = Drama3Linear(dim, intermediate, config, path + ".gate_proj", experts: experts)
        _up.wrappedValue = Drama3Linear(dim, intermediate, config, path + ".up_proj", experts: experts)
        _down.wrappedValue = Drama3Linear(intermediate, dim, config, path + ".down_proj", experts: experts)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let expanded = x.expandedDimensions(axes: [-2, -3])
        return down(silu(gate(expanded, indices: indices)) * up(expanded, indices: indices), indices: indices).squeezed(axis: -2)
    }
}

final class Drama3MoE: Module {
    @ModuleInfo var gate: Drama3Linear
    @ModuleInfo(key: "switch_mlp") var experts: Drama3MLP
    @ModuleInfo(key: "shared_expert") var shared: Drama3MLP
    @ModuleInfo(key: "shared_expert_gate") var sharedGate: Drama3Linear
    let topK: Int
    let normalize: Bool
    init(_ c: FishAudioDrama3Config, _ path: String) {
        let t = c.text
        topK = t.expertsPerToken; normalize = t.normalizeExpertWeights
        _gate.wrappedValue = Drama3Linear(t.hiddenSize, t.expertCount, c, path + ".gate")
        _experts.wrappedValue = Drama3MLP(t.hiddenSize, t.expertSize, c, path + ".switch_mlp", experts: t.expertCount)
        _shared.wrappedValue = Drama3MLP(t.hiddenSize, t.sharedExpertSize, c, path + ".shared_expert")
        _sharedGate.wrappedValue = Drama3Linear(t.hiddenSize, 1, c, path + ".shared_expert_gate")
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let probabilities = softmax(gate(x).asType(.float32), axis: -1)
        let start = probabilities.dim(-1) - topK
        let indices = argPartition(probabilities, kth: start, axis: -1)[.ellipsis, start...]
        var weights = takeAlong(probabilities, indices, axis: -1)
        if normalize { weights = weights / weights.sum(axis: -1, keepDims: true) }
        let routed = (experts(x, indices: indices).asType(.float32) * weights.expandedDimensions(axis: -1)).sum(axis: -2).asType(x.dtype)
        return routed + sigmoid(sharedGate(x)) * shared(x)
    }
}

final class Drama3TextLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Drama3Attention?
    @ModuleInfo(key: "linear_attn") var delta: Drama3DeltaNet?
    @ModuleInfo(key: "input_layernorm") var inputNorm: Drama3Norm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: Drama3Norm
    @ModuleInfo var mlp: Drama3MoE
    init(_ c: FishAudioDrama3Config, index: Int) {
        let path = "backbone.model.layers.\(index)"
        if c.text.layerTypes[index] == "full_attention" {
            _attention.wrappedValue = Drama3Attention(c, path + ".self_attn")
        } else { _delta.wrappedValue = Drama3DeltaNet(c, path + ".linear_attn") }
        _inputNorm.wrappedValue = Drama3Norm(c.text.hiddenSize, c.text.normEps)
        _postNorm.wrappedValue = Drama3Norm(c.text.hiddenSize, c.text.normEps)
        _mlp.wrappedValue = Drama3MoE(c, path + ".mlp")
        super.init()
    }
    func callAsFunction(_ x: MLXArray, cache: Drama3Cache) -> MLXArray {
        let input = inputNorm(x)
        let mixed = attention.map { $0(input, cache: cache) } ?? delta!(input, cache: cache)
        let hidden = x + mixed
        return hidden + mlp(postNorm(hidden))
    }
}

final class Drama3TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embeddings: Embedding
    @ModuleInfo var layers: [Drama3TextLayer]
    @ModuleInfo var norm: Drama3Norm
    init(_ c: FishAudioDrama3Config) {
        _embeddings.wrappedValue = Embedding(weight: .zeros([c.text.vocabularySize, c.text.hiddenSize], dtype: .bfloat16))
        _layers.wrappedValue = (0..<c.text.layerCount).map { Drama3TextLayer(c, index: $0) }
        _norm.wrappedValue = Drama3Norm(c.text.hiddenSize, c.text.normEps)
        super.init()
    }
    func callAsFunction(_ embeddings: MLXArray, caches: [Drama3Cache]) -> MLXArray {
        var h = embeddings
        for (layer, cache) in zip(layers, caches) { h = layer(h, cache: cache) }
        return norm(h)
    }
}

final class Drama3Backbone: Module {
    @ModuleInfo var model: Drama3TextModel
    init(_ config: FishAudioDrama3Config) {
        _model.wrappedValue = Drama3TextModel(config)
        super.init()
    }
}

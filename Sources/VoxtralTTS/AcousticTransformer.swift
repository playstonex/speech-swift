import Foundation
import MLX
import MLXCommon
import MLXFast
import MLXNN
import MLXRandom

final class VoxtralAcousticAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    @ModuleInfo var wq: Linear
    @ModuleInfo var wk: Linear
    @ModuleInfo var wv: Linear
    @ModuleInfo var wo: Linear

    init(_ config: VoxtralTransformerConfig) {
        heads = config.heads
        kvHeads = config.kvHeads
        headDim = config.headDim
        _wq.wrappedValue = Linear(config.dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(config.dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(config.dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, config.dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1)
        let q = wq(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let k = wk(x).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)
        return wo(SDPA.attendAndMerge(qHeads: q, kHeads: k, vHeads: v,
                                     scale: 1 / sqrt(Float(headDim)), mask: MLXFast.ScaledDotProductAttentionMaskMode.none))
    }
}

final class VoxtralFeedForward: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        _w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        _w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        _w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

final class VoxtralAcousticLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo var attention: VoxtralAcousticAttention
    @ModuleInfo(key: "feed_forward") var feedForward: VoxtralFeedForward

    init(_ config: VoxtralTransformerConfig, normEps: Float) {
        _attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: normEps)
        _attention.wrappedValue = VoxtralAcousticAttention(config)
        _feedForward.wrappedValue = VoxtralFeedForward(dim: config.dim, hiddenDim: config.hiddenDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(attentionNorm(x))
        return h + feedForward(ffnNorm(h))
    }
}

final class VoxtralAcousticTransformer: Module {
    let config: VoxtralAudioModelConfig
    @ModuleInfo(key: "input_projection") var inputProjection: Linear
    @ModuleInfo(key: "llm_projection") var llmProjection: Linear
    @ModuleInfo(key: "time_projection") var timeProjection: Linear
    @ModuleInfo(key: "semantic_codebook_output") var semanticOutput: Linear
    @ModuleInfo(key: "acoustic_codebook_output") var acousticOutput: Linear
    @ModuleInfo var layers: [VoxtralAcousticLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: VoxtralTTSConfig) {
        self.config = config.audio
        let a = config.audio, t = a.transformer, eps = config.transformer.normEps!
        _inputProjection.wrappedValue = Linear(a.acousticCodebooks, t.dim, bias: false)
        _llmProjection.wrappedValue = Linear(config.transformer.dim, t.dim, bias: false)
        _timeProjection.wrappedValue = Linear(t.dim, t.dim, bias: false)
        _semanticOutput.wrappedValue = Linear(t.dim, (a.semanticSize / 128 + 1) * 128, bias: false)
        _acousticOutput.wrappedValue = Linear(t.dim, a.acousticCodebooks, bias: false)
        _layers.wrappedValue = (0..<t.layers).map { _ in VoxtralAcousticLayer(t, normEps: eps) }
        _norm.wrappedValue = RMSNorm(dimensions: t.dim, eps: eps)
        super.init()
    }

    static func timeEmbedding(_ time: Float, dimension: Int) -> MLXArray {
        let half = dimension / 2
        let frequencies = exp(-log(Float(10_000)) * MLXArray(0..<half).asType(.float32) / Float(half))
        let phase = frequencies * time
        return concatenated([cos(phase), sin(phase)]).reshaped(1, dimension)
    }

    static func semanticCode(_ logits: MLXArray, size: Int) -> MLXArray {
        let positions = MLXArray(0..<logits.dim(-1))
        let masked = MLX.where((positions .== 0) .|| (positions .>= size + 2), Float(-1e9), logits.asType(.float32))
        return argMax(masked, axis: -1).asType(.int32)
    }

    static func acousticCodes(_ latent: MLXArray, size: Int) -> MLXArray {
        (clip(round((clip(latent, min: -1, max: 1) + 1) * Float(size - 1) / 2), min: 0, max: Float(size - 1))).asType(.int32) + 2
    }

    private func velocity(_ latent: MLXArray, time: Float, hidden: MLXArray) -> MLXArray {
        let batch = hidden.dim(0)
        let embeddedTime = broadcast(timeProjection(Self.timeEmbedding(time, dimension: config.transformer.dim)), to: [batch, config.transformer.dim])
        var x = stacked([inputProjection(latent), embeddedTime, llmProjection(hidden)], axis: 1)
        for layer in layers { x = layer(x) }
        return acousticOutput(norm(x)[0..., 0, 0...])
    }

    func decodeFrame(hidden: MLXArray, options: VoxtralTTSOptions, frame: Int) throws -> MLXArray {
        let semantic = Self.semanticCode(semanticOutput(hidden), size: config.semanticSize)
        // A local key avoids reseeding other models using the process-wide RNG.
        var latent = MLXRandom.normal([hidden.dim(0), config.acousticCodebooks],
                                     key: MLXRandom.key(options.seed &+ UInt64(frame)))
        let bothHidden = concatenated([hidden, MLXArray.zeros(like: hidden)], axis: 0)
        let dt = Float(1) / Float(options.denoisingSteps - 1)
        for step in 0..<(options.denoisingSteps - 1) {
            try Task.checkCancellation()
            let both = velocity(concatenated([latent, latent], axis: 0), time: Float(step) * dt, hidden: bothHidden)
            let batch = hidden.dim(0)
            let conditional = both[0..<batch]
            let unconditional = both[batch...]
            latent = latent + (options.guidance * conditional + (1 - options.guidance) * unconditional) * dt
        }
        return concatenated([semantic.expandedDimensions(axis: -1), Self.acousticCodes(latent, size: config.acousticSize)], axis: -1)
    }
}

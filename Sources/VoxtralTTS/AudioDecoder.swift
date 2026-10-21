import Foundation
import MLX
import MLXNN

final class VoxtralConvWeight: Module {
    @ParameterInfo var original0: MLXArray
    @ParameterInfo var original1: MLXArray

    init(output: Int, input: Int, kernel: Int) {
        _original0.wrappedValue = MLXArray.ones([output, 1, 1])
        _original1.wrappedValue = MLXArray.zeros([output, input, kernel])
        super.init()
    }

    var normalized: MLXArray {
        original0 * original1 / sqrt(sum(original1 * original1, axes: [1, 2], keepDims: true) + 1e-12)
    }
}

final class VoxtralConvParameters: Module {
    @ModuleInfo var weight: VoxtralConvWeight
    init(output: Int, input: Int, kernel: Int) {
        _weight.wrappedValue = VoxtralConvWeight(output: output, input: input, kernel: kernel)
        super.init()
    }
}

final class VoxtralCausalConv: Module {
    @ModuleInfo var parametrizations: VoxtralConvParameters
    let reflect: Bool

    init(output: Int, input: Int, kernel: Int, reflect: Bool = false) {
        self.reflect = reflect
        _parametrizations.wrappedValue = VoxtralConvParameters(output: output, input: input, kernel: kernel)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, stride: Int = 1) -> MLXArray {
        let weight = parametrizations.weight.normalized
        if stride > 1 {
            let result = convTransposed1d(input, weight.transposed(1, 2, 0), stride: stride)
            return result[0..., 0..<(input.dim(1) * stride), 0...]
        }
        let padding = weight.dim(2) - 1
        let indices: [Int32]
        if reflect {
            // Short signals receive zero padding before reflection upstream.
            let extra = max(0, padding - input.dim(1) + 1)
            let extended = extra > 0
                ? concatenated([input, MLXArray.zeros([input.dim(0), extra, input.dim(2)], dtype: input.dtype)], axis: 1)
                : input
            indices = (1...max(1, padding)).reversed().prefix(padding).map(Int32.init)
            let prefix = extended[0..., MLXArray(indices), 0...]
            let result = conv1d(concatenated([prefix, extended], axis: 1), weight.transposed(0, 2, 1))
            return result[0..., 0..<input.dim(1), 0...]
        }
        let prefix = broadcast(input[0..., 0..<1, 0...], to: [input.dim(0), padding, input.dim(2)])
        return conv1d(concatenated([prefix, input], axis: 1), weight.transposed(0, 2, 1))
    }

    /// History belongs to one utterance. Transposed convolutions carry their
    /// overlapping tail into the next chunk; the utterance's final tail is trimmed.
    func stream(_ input: MLXArray, stride: Int, history: inout MLXArray?) -> MLXArray {
        let weight = parametrizations.weight.normalized
        if stride > 1 {
            var output = convTransposed1d(input, weight.transposed(1, 2, 0), stride: stride)
            if let previous = history {
                output = output + concatenated([previous,
                    MLXArray.zeros([output.dim(0), output.dim(1) - previous.dim(1), output.dim(2)], dtype: output.dtype)], axis: 1)
            }
            let length = input.dim(1) * stride
            history = output[0..., length..., 0...]
            eval(history!)
            return output[0..., 0..<length, 0...]
        }
        let padding = weight.dim(2) - 1
        let prefix: MLXArray
        if let previous = history {
            prefix = previous
        } else if reflect {
            let extra = max(0, padding - input.dim(1) + 1)
            let extended = extra > 0
                ? concatenated([input, MLXArray.zeros([input.dim(0), extra, input.dim(2)], dtype: input.dtype)], axis: 1)
                : input
            let indices = (1...max(1, padding)).reversed().prefix(padding).map(Int32.init)
            prefix = extended[0..., MLXArray(indices), 0...]
        } else {
            prefix = broadcast(input[0..., 0..<1, 0...], to: [input.dim(0), padding, input.dim(2)])
        }
        let padded = concatenated([prefix, input], axis: 1)
        history = padded[0..., (padded.dim(1) - padding)..., 0...]
        eval(history!)
        return conv1d(padded, weight.transposed(0, 2, 1))
    }
}

final class VoxtralConvBlock: Module {
    @ModuleInfo var conv: VoxtralCausalConv
    init(output: Int, input: Int, kernel: Int, reflect: Bool = false) {
        _conv.wrappedValue = VoxtralCausalConv(output: output, input: input, kernel: kernel, reflect: reflect)
        super.init()
    }
}

final class VoxtralCodecAttention: Module {
    let config: VoxtralCodecConfig
    @ModuleInfo var wq: Linear
    @ModuleInfo var wk: Linear
    @ModuleInfo var wv: Linear
    @ModuleInfo var wo: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    init(_ config: VoxtralCodecConfig) {
        self.config = config
        _wq.wrappedValue = Linear(config.dim, config.heads * config.headDim, bias: false)
        _wk.wrappedValue = Linear(config.dim, config.kvHeads * config.headDim, bias: false)
        _wv.wrappedValue = Linear(config.dim, config.kvHeads * config.headDim, bias: false)
        _wo.wrappedValue = Linear(config.heads * config.headDim, config.dim, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: config.heads * config.headDim, eps: config.qkNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: config.kvHeads * config.headDim, eps: config.qkNormEps)
        super.init()
    }

    static func alibiSlopes(heads: Int) -> [Float] {
        func powerOfTwo(_ n: Int) -> [Float] {
            let start = pow(Float(2), -pow(Float(2), -(log2(Float(n)) - 3)))
            return (0..<n).map { start * pow(start, Float($0)) }
        }
        let closest = 1 << Int(floor(log2(Double(heads))))
        if closest == heads { return powerOfTwo(heads) }
        let additional = powerOfTwo(2 * closest).enumerated().filter { $0.offset % 2 == 0 }.map(\.element)
        return powerOfTwo(closest) + Array(additional.prefix(heads - closest))
    }

    func callAsFunction(_ input: MLXArray, window: Int) throws -> MLXArray {
        var cache: (MLXArray, MLXArray)?
        return try attend(input, window: window, cache: &cache, retainCache: false)
    }

    func stream(_ input: MLXArray, window: Int, cache: inout (MLXArray, MLXArray)?) throws -> MLXArray {
        try attend(input, window: window, cache: &cache, retainCache: true)
    }

    private func attend(_ input: MLXArray, window: Int, cache: inout (MLXArray, MLXArray)?, retainCache: Bool) throws -> MLXArray {
        let b = input.dim(0), t = input.dim(1), h = config.heads, d = config.headDim
        let q = queryNorm(wq(input)).reshaped(b, t, h, d).transposed(0, 2, 1, 3)
        let newK = keyNorm(wk(input)).reshaped(b, t, h, d).transposed(0, 2, 1, 3)
        let newV = wv(input).reshaped(b, t, h, d).transposed(0, 2, 1, 3)
        let previous = cache?.0.dim(2) ?? 0
        let k = cache.map { concatenated([$0.0, newK], axis: 2) } ?? newK
        let v = cache.map { concatenated([$0.1, newV], axis: 2) } ?? newV
        let slopes = MLXArray(Self.alibiSlopes(heads: h)).reshaped(h, 1, 1)
        eval(q, k, v)
        var pieces: [MLXArray] = []
        // Attention sees only the local causal window. Tiling bounds memory
        // independently of utterance duration and preserves the full decoder math.
        for start in Swift.stride(from: 0, to: t, by: 128) {
            try Task.checkCancellation()
            let end = min(start + 128, t), keyStart = max(0, previous + start - window)
            let queries = q[0..., 0..., start..<end, 0...]
            let keys = k[0..., 0..., keyStart..<(previous + end), 0...]
            let values = v[0..., 0..., keyStart..<(previous + end), 0...]
            let distance = MLXArray(keyStart..<(previous + end)).expandedDimensions(axis: 0)
                - MLXArray((previous + start)..<(previous + end)).expandedDimensions(axis: 1)
            let bias = slopes * distance.asType(.float32).expandedDimensions(axis: 0)
                + MLX.where((distance .> 0) .|| (distance .< -window), Float(-1e9), Float(0))
            let scores = matmul(queries, keys.transposed(0, 1, 3, 2)) * (1 / sqrt(Float(d))) + bias
            let probabilities = softmax(scores.asType(.float32), axis: -1).asType(input.dtype)
            let piece = matmul(probabilities, values)
            eval(piece)
            pieces.append(piece)
        }
        if retainCache {
            let keepFrom = max(0, k.dim(2) - window)
            cache = (k[0..., 0..., keepFrom..., 0...], v[0..., 0..., keepFrom..., 0...])
            eval(cache!.0, cache!.1)
        }
        return wo(concatenated(pieces, axis: 2).transposed(0, 2, 1, 3).reshaped(b, t, h * d))
    }
}

final class VoxtralCodecLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo var attention: VoxtralCodecAttention
    @ModuleInfo(key: "feed_forward") var feedForward: VoxtralFeedForward
    @ParameterInfo(key: "attention_scale") var attentionScale: MLXArray
    @ParameterInfo(key: "ffn_scale") var ffnScale: MLXArray

    init(_ config: VoxtralCodecConfig) {
        _attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _attention.wrappedValue = VoxtralCodecAttention(config)
        _feedForward.wrappedValue = VoxtralFeedForward(dim: config.dim, hiddenDim: config.hiddenDim)
        _attentionScale.wrappedValue = MLXArray.ones([config.dim])
        _ffnScale.wrappedValue = MLXArray.ones([config.dim])
        super.init()
    }

    func callAsFunction(_ input: MLXArray, window: Int) throws -> MLXArray {
        let hidden = try input + attention(attentionNorm(input), window: window) * attentionScale
        return hidden + feedForward(ffnNorm(hidden)) * ffnScale
    }

    func stream(_ input: MLXArray, window: Int, cache: inout (MLXArray, MLXArray)?) throws -> MLXArray {
        let hidden = try input + attention.stream(attentionNorm(input), window: window, cache: &cache) * attentionScale
        return hidden + feedForward(ffnNorm(hidden)) * ffnScale
    }
}

final class VoxtralCodecTransformer: Module {
    @ModuleInfo var layers: [VoxtralCodecLayer]
    init(_ config: VoxtralCodecConfig, layers: Int) {
        _layers.wrappedValue = (0..<layers).map { _ in VoxtralCodecLayer(config) }
        super.init()
    }
    func callAsFunction(_ input: MLXArray, window: Int) throws -> MLXArray {
        var hidden = input
        for layer in layers {
            try Task.checkCancellation()
            hidden = try layer(hidden, window: window)
            eval(hidden)
        }
        return hidden
    }

    func stream(_ input: MLXArray, window: Int, caches: inout [(MLXArray, MLXArray)?]) throws -> MLXArray {
        var hidden = input
        for index in layers.indices {
            try Task.checkCancellation()
            hidden = try layers[index].stream(hidden, window: window, cache: &caches[index])
            eval(hidden)
        }
        return hidden
    }
}

final class VoxtralSemanticCodebook: Module {
    @ParameterInfo(key: "cluster_usage") var clusterUsage: MLXArray
    @ParameterInfo(key: "embedding_sum") var embeddingSum: MLXArray
    init(_ config: VoxtralCodecConfig) {
        _clusterUsage.wrappedValue = MLXArray.ones([config.semanticSize])
        _embeddingSum.wrappedValue = MLXArray.zeros([config.semanticSize, config.semanticDim])
        super.init()
    }
    func callAsFunction(_ indices: MLXArray) -> MLXArray {
        let table = embeddingSum.asType(.float32) / maximum(clusterUsage.asType(.float32).expandedDimensions(axis: -1), 1e-8)
        return table[indices]
    }
}

final class VoxtralCodebook: Module {
    let config: VoxtralCodecConfig
    @ModuleInfo(key: "semantic_codebook") var semantic: VoxtralSemanticCodebook
    init(_ config: VoxtralCodecConfig) {
        self.config = config
        _semantic.wrappedValue = VoxtralSemanticCodebook(config)
        super.init()
    }
    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        let semanticEmbedding = semantic(codes[0..., 0..., 0] - 2)
        let acousticEmbedding = 2 * (codes[0..., 0..., 1...] - 2).asType(.float32) / Float(config.acousticSize - 1) - 1
        return concatenated([semanticEmbedding, acousticEmbedding], axis: -1)
    }
}

final class VoxtralAudioDecoder: Module {
    struct State {
        var convolutions: [MLXArray?] = Array(repeating: nil, count: 4)
        var attention: [[(MLXArray, MLXArray)?]]
        var output: MLXArray?

        init(_ config: VoxtralCodecConfig) {
            attention = config.lengths.map { Array(repeating: nil, count: $0) }
        }
    }
    let config: VoxtralCodecConfig
    @ModuleInfo var quantizer: VoxtralCodebook
    @ModuleInfo(key: "decoder_blocks") var blocks: [Module]
    @ModuleInfo(key: "output_proj") var outputProjection: VoxtralConvBlock

    init(_ config: VoxtralCodecConfig) {
        self.config = config
        _quantizer.wrappedValue = VoxtralCodebook(config)
        var blocks: [Module] = []
        for stage in 0..<4 {
            blocks.append(VoxtralConvBlock(output: config.dim,
                input: stage == 0 ? config.semanticDim + config.acousticDim : config.dim,
                kernel: config.kernels[stage]))
            blocks.append(VoxtralCodecTransformer(config, layers: config.lengths[stage]))
        }
        _blocks.wrappedValue = blocks
        _outputProjection.wrappedValue = VoxtralConvBlock(output: config.patchSize, input: config.dim,
                                                         kernel: config.patchKernel, reflect: true)
        super.init()
    }

    func decode(_ codes: MLXArray) throws -> MLXArray {
        var hidden = quantizer(codes)
        for stage in 0..<4 {
            try Task.checkCancellation()
            let conv = blocks[stage * 2] as! VoxtralConvBlock
            let transformer = blocks[stage * 2 + 1] as! VoxtralCodecTransformer
            hidden = conv.conv(hidden, stride: config.strides[stage])
            hidden = try transformer(hidden, window: 2 << stage)
        }
        return outputProjection.conv(hidden).reshaped(codes.dim(0), -1)
    }

    func decode(_ codes: MLXArray, state: inout State) throws -> MLXArray {
        var hidden = quantizer(codes)
        for stage in 0..<4 {
            try Task.checkCancellation()
            let conv = blocks[stage * 2] as! VoxtralConvBlock
            let transformer = blocks[stage * 2 + 1] as! VoxtralCodecTransformer
            hidden = conv.conv.stream(hidden, stride: config.strides[stage], history: &state.convolutions[stage])
            hidden = try transformer.stream(hidden, window: 2 << stage, caches: &state.attention[stage])
        }
        return outputProjection.conv.stream(hidden, stride: 1, history: &state.output).reshaped(codes.dim(0), -1)
    }
}

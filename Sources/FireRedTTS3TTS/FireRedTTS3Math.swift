import Foundation
import MLX
import MLXFFT
import MLXFast
import MLXCommon
import MLXNN

/// Original checkpoint layout is retained, including all convolution axes.
final class FireRedTTS3Network {
    let config: FireRedTTS3Config
    let weights: [String: MLXArray]
    let weightBytes: Int

    init(config: FireRedTTS3Config, weights: [String: MLXArray]) throws {
        self.config = config
        self.weights = weights
        let expected = Self.tensorShapes(config)
        guard Set(weights.keys) == Set(expected.keys) else {
            let missing = Set(expected.keys).subtracting(weights.keys).sorted().prefix(3)
            let extra = Set(weights.keys).subtracting(expected.keys).sorted().prefix(3)
            throw FireRedTTS3Error.invalidWeights("missing \(missing), unexpected \(extra)")
        }
        for (key, shape) in expected {
            guard weights[key]!.shape == shape, weights[key]!.dtype == .float32 else {
                throw FireRedTTS3Error.invalidWeights("\(key): expected Float32 \(shape), got \(weights[key]!.shape)")
            }
        }
        weightBytes = weights.values.reduce(0) { $0 + $1.nbytes }
    }

    func linear(_ x: MLXArray, _ prefix: String) -> MLXArray {
        var y = matmul(x, weights[prefix + ".weight"]!.T)
        if let b = weights[prefix + ".bias"] { y = y + b }
        return y
    }

    func rms(_ x: MLXArray, _ prefix: String, eps: Float = 1e-6) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weights[prefix + ".weight"]!, eps: eps)
    }

    func qwen(_ input: MLXArray, prefix: String, config q: FireRedTTS3QwenConfig,
              cache: inout [(MLXArray, MLXArray)?], offset: Int) throws -> MLXArray {
        var h = input
        let batch = h.dim(0), length = h.dim(1)
        for layer in 0..<q.numHiddenLayers {
            try Task.checkCancellation()
            let p = "\(prefix).layers.\(layer)"
            let normalized = rms(h, p + ".input_layernorm", eps: q.rmsNormEps)
            var query = linear(normalized, p + ".self_attn.q_proj").reshaped(batch, length, q.numAttentionHeads, q.headDim)
            var key = linear(normalized, p + ".self_attn.k_proj").reshaped(batch, length, q.numKeyValueHeads, q.headDim)
            query = rms(query, p + ".self_attn.q_norm", eps: q.rmsNormEps).transposed(0, 2, 1, 3)
            key = rms(key, p + ".self_attn.k_norm", eps: q.rmsNormEps).transposed(0, 2, 1, 3)
            let value = linear(normalized, p + ".self_attn.v_proj")
                .reshaped(batch, length, q.numKeyValueHeads, q.headDim).transposed(0, 2, 1, 3)
            query = MLXFast.RoPE(query, dimensions: q.headDim, traditional: false, base: q.ropeTheta, scale: 1, offset: offset)
            key = MLXFast.RoPE(key, dimensions: q.headDim, traditional: false, base: q.ropeTheta, scale: 1, offset: offset)
            var keys = key, values = value
            if let old = cache[layer] {
                keys = concatenated([old.0, key], axis: 2)
                values = concatenated([old.1, value], axis: 2)
            }
            cache[layer] = (keys, values)
            let keyPositions = MLXArray(0..<Int32(keys.dim(2))).reshaped([1, 1, 1, keys.dim(2)])
            let queryPositions = MLXArray(Int32(offset)..<Int32(offset + length)).reshaped([1, 1, length, 1])
            var allowed = keyPositions .<= queryPositions
            if q.slidingWindow > 0 { allowed = logicalAnd(allowed, keyPositions .> (queryPositions - q.slidingWindow)) }
            let mask = MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity))
            let attended = SDPA.attendAndMerge(qHeads: query, kHeads: keys, vHeads: values,
                scale: 1 / sqrt(Float(q.headDim)), mask: mask)
            h = h + linear(attended, p + ".self_attn.o_proj")
            let ff = rms(h, p + ".post_attention_layernorm", eps: q.rmsNormEps)
            h = h + linear(silu(linear(ff, p + ".mlp.gate_proj")) * linear(ff, p + ".mlp.up_proj"), p + ".mlp.down_proj")
        }
        return rms(h, prefix + ".norm", eps: q.rmsNormEps)
    }

    func backbone(_ input: MLXArray, cache: inout [(MLXArray, MLXArray)?], offset: Int) throws -> MLXArray {
        try qwen(input, prefix: "backbone_llm.model", config: config.backbone, cache: &cache, offset: offset)
    }

    func embedding(_ tokens: [Int]) -> MLXArray {
        weights["backbone_llm.model.embed_tokens.weight"]![MLXArray(tokens.map(Int32.init))].expandedDimensions(axis: 0)
    }

    func textLogits(_ hidden: MLXArray) -> MLXArray {
        matmul(hidden, weights["backbone_llm.model.embed_tokens.weight"]!.T)
    }

    func attention(_ x: MLXArray, _ prefix: String, heads: Int) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1), dim = x.dim(2) / heads
        var q = linear(x, prefix + ".to_q").reshaped(batch, length, heads, dim).transposed(0, 2, 1, 3)
        var k = linear(x, prefix + ".to_k").reshaped(batch, length, heads, dim).transposed(0, 2, 1, 3)
        let v = linear(x, prefix + ".to_v").reshaped(batch, length, heads, dim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: dim, traditional: true, base: 10000, scale: 1, offset: 0)
        k = MLXFast.RoPE(k, dimensions: dim, traditional: true, base: 10000, scale: 1, offset: 0)
        return linear(SDPA.attendAndMerge(qHeads: q, kHeads: k, vHeads: v,
            scale: 1 / sqrt(Float(dim))), prefix + ".to_out.0")
    }

    func feedForward(_ x: MLXArray, _ prefix: String) -> MLXArray {
        linear(geluApproximate(linear(x, prefix + ".ff.0.0")), prefix + ".ff.2")
    }

    func encodePatch(_ latents: MLXArray) -> MLXArray {
        let p = "patch_encoder", c = config.patchEncoder
        let x = linear(latents, p + ".in_proj").reshaped([-1, config.patchSize, c.hiddenSize])
        let cls = MLX.broadcast(weights[p + ".cls_tok"]!, to: [x.dim(0), 1, c.hiddenSize])
        var h = concatenated([cls, x], axis: 1)
        for i in 0..<c.numHiddenLayers {
            let layer = p + ".blocks.\(i)"
            h = h + attention(rms(h, layer + ".norm1"), layer + ".attn", heads: c.numAttentionHeads)
            h = h + feedForward(rms(h, layer + ".norm2"), layer + ".mlp")
        }
        return linear(rms(h, p + ".out_proj.norm_final"), p + ".out_proj.linear")[0..., 0, 0...]
            .reshaped([1, -1, config.backbone.hiddenSize])
    }

    func flow(_ input: MLXArray, time: MLXArray) -> MLXArray {
        let p = "dit", c = config.flow
        let half = 128
        let frequency = exp(MLXArray(0..<Int32(half)).asType(.float32) * (-Float(log(10000.0)) / Float(half - 1)))
        let phase = time.reshaped([-1, 1]) * 1000 * frequency.reshaped([1, half])
        let sinusoidal = concatenated([sin(phase), cos(phase)], axis: -1)
        let condition = linear(silu(linear(sinusoidal, p + ".t_embedder.time_mlp.0")), p + ".t_embedder.time_mlp.2")
            .expandedDimensions(axis: 1)
        var h = linear(input, p + ".in_proj")
        for i in 0..<c.numHiddenLayers {
            let layer = p + ".blocks.\(i)"
            let a = split(linear(silu(condition), layer + ".adaLN_modulation.1"), parts: 9, axis: -1)
            h = h + a[2] * attention(rms(h, layer + ".norm1") * (1 + a[1]) + a[0], layer + ".attn", heads: c.numAttentionHeads)
            var conv = rms(h, layer + ".norm2") * (1 + a[7]) + a[6]
            conv = conv1d(conv, layer + ".conv.block.0")
            let softplus = maximum(conv, MLXArray(Float(0))) + log1p(exp(-abs(conv)))
            conv = conv * tanh(softplus)
            conv = conv1d(conv, layer + ".conv.block.2")
            h = h + a[8] * conv
            h = h + a[5] * feedForward(rms(h, layer + ".norm3") * (1 + a[4]) + a[3], layer + ".mlp")
        }
        let modulation = split(linear(silu(condition), p + ".final_layer.adaLN_modulation.1"), parts: 2, axis: -1)
        let mean = h.mean(axis: -1, keepDims: true)
        let centered = h - mean
        h = centered * rsqrt((centered * centered).mean(axis: -1, keepDims: true) + 1e-6)
        return linear(h * (1 + modulation[1]) + modulation[0], p + ".final_layer.linear")
    }

    private func conv1d(_ x: MLXArray, _ prefix: String) -> MLXArray {
        MLX.conv1d(x, weights[prefix + ".weight"]!.transposed(0, 2, 1), padding: 1) + weights[prefix + ".bias"]!
    }

    func decode(_ latents: MLXArray) throws -> MLXArray {
        let projected = linear(latents / config.latentScale, "decoder.in_proj")
            .reshaped([1, -1, config.decoder.hiddenSize])
        var cache = Array<(MLXArray, MLXArray)?>(repeating: nil, count: config.decoder.numHiddenLayers)
        let h = try qwen(projected, prefix: "decoder.qwen3", config: config.decoder, cache: &cache, offset: 0)
        let spectral = split(linear(h, "decoder.istft_head.out"), parts: 2, axis: -1)
        let magnitude = minimum(exp(spectral[0]), MLXArray(Float(100)))
        let spectrum = magnitude * cos(spectral[1]) + (magnitude * sin(spectral[1])).asImaginary()
        return Self.istft(spectrum, window: weights["decoder.istft_head.istft.window"]!, hop: config.audioPatchSize)
    }

    static func istft(_ spectrum: MLXArray, window: MLXArray, hop: Int) -> MLXArray {
        let nFFT = window.size, subdivisions = nFFT / hop
        let count = spectrum.dim(1), batch = spectrum.dim(0)
        let frames = MLXFFT.irfft(spectrum, n: nFFT, axis: -1) * window.reshaped([1, 1, nFFT])
        let grouped = frames.reshaped([batch, count, subdivisions, hop])
        let squared = (window * window).reshaped([subdivisions, hop])
        var samples: MLXArray?, envelope: MLXArray?
        for i in 0..<subdivisions {
            let chunk = MLX.padded(grouped[0..., 0..., i, 0...], widths: [.init(0), .init((i, subdivisions - 1 - i)), .init(0)])
            samples = samples.map { $0 + chunk } ?? chunk
            let repeated = MLX.broadcast(squared[i, 0...].reshaped([1, hop]), to: [count, hop])
            let padded = MLX.padded(repeated, widths: [.init((i, subdivisions - 1 - i)), .init(0)])
            envelope = envelope.map { $0 + padded } ?? padded
        }
        let total = (count + subdivisions - 1) * hop, pad = (nFFT - hop) / 2
        let audio = samples!.reshaped([batch, total])[0..., pad..<(total-pad)]
        let denominator = envelope!.reshaped([total])[pad..<(total-pad)]
        return audio / maximum(denominator, MLXArray(Float(1e-11)))
    }

    static func tensorShapes(_ c: FireRedTTS3Config) -> [String: [Int]] {
        var result = [String: [Int]]()
        func linear(_ prefix: String, _ output: Int, _ input: Int, bias: Bool = true) {
            result[prefix + ".weight"] = [output, input]
            if bias { result[prefix + ".bias"] = [output] }
        }
        func norm(_ prefix: String, _ size: Int) { result[prefix + ".weight"] = [size] }
        func qwen(_ prefix: String, _ q: FireRedTTS3QwenConfig) {
            norm(prefix + ".norm", q.hiddenSize)
            for i in 0..<q.numHiddenLayers {
                let p = prefix + ".layers.\(i)"
                norm(p + ".input_layernorm", q.hiddenSize);norm(p + ".post_attention_layernorm", q.hiddenSize)
                for (name, heads) in [("q",q.numAttentionHeads),("k",q.numKeyValueHeads),("v",q.numKeyValueHeads)] {
                    linear(p + ".self_attn.\(name)_proj", heads*q.headDim, q.hiddenSize, bias: false)
                }
                norm(p + ".self_attn.q_norm", q.headDim);norm(p + ".self_attn.k_norm", q.headDim)
                linear(p + ".self_attn.o_proj", q.hiddenSize, q.numAttentionHeads*q.headDim, bias: false)
                linear(p + ".mlp.gate_proj", q.intermediateSize, q.hiddenSize, bias: false)
                linear(p + ".mlp.up_proj", q.intermediateSize, q.hiddenSize, bias: false)
                linear(p + ".mlp.down_proj", q.hiddenSize, q.intermediateSize, bias: false)
            }
        }
        func block(_ prefix: String, _ t: FireRedTTS3TransformerConfig, flow: Bool) {
            norm(prefix + ".norm1",t.hiddenSize);norm(prefix + ".norm2",t.hiddenSize)
            if flow { norm(prefix + ".norm3",t.hiddenSize) }
            for name in ["to_q","to_k","to_v","to_out.0"] { linear(prefix + ".attn.\(name)",t.hiddenSize,t.hiddenSize) }
            linear(prefix + ".mlp.ff.0.0",t.intermediateSize,t.hiddenSize)
            linear(prefix + ".mlp.ff.2",t.hiddenSize,t.intermediateSize)
            if flow {
                linear(prefix + ".adaLN_modulation.1",9*t.hiddenSize,t.hiddenSize)
                for i in [0,2] {
                    result[prefix + ".conv.block.\(i).weight"] = [t.hiddenSize,t.hiddenSize,3]
                    result[prefix + ".conv.block.\(i).bias"] = [t.hiddenSize]
                }
            }
        }
        qwen("backbone_llm.model", c.backbone)
        result["backbone_llm.model.embed_tokens.weight"] = [c.backbone.vocabSize,c.backbone.hiddenSize]
        linear("dit_head",c.flow.hiddenSize,c.backbone.hiddenSize);linear("stop_head",1,c.backbone.hiddenSize)
        let p = c.patchEncoder, f = c.flow
        result["patch_encoder.cls_tok"] = [1,1,p.hiddenSize]
        linear("patch_encoder.in_proj",p.hiddenSize,c.latentDimension)
        norm("patch_encoder.out_proj.norm_final",p.hiddenSize)
        linear("patch_encoder.out_proj.linear",c.backbone.hiddenSize,p.hiddenSize)
        for i in 0..<p.numHiddenLayers { block("patch_encoder.blocks.\(i)",p,flow:false) }
        linear("dit.in_proj",f.hiddenSize,c.latentDimension+f.hiddenSize)
        linear("dit.t_embedder.time_mlp.0",f.hiddenSize,256);linear("dit.t_embedder.time_mlp.2",f.hiddenSize,f.hiddenSize)
        for i in 0..<f.numHiddenLayers { block("dit.blocks.\(i)",f,flow:true) }
        linear("dit.final_layer.adaLN_modulation.1",2*f.hiddenSize,f.hiddenSize)
        linear("dit.final_layer.linear",c.latentDimension,f.hiddenSize)
        qwen("decoder.qwen3",c.decoder)
        linear("decoder.in_proj",c.decoderUpsampleRate*c.decoder.hiddenSize,c.latentDimension)
        linear("decoder.istft_head.out",c.audioPatchSize*4+2,c.decoder.hiddenSize)
        result["decoder.istft_head.istft.window"] = [c.audioPatchSize*4]
        return result
    }
}

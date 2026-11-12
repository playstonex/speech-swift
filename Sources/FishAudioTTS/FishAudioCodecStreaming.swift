import MLX
import MLXCommon
import MLXNN

/// Caches belong to one decoder invocation, never to the shared codec weights.
struct FishAudioCodecStreamingState {
    struct Attention {
        var keys: MLXArray
        var values: MLXArray
        var offset: Int
    }
    var convolutions: [ObjectIdentifier: MLXArray] = [:]
    var attention: [ObjectIdentifier: Attention] = [:]

    var arrays: [MLXArray] {
        Array(convolutions.values) + attention.values.flatMap { [$0.keys, $0.values] }
    }
}

extension FishCodecConv1dNCL {
    func streaming(_ x: MLXArray, state: inout FishAudioCodecStreamingState) -> MLXArray {
        // All decoder convolutions have unit stride; encoder downsampling is separate.
        precondition(causal && stride == 1)
        let context = effectiveKernelSize - 1
        guard context > 0 else { return conv(x.transposed(0, 2, 1)).transposed(0, 2, 1) }
        let id = ObjectIdentifier(self)
        let past = state.convolutions[id] ?? MLXArray.zeros([x.dim(0), x.dim(1), context], dtype: x.dtype)
        let joined = concatenated([past, x], axis: 2)
        state.convolutions[id] = joined[0..., 0..., (joined.dim(2) - context)...]
        return conv(joined.transposed(0, 2, 1)).transposed(0, 2, 1)
    }
}

extension FishCodecConvTransposed1dNCL {
    func streaming(_ x: MLXArray, state: inout FishAudioCodecStreamingState) -> MLXArray {
        var output = conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        guard trimRight > 0 else { return output }
        let id = ObjectIdentifier(self)
        if let past = state.convolutions[id] {
            output = concatenated([output[0..., 0..., 0..<trimRight] + past,
                                   output[0..., 0..., trimRight...]], axis: 2)
        }
        let end = output.dim(2) - trimRight
        var tail = output[0..., 0..., end...]
        // The next chunk already contains bias. Carry only the convolution contribution.
        if let bias = conv.bias { tail = tail - bias.reshaped(1, -1, 1) }
        state.convolutions[id] = tail
        return output[0..., 0..., 0..<end]
    }
}

extension FishCodecTransformerAttention {
    func streaming(_ x: MLXArray, window: Int, state: inout FishAudioCodecStreamingState) -> MLXArray {
        let id = ObjectIdentifier(self)
        let past = state.attention[id]
        let offset = past?.offset ?? 0
        let count = x.dim(1), hidden = numHeads * headDim
        let parts = split(wqkv(x), indices: [hidden, hidden * 2], axis: -1)
        let q = rope(parts[0].reshaped(x.dim(0), count, numHeads, headDim).transposed(0, 2, 1, 3), offset: offset)
        var keys = rope(parts[1].reshaped(x.dim(0), count, numHeads, headDim).transposed(0, 2, 1, 3), offset: offset)
        var values = parts[2].reshaped(x.dim(0), count, numHeads, headDim).transposed(0, 2, 1, 3)
        if let past {
            keys = concatenated([past.keys, keys], axis: 2)
            values = concatenated([past.values, values], axis: 2)
        }
        let firstKey = offset + count - keys.dim(2)
        let rows = MLXArray(Int32(offset)..<Int32(offset + count)).expandedDimensions(axis: 1)
        let cols = MLXArray(Int32(firstKey)..<Int32(offset + count)).expandedDimensions(axis: 0)
        let allowed = (cols .<= rows) .&& (cols .>= (rows - Int32(window - 1)))
        let mask = MLX.where(allowed, MLXArray(Float(0)), MLXArray(Float(-1e9)))
            .reshaped(1, 1, count, keys.dim(2)).asType(x.dtype)
        let result = wo(SDPA.attendAndMerge(qHeads: q, kHeads: keys, vHeads: values, scale: scale, mask: mask))
        let keep = min(window - 1, keys.dim(2)), start = keys.dim(2) - keep
        state.attention[id] = .init(keys: keys[0..., 0..., start..., 0...],
                                    values: values[0..., 0..., start..., 0...], offset: offset + count)
        return result
    }
}

extension FishCodecPostTransformer {
    func streaming(_ x: MLXArray, state: inout FishAudioCodecStreamingState) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        for layer in layers {
            h = h + layer.attentionLayerScale(layer.attention.streaming(layer.attentionNorm(h), window: windowSize, state: &state))
            h = h + layer.ffnLayerScale(layer.feedForward(layer.ffnNorm(h)))
        }
        return norm(h).transposed(0, 2, 1)
    }
}

extension FishCodecConvNeXtBlock {
    func streaming(_ x: MLXArray, state: inout FishAudioCodecStreamingState) -> MLXArray {
        var h = norm(dwConv.streaming(x, state: &state).transposed(0, 2, 1))
        h = gamma(pwConv2(gelu(pwConv1(h)))).transposed(0, 2, 1)
        return x + h
    }
}

extension FishCodecResidualUnit {
    func streaming(_ x: MLXArray, state: inout FishAudioCodecStreamingState) -> MLXArray {
        let h = conv1.streaming(snake1(x), state: &state)
        return x + conv2.streaming(snake2(h), state: &state)
    }
}

extension FishAudioCodec {
    /// Decode only new frames. The final transposed-convolution tail is trimmed,
    /// matching full decoding; no additional flush samples belong to the utterance.
    func decodeStreaming(_ generated: FishAudioGeneratedCodebooks,
                         state: inout FishAudioCodecStreamingState) throws -> [Float] {
        let count = generated.frameCount
        guard generated.codebookCount == FishAudioCodecDefaults.totalCodebooks,
              generated.codes.allSatisfy({ $0.count == count }),
              generated.codes.allSatisfy({ row in
                  row.allSatisfy { $0 >= 0 && $0 < FishAudioCodecDefaults.semanticCodebookSize }
              }) else { throw FishAudioError.invalidCodebookShape("invalid streaming codec frames") }
        guard count > 0 else { return [] }
        let codes = MLXArray(generated.codes.flatMap { $0.map(Int32.init) }).reshaped(1, generated.codebookCount, count)
        // Tower heads use 4096 values; residual quantizers clamp to their codec
        // codebooks, matching buffered decoding.
        var h = quantizer.semanticQuantizer.decode(codes[0..., 0..<1, 0...])
            + quantizer.quantizer.decode(codes[0..., 1..., 0...])
        h = quantizer.postModule.streaming(h, state: &state)
        for block in quantizer.upsample {
            h = block.convNext.streaming(block.conv.streaming(h, state: &state), state: &state)
        }
        h = decoder.inputConv.streaming(h, state: &state)
        for block in decoder.blocks {
            h = block.convTranspose.streaming(block.snake(h), state: &state)
            h = block.residual1.streaming(h, state: &state)
            h = block.residual2.streaming(h, state: &state)
            h = block.residual3.streaming(h, state: &state)
        }
        let wav = tanh(decoder.finalConv.streaming(decoder.finalSnake(h), state: &state)).reshaped(-1).asType(.float32)
        eval([wav] + state.arrays)
        try Task.checkCancellation()
        let samples = wav.asArray(Float.self)
        guard samples.count == count * FishAudioCodecDefaults.samplesPerFrame,
              samples.allSatisfy(\.isFinite) else {
            throw FishAudioError.invalidInput("Fish DAC decoded invalid streaming PCM")
        }
        return samples
    }
}

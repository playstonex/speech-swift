import MLX
import MLXNN
import PersonaPlex

/// Keeps the absolute RoPE position while retaining only the recent context.
final class KyutaiSTTCache: KVCache {
    let capacity: Int
    private(set) var offset = 0
    private(set) var keysArray: MLXArray?
    private(set) var valuesArray: MLXArray?

    init(capacity: Int) { self.capacity = capacity }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let k = keysArray.map { concatenated([$0, keys], axis: 2) } ?? keys
        let v = valuesArray.map { concatenated([$0, values], axis: 2) } ?? values
        offset += keys.shape[2]
        let start = max(0, k.shape[2] - capacity)
        keysArray = k[0..., 0..., start..., 0...]
        valuesArray = v[0..., 0..., start..., 0...]
        return (keysArray!, valuesArray!)
    }

    func replaceArrays(keys: MLXArray, values: MLXArray) {
        keysArray = keys
        valuesArray = values
    }

    func trim(_ count: Int) {
        if count >= offset {
            keysArray = nil
            valuesArray = nil
            offset = 0
        }
    }
}

final class KyutaiSTTDecoder: Module {
    @ModuleInfo var layers: [TemporalTransformerLayer]
    @ModuleInfo var out_norm: RMSNormF32
    @ModuleInfo var text_emb: Embedding
    @ModuleInfo var emb: [Embedding]
    @ModuleInfo var text_linear: Linear

    init(config: KyutaiSTTConfig) {
        _layers.wrappedValue = (0..<config.numLayers).map { _ in
            TemporalTransformerLayer(cfg: config.transformerConfig)
        }
        _out_norm.wrappedValue = RMSNormF32(dimensions: config.dim)
        _text_emb.wrappedValue = Embedding(embeddingCount: config.textCard + 1, dimensions: config.dim)
        _emb.wrappedValue = (0..<config.numCodebooks).map { _ in
            Embedding(embeddingCount: config.card + 1, dimensions: config.dim)
        }
        _text_linear.wrappedValue = Linear(config.dim, config.textCard, bias: false)
        super.init()
    }

    func step(text: Int, audio: MLXArray, cache: [KyutaiSTTCache], position: Int) -> MLXArray {
        var hidden = text_emb(MLXArray([Int32(text)]).reshaped(1, 1))
        for i in emb.indices {
            hidden = hidden + emb[i](audio[0..<1, i, 0..<1])
        }
        for (layer, c) in zip(layers, cache) {
            hidden = layer(hidden, cache: c, offset: position)
        }
        return text_linear(out_norm(hidden))
    }

    static func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var output: [String: MLXArray] = [:]
        for (rawKey, rawValue) in weights {
            var key = rawKey.hasPrefix("transformer.") ? String(rawKey.dropFirst(12)) : rawKey
            key = key.replacingOccurrences(of: ".in_proj_weight", with: ".in_proj.weight")
            var value = rawValue
            if key.hasSuffix(".alpha") {
                key = String(key.dropLast(6)) + ".weight"
                value = value.reshaped(-1)
            }
            guard output[key] == nil else {
                throw KyutaiSTTError.incompatibleCheckpoint("duplicate tensor: \(key)")
            }
            output[key] = value
        }
        return output
    }
}

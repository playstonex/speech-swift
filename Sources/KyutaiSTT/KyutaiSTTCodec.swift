import MLX
import MLXNN
import PersonaPlex

/// Recognition needs the encoder, transformer, downsampler and codebooks only.
/// Keeping the audio decoder out of the module avoids loading its unused weights.
final class KyutaiSTTCodec: Module {
    let cfg: MimiConfig
    @ModuleInfo var encoder: SeanetEncoder
    @ModuleInfo var encoder_transformer: ProjectedTransformer
    @ModuleInfo var downsample: ConvDownsample1d
    @ModuleInfo var quantizer: SplitResidualVectorQuantizer

    init(numCodebooks: Int) {
        let config = MimiConfig.moshiko(numCodebooks: numCodebooks, padMode: .constant)
        cfg = config
        _encoder.wrappedValue = SeanetEncoder(cfg: config.seanet)
        _encoder_transformer.wrappedValue = ProjectedTransformer(
            cfg: config.transformer, inputDim: config.seanet.dimension,
            outputDims: [config.seanet.dimension])
        _downsample.wrappedValue = ConvDownsample1d(stride: 2, dim: config.seanet.dimension, causal: true)
        _quantizer.wrappedValue = SplitResidualVectorQuantizer(
            dim: config.codebookDim, inputDim: config.seanet.dimension,
            outputDim: config.seanet.dimension, nq: numCodebooks, bins: config.codebookSize)
        super.init()
    }

    static func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        let mapped = Mimi.sanitizeWeights(weights)
        var output: [String: MLXArray] = [:]
        for (key, value) in mapped {
            if ["encoder.", "encoder_transformer.", "downsample.", "quantizer."].contains(where: key.hasPrefix) {
                output[key] = value.asType(.float32)
            } else if !["decoder.", "decoder_transformer.", "upsample."].contains(where: key.hasPrefix) {
                throw KyutaiSTTError.incompatibleCheckpoint("unknown Mimi tensor: \(key)")
            }
        }
        return output
    }
}

import MLXCommon
import MLXNN
import Qwen3TTS

enum BreezeTTSCodec {
    static func makeUpsampler(channels: Int) -> CausalTransposeConv1d {
        // Breeze's bundled Qwen3 codec uses kernel=2, stride=2 here.
        // The larger kernel assumed by the shared decoder would retain a
        // right-trim after loading these shorter weights and drop 2,880 samples.
        CausalTransposeConv1d(inputChannels: channels, outputChannels: channels, kernelSize: 2, stride: 2)
    }

    static func makeDecoder(config: SpeechTokenizerDecoderConfig) -> SpeechTokenizerDecoder {
        withThrowawayRandomState {
            let decoder = SpeechTokenizerDecoder(config: config)
            decoder.update(modules: ModuleChildren(values: [
                "preUpsample1": .value(makeUpsampler(channels: config.latentDim)),
                "preUpsample2": .value(makeUpsampler(channels: config.latentDim))
            ]))
            return decoder
        }
    }
}

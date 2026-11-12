import MLX
import XCTest

@testable import Qwen3TTS

final class DecoderFrameGeometryTests: XCTestCase {
    func testPreUpsamplingMatchesCheckpointKernelWidths() {
        let decoder = makeFrameGeometryDecoder()
        for layer in [decoder.preUpsample1, decoder.preUpsample2] {
            XCTAssertEqual(layer.conv.weight.dim(1), 2)
            XCTAssertEqual(layer.trimRight, 0)
        }
    }

    func testLoadedPreUpsamplingWeightsPreserveBothTimeExpansions() {
        let decoder = makeFrameGeometryDecoder()
        loadCheckpointShapedUpsamplingWeights(into: decoder)
        let input = MLXArray.zeros([1, 9, decoder.config.latentDim])
        let first = decoder.preUpsample1(input)
        let second = decoder.preUpsample2(first)
        XCTAssertEqual(first.dim(1), 18, "The first upsampler must double every input frame")
        XCTAssertEqual(second.dim(1), 36, "Both upsamplers must preserve all four output positions")
    }

    func testLoadedDecoderPreserves1920SamplesPerCodecFrame() {
        let decoder = makeFrameGeometryDecoder()
        loadCheckpointShapedUpsamplingWeights(into: decoder)
        for frames in [1, 3, 10, 25, 35] {
            let codes = MLXArray.zeros([1, decoder.config.numQuantizers, frames], type: Int32.self)
            let output = decoder(codes)
            XCTAssertEqual(output.shape, [1, frames * 1920, 1], "Frame count: \(frames)")
        }
    }
}

private func makeFrameGeometryDecoder() -> SpeechTokenizerDecoder {
    var config = SpeechTokenizerDecoderConfig()
    config.latentDim = 4
    config.decoderDim = 16
    config.hiddenSize = 4
    config.numHeads = 1
    config.numKVHeads = 1
    config.headDim = 4
    config.numLayers = 1
    config.numQuantizers = 2
    config.semanticCodebookSize = 8
    config.acousticCodebookSize = 8
    config.codebookDim = 2
    return SpeechTokenizerDecoder(config: config)
}

private func loadCheckpointShapedUpsamplingWeights(into decoder: SpeechTokenizerDecoder) {
    // The pretrained codec stores both pre-upsampling kernels as [C, C, 2].
    // Loading replaces the tensor, but cannot correct a constructor's trim count.
    for layer in [decoder.preUpsample1, decoder.preUpsample2] {
        let channels = decoder.config.latentDim
        layer.conv.update(parameters: .init(values: [
            "weight": .value(MLXArray.zeros([channels, 2, channels]))
        ]))
    }
}

final class E2EDecoderChunkGeometryTests: E2ETestCase {
    func testChunkedDecoderKeepsAllFramesAcrossChunkAndOverlapBoundaries() {
        let decoder = makeFrameGeometryDecoder()
        loadCheckpointShapedUpsamplingWeights(into: decoder)
        for frames in [35, 36, 40, 50, 51, 76] {
            let codes = MLXArray.zeros([1, decoder.config.numQuantizers, frames], type: Int32.self)
            let output = decoder.chunkedDecode(codes: codes)
            XCTAssertEqual(output.shape, [1, frames * 1920, 1], "Frame count: \(frames)")
        }
    }
}

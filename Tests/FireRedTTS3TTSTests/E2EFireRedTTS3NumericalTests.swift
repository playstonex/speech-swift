import Foundation
@testable import FireRedTTS3TTS
import MLX
import XCTest

final class E2EFireRedTTS3NumericalTests: XCTestCase {
    struct Tensor: Decodable {
        let shape: [Int]
        let values: [Float]
        var array: MLXArray { MLXArray(values).reshaped(shape) }
    }
    struct Fixture: Decodable {
        let config: FireRedTTS3Config
        let weights: [String: Tensor]
        let inputIds: [Int]
        let patchInput: Tensor
        let flowInput: Tensor
        let flowTimes: Tensor
        let decoderInput: Tensor
        let slidingInput: Tensor
        let expected: [String: Tensor]
    }

    private func load() throws -> (FireRedTTS3Network, Fixture) {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "network-fixture", withExtension: "json", subdirectory: "Resources"))
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let fixture = try decoder.decode(Fixture.self, from: Data(contentsOf: url))
        return (try FireRedTTS3Network(config: fixture.config, weights: fixture.weights.mapValues(\.array)), fixture)
    }

    private func compare(_ actual: MLXArray, _ reference: Tensor, tolerance: Float = 2e-5,
                         file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, reference.shape, file: file, line: line)
        let error = abs(actual - reference.array).max().item(Float.self)
        XCTAssertLessThanOrEqual(error, tolerance, "Maximum absolute error \(error)", file: file, line: line)
    }

    func testBackbonePrefillAndCachedStepsAgainstQwen3CPU() throws {
        let (network, f) = try load()
        var cache = Array<(MLXArray, MLXArray)?>(repeating: nil, count: f.config.backbone.numHiddenLayers)
        let full = try network.backbone(network.embedding(f.inputIds), cache: &cache, offset: 0)
        compare(full, f.expected["backbone_hidden"]!)
        compare(network.textLogits(full[0..., -1, 0...]), f.expected["backbone_logits"]!)
        cache = Array(repeating: nil, count: f.config.backbone.numHiddenLayers)
        _ = try network.backbone(network.embedding(Array(f.inputIds.prefix(3))), cache: &cache, offset: 0)
        let tail = try network.backbone(network.embedding(Array(f.inputIds.dropFirst(3))), cache: &cache, offset: 3)
        compare(tail, f.expected["backbone_cached_tail"]!)
    }

    func testPatchEncodingAndFlowAgainstPinnedUpstreamCPU() throws {
        let (network, f) = try load()
        compare(network.encodePatch(f.patchInput.array), f.expected["patch_encoded"]!)
        compare(network.flow(f.flowInput.array, time: f.flowTimes.array), f.expected["flow_velocity"]!)
    }

    func testSlidingDecoderAndWaveformAgainstPinnedUpstreamCPU() throws {
        let (network, f) = try load()
        var cache = Array<(MLXArray, MLXArray)?>(repeating: nil, count: f.config.decoder.numHiddenLayers)
        compare(try network.qwen(f.slidingInput.array, prefix: "decoder.qwen3", config: f.config.decoder,
            cache: &cache, offset: 0), f.expected["sliding_hidden"]!)
        compare(try network.decode(f.decoderInput.array), f.expected["decoder_audio"]!)
    }

    func testNonfinitePlanningScoresAndTemperatureOverflowFail() {
        let random = MLXRandom.RandomState(seed: 2)
        for values in [[Float.nan], [.infinity], [-Float.infinity]] {
            XCTAssertThrowsError(try FireRedTTS3Model.sample(MLXArray(values), options: .init(), random: random))
        }
        var options = FireRedTTS3Options(); options.textTemperature = 1e-40
        XCTAssertThrowsError(try FireRedTTS3Model.sample(MLXArray([Float(1)]), options: options, random: random))
    }

    func testISTFTBoundariesAgainstIndependentFrameLoop() throws {
        struct Case: Decodable {
            let hopLength: Int
            let window: [Float]
            let spectrumReal: [[[Float]]]
            let spectrumImag: [[[Float]]]
            let expectedPcm: [[Float]]
        }
        struct Cases: Decodable { let cases: [Case] }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "istft-fixture", withExtension: "json", subdirectory: "Resources"))
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        for item in try decoder.decode(Cases.self, from: Data(contentsOf: url)).cases {
            let shape = [item.spectrumReal.count, item.spectrumReal[0].count, item.spectrumReal[0][0].count]
            let real = MLXArray(item.spectrumReal.flatMap { $0.flatMap { $0 } }).reshaped(shape)
            let imag = MLXArray(item.spectrumImag.flatMap { $0.flatMap { $0 } }).reshaped(shape)
            let result = FireRedTTS3Network.istft(real + imag.asImaginary(), window: MLXArray(item.window), hop: item.hopLength)
            compare(result, Tensor(shape: [item.expectedPcm.count, item.expectedPcm[0].count],
                values: item.expectedPcm.flatMap { $0 }), tolerance: 2e-6)
        }
    }
}

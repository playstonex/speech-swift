@testable import BreezeTTS
import Foundation
import XCTest

final class BreezeTTSTests: XCTestCase {
    private func config(_ mutate: (inout [String: Any]) -> Void = { _ in }) throws -> BreezeTTSConfig {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "config", withExtension: "json", subdirectory: "Resources"))
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        mutate(&object)
        return try JSONDecoder().decode(BreezeTTSConfig.self, from: JSONSerialization.data(withJSONObject: object))
    }

    func testOfficialConfiguration() throws {
        let c = try config()
        XCTAssertEqual(c.numCodebooks, 16)
        XCTAssertEqual(c.audioVocabSize, 2051)
        XCTAssertEqual(c.outputSampleRate, 24_000)
        XCTAssertEqual(c.backboneConfig.hiddenSize, 2048)
        XCTAssertEqual(c.backboneConfig.ropeTheta, 1_000_000)
        XCTAssertEqual(c.textEncoderConfig.hiddenSize, 1152)
        XCTAssertEqual(c.textEncoderConfig.layerTypes.filter { $0 == "full_attention" }.count, 4)
        XCTAssertEqual(c.textEncoderConfig.ropeParameters["full_attention"]?.factor, 8)
        XCTAssertEqual(c.depthDecoderConfig.ropeScaling?.ropeType, "llama3")
    }

    func testQuantizationPropagatesToAllLinearStacks() throws {
        let c = try config { $0["quantization"] = ["bits": 8, "group_size": 64] }
        XCTAssertEqual(c.backboneConfig.quantization?.bits, 8)
        XCTAssertEqual(c.depthDecoderConfig.quantization?.bits, 8)
        XCTAssertEqual(c.textEncoderConfig.quantization?.bits, 8)
        XCTAssertThrowsError(try config { $0["quantization"] = ["bits": 3, "group_size": 64] })
        XCTAssertThrowsError(try config { $0["quantization"] = ["bits": 8, "group_size": 32] })
    }

    func testRejectsIncompatibleArchitecture() throws {
        XCTAssertThrowsError(try config { $0["audio_num_codebooks"] = 8 })
        XCTAssertThrowsError(try config { $0["codec_config"] = ["sampling_rate": 48_000] })
        XCTAssertThrowsError(try config { $0["text_encoder_config"] = ["num_hidden_layers": 2, "layer_types": ["full_attention"]] })
        XCTAssertThrowsError(try config { $0["text_encoder_config"] = ["num_hidden_layers": 2, "sliding_window_pattern": 0] })
    }

    func testPromptTemplatesMatchUpstream() {
        XCTAssertEqual(BreezeTTSModel.promptText(text: "Hello.", instruction: nil), "[S0]Hello.")
        XCTAssertEqual(BreezeTTSModel.promptText(text: "你好。", instruction: "温柔的声音"),
                       "[S0]<ins_bos>温柔的声音<ins_eos>你好。")
        XCTAssertEqual(BreezeTTSModel.promptText(text: "Hello.", instruction: " \n"), "[S0]Hello.")
    }

    func testCodecUsesBundledNormalizationEpsilon() throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "codec-config", withExtension: "json", subdirectory: "Resources"))
        let c = try BreezeTTSModel.codecConfiguration(Data(contentsOf: url))
        XCTAssertEqual(c.rmsNormEps, 1e-5)
        XCTAssertEqual(c.codebookDim, 256)
        XCTAssertThrowsError(try BreezeTTSModel.codecConfiguration(Data("{}".utf8)))
    }

    func testDepthLlama3ScalingUsesAllThreeFrequencyRegions() throws {
        let c = try config()
        let actual = breezeRoPEFrequencies(dimensions: 128, base: 500_000, scaling: c.depthDecoderConfig.ropeScaling)
        for i in [0, 20, 28, 35, 63] {
            let frequency = pow(Float(500_000), Float(2 * i) / 128)
            let wavelength = 2 * Float.pi * frequency
            let expected: Float
            if wavelength < 2048 { expected = frequency }
            else if wavelength > 8192 { expected = frequency * 32 }
            else {
                let smooth = (16 / wavelength - 0.001953125) / (0.0078125 - 0.001953125)
                expected = frequency / ((1 - smooth) / 32 + smooth)
            }
            XCTAssertEqual(actual[i], expected, accuracy: max(0.0001, expected * 0.00001))
        }
    }

    func testRejectsInvalidGenerationControls() throws {
        try BreezeTTSOptions().validate()
        for options in [BreezeTTSOptions(temperature: .nan), BreezeTTSOptions(topK: -1),
                        BreezeTTSOptions(topP: 0), BreezeTTSOptions(cfgScale: .infinity),
                        BreezeTTSOptions(maxFrames: 0), BreezeTTSOptions(firstChunkFrames: 0),
                        BreezeTTSOptions(chunkFrames: -1), BreezeTTSOptions(decoderLeftContext: -1)] {
            XCTAssertThrowsError(try options.validate())
        }
    }

    func testRejectsIncompleteOrInvalidReference() throws {
        try BreezeTTSModel.validateInput(text: "Hello", referenceAudio: nil, referenceText: nil, language: "en")
        XCTAssertThrowsError(try BreezeTTSModel.validateInput(text: " ", referenceAudio: nil, referenceText: nil, language: nil))
        XCTAssertThrowsError(try BreezeTTSModel.validateInput(text: "Hi", referenceAudio: [0], referenceText: nil, language: nil))
        XCTAssertThrowsError(try BreezeTTSModel.validateInput(text: "Hi", referenceAudio: nil, referenceText: "Hi", language: nil))
        XCTAssertThrowsError(try BreezeTTSModel.validateInput(text: "Hi", referenceAudio: [.nan], referenceText: "Hi", language: nil))
        XCTAssertThrowsError(try BreezeTTSModel.validateInput(text: "Hi", referenceAudio: nil, referenceText: nil, language: "de"))
    }

    func testMissingBundleFailsBeforeLoadingWeights() async throws {
        do {
            _ = try await BreezeTTSModel.fromBundle(URL(fileURLWithPath: "/nonexistent/breeze-bundle"))
            XCTFail("Expected a missing-file error")
        } catch BreezeTTSError.missingFile(let file) { XCTAssertEqual(file, "config.json") }
    }
}

import Foundation
import MLX
import MLXNN
import XCTest
@testable import FishAudioTTS

final class E2EFishAudioCodecStreamingMathTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        setenv("MLX_ENABLE_TF32", "0", 1)
    }

    func testCausalConvolutionPreservesDilatedHistory() throws {
        let layer = FishCodecConv1dNCL(inChannels: 1, outChannels: 1, kernelSize: 3, dilation: 3, causal: true)
        try layer.conv.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray([Float(0.1), 0.2, 0.3]).reshaped(1, 3, 1), "bias": MLXArray([Float(0.25)])
        ]), verify: .all)
        let input = MLXArray((1...17).map(Float.init)).reshaped(1, 1, 17)
        var expected: [Float] = []
        for i in 0..<17 {
            expected.append(0.25 + 0.3 * Float(i + 1) + (i >= 3 ? 0.2 * Float(i - 2) : 0)
                + (i >= 6 ? 0.1 * Float(i - 5) : 0))
        }
        var state = FishAudioCodecStreamingState()
        var output: [Float] = []
        var start = 0
        for size in [1, 3, 2, 11] {
            let chunk = layer.streaming(input[0..., 0..., start..<(start + size)], state: &state)
            eval([chunk] + state.arrays)
            output += chunk.asArray(Float.self)
            XCTAssertEqual(state.convolutions[ObjectIdentifier(layer)]?.dim(2), 6)
            start += size
        }
        assertClose(output, expected)
        assertClose(layer(input).asArray(Float.self), expected)
    }

    func testTransposedConvolutionCarriesOverlapWithoutDuplicatingBias() throws {
        let layer = FishCodecConvTransposed1dNCL(inChannels: 1, outChannels: 1, kernelSize: 4, stride: 2, causal: true)
        try layer.conv.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray([Float(0.1), 0.2, 0.3, 0.4]).reshaped(1, 4, 1), "bias": MLXArray([Float(0.25)])
        ]), verify: .all)
        let input = MLXArray([Float(1), 2, 3]).reshaped(1, 1, 3)
        let expected: [Float] = [0.35, 0.45, 0.75, 1.05, 1.15, 1.65]
        for sizes in [[1, 1, 1], [1, 2], [3]] {
            var state = FishAudioCodecStreamingState(), start = 0
            var output: [Float] = []
            for size in sizes {
                let chunk = layer.streaming(input[0..., 0..., start..<(start + size)], state: &state)
                eval([chunk] + state.arrays)
                output += chunk.asArray(Float.self)
                start += size
            }
            assertClose(output, expected)
        }
        assertClose(layer(input).asArray(Float.self), expected)
    }

    func testAttentionWindowAndRotaryOffsetAcrossIrregularChunks() throws {
        let layer = FishCodecTransformerAttention(hiddenSize: 4, numHeads: 2, headDim: 2)
        try layer.wqkv.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray((0..<48).map { Float(sin(Double($0))) * 0.2 }).reshaped(12, 4)
        ]), verify: .all)
        try layer.wo.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray((0..<16).map { Float(cos(Double($0))) * 0.2 }).reshaped(4, 4)
        ]), verify: .all)
        let input = MLXArray((0..<52).map { Float(sin(Double($0) * 0.3)) }).reshaped(1, 13, 4)
        var mask: [Float] = []
        for q in 0..<13 { for k in 0..<13 { mask.append(k <= q && k >= q - 4 ? 0 : -1e9) } }
        let expected = layer(input, mask: MLXArray(mask).reshaped(1, 1, 13, 13)).asArray(Float.self)
        var state = FishAudioCodecStreamingState(), start = 0
        var output: [Float] = []
        for size in [1, 3, 7, 2] {
            let chunk = layer.streaming(input[0..., start..<(start + size), 0...], window: 5, state: &state)
            eval([chunk] + state.arrays)
            output += chunk.asArray(Float.self)
            start += size
            let cache = try XCTUnwrap(state.attention[ObjectIdentifier(layer)])
            XCTAssertLessThanOrEqual(cache.keys.dim(2), 4)
            XCTAssertEqual(cache.offset, start)
        }
        assertClose(output, expected)
    }

    private func assertClose(_ actual: [Float], _ expected: [Float], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (a, b) in zip(actual, expected) { XCTAssertEqual(a, b, accuracy: 2e-5, file: file, line: line) }
    }
}

final class E2EFishAudioCodecStreamingTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        setenv("MLX_ENABLE_TF32", "0", 1)
    }

    func testLazyEncoderMatchesEagerReferenceCodesAndDecoder() throws {
        guard let path = ProcessInfo.processInfo.environment["FISH_DRAMA3_BUNDLE"] else {
            throw XCTSkip("Set FISH_DRAMA3_BUNDLE to a local codec bundle")
        }
        let directory = URL(fileURLWithPath: path)
        let lazy = try FishAudioCodec.loadForDrama3(from: directory, memoryOptions: .balanced)
        XCTAssertFalse(lazy.encodingWeightsLoaded)
        let codes = FishAudioGeneratedCodebooks(codes: (0..<10).map { row in
            (0..<3).map { (row * 29 + $0 * 37) % (row == 0 ? 4096 : 1024) }
        })
        let decoded = try lazy.decode(codes)
        XCTAssertFalse(lazy.encodingWeightsLoaded, "Decoding must not materialize reference weights")
        let eager = try FishAudioCodec.load(from: directory)
        let expected = try eager.decode(codes)
        XCTAssertEqual(decoded.count, expected.count)
        XCTAssertLessThan(zip(decoded, expected).map { abs($0 - $1) }.max() ?? .infinity, 2e-5)
        let reference = (0..<6144).map { Float(0.2 * sin(2 * Double.pi * 440 * Double($0) / 44100)) }
        let referenceCodes = try eager.encode(audio: reference)
        XCTAssertEqual(try lazy.encode(audio: reference), referenceCodes)
        XCTAssertTrue(lazy.encodingWeightsLoaded)
        XCTAssertEqual(try lazy.encode(audio: reference), referenceCodes)
    }

    func testFullTowerVocabularyMatchesBufferedResidualClamping() throws {
        guard let path = ProcessInfo.processInfo.environment["FISH_DRAMA3_BUNDLE"] else {
            throw XCTSkip("Set FISH_DRAMA3_BUNDLE to a local codec bundle")
        }
        let codec = try FishAudioCodec.load(from: URL(fileURLWithPath: path))
        let codes = FishAudioGeneratedCodebooks(codes: [[0, 1, 4095]] + Array(repeating: [0, 1024, 4095], count: 9))
        let expected = try codec.decode(codes)
        var state = FishAudioCodecStreamingState()
        var actual: [Float] = []
        for frame in 0..<3 {
            let chunk = FishAudioGeneratedCodebooks(codes: codes.codes.map { [$0[frame]] })
            actual += try codec.decodeStreaming(chunk, state: &state)
        }
        XCTAssertEqual(actual.count, expected.count)
        let maximum = zip(actual, expected).map { abs($0 - $1) }.max() ?? .infinity
        XCTAssertLessThan(maximum, 2e-4)
    }

    func testReleasedCodecFullAndStreamingParityIncludingWindowRollover() throws {
        guard let path = ProcessInfo.processInfo.environment["FISH_DRAMA3_BUNDLE"] else {
            throw XCTSkip("Set FISH_DRAMA3_BUNDLE to a local codec bundle")
        }
        let codec = try FishAudioCodec.load(from: URL(fileURLWithPath: path))
        let codes = FishAudioGeneratedCodebooks(codes: (0..<10).map { row in
            (0..<137).map { (row * 29 + $0 * 37) % (row == 0 ? 4096 : 1024) }
        })
        let expected = try codec.decode(codes)
        XCTAssertEqual(expected.count, 137 * 2048)
        for size in [1, 3, 17, 128] {
            var state = FishAudioCodecStreamingState()
            var audio: [Float] = []
            for start in stride(from: 0, to: 137, by: size) {
                let end = min(137, start + size)
                let chunk = FishAudioGeneratedCodebooks(codes: codes.codes.map { Array($0[start..<end]) })
                audio += try codec.decodeStreaming(chunk, state: &state)
                XCTAssertTrue(state.attention.values.allSatisfy { $0.keys.dim(2) <= 127 })
            }
            XCTAssertEqual(audio.count, expected.count)
            let maximum = zip(audio, expected).map { abs($0 - $1) }.max() ?? .infinity
            let squaredError: Float = zip(audio, expected).reduce(0) { sum, pair in
                let difference = pair.0 - pair.1
                return sum + difference * difference
            }
            let rms = sqrt(squaredError / Float(expected.count))
            print("Fish DAC streaming chunk \(size): max error \(maximum), RMS \(rms)")
            XCTAssertLessThan(maximum, 2e-4)
            XCTAssertLessThan(rms, 2e-5)
        }
        var state = FishAudioCodecStreamingState()
        XCTAssertEqual(try codec.decodeStreaming(.init(codes: Array(repeating: [], count: 10)), state: &state), [])
        XCTAssertThrowsError(try codec.decodeStreaming(.init(codes: [[1], []]), state: &state))
        XCTAssertThrowsError(try codec.decodeStreaming(.init(codes: [[4096]] + Array(repeating: [0], count: 9)), state: &state))
        XCTAssertThrowsError(try codec.decodeStreaming(.init(codes: [[0], [4096]] + Array(repeating: [0], count: 8)), state: &state))
    }
}

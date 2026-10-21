import Foundation
import XCTest
import MLX
import MLXNN
@testable import StableAudio3MusicGen

final class WeightLoadingRandomStateTests: XCTestCase {
    func testRejectedDecoderWeightsPreserveCallerRandomKey() throws {
        let directory = try makeBundle([
            "mapping.weight": MLXArray.zeros([1, 1]),
        ])
        let caller = MLXRandom.RandomState(seed: 42)
        let originalKey = caller.innerState()[0]

        try withRandomState(caller) {
            XCTAssertThrowsError(try sa3LoadSAMELDecoder(dir: directory)) { error in
                guard case UpdateError.mismatchedSize = error else {
                    return XCTFail("Expected rejection of the checkpoint shape, got \(error)")
                }
            }
        }

        XCTAssertTrue(caller.innerState()[0] === originalKey)
    }

    func testRejectedTextEncoderWeightsPreserveCallerRandomKey() throws {
        // A minimal SentencePiece ModelProto containing one "a" piece.
        let tokenizer = MLXArray([UInt8(0x0a), 3, 0x0a, 1, 0x61])
        let directory = try makeBundle([
            "TOKENIZER_MODEL": tokenizer,
            "norm.weight": MLXArray.zeros([1, 1]),
        ])
        let caller = MLXRandom.RandomState(seed: 42)
        let originalKey = caller.innerState()[0]

        try withRandomState(caller) {
            XCTAssertThrowsError(try sa3LoadT5Gemma(dir: directory)) { error in
                guard case UpdateError.mismatchedSize = error else {
                    return XCTFail("Expected rejection of the checkpoint shape, got \(error)")
                }
            }
        }

        XCTAssertTrue(caller.innerState()[0] === originalKey)
    }

    private func makeBundle(_ weights: [String: MLXArray]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("stable-audio-loading-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }
}

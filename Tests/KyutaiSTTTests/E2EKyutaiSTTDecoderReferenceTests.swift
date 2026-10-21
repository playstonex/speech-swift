import Foundation
import MLX
import XCTest
@testable import KyutaiSTT

/// Separates decoder math from floating-point codebook decisions in Mimi.
final class E2EKyutaiSTTDecoderReferenceTests: XCTestCase {
    func testDecoderWithReferenceAudioCodes() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["KYUTAI_STT_MODEL_PATH"],
              let referencePath = ProcessInfo.processInfo.environment["KYUTAI_STT_DECODER_REFERENCE_PATH"] else {
            throw XCTSkip("Set the model and matching-precision decoder reference JSON")
        }
        struct Reference: Decodable {
            let frame_token_ids: [Int]
            let mimi_codebooks: [[Int32]]
        }
        let reference = try JSONDecoder().decode(
            Reference.self, from: Data(contentsOf: URL(fileURLWithPath: referencePath)))
        XCTAssertFalse(reference.frame_token_ids.isEmpty)
        XCTAssertEqual(reference.frame_token_ids.count, reference.mimi_codebooks.count)
        let model = try await KyutaiSTTModel.load(modelPath, offlineMode: true)
        let cache = (0..<model.config.numLayers).map { _ in KyutaiSTTCache(capacity: model.config.context) }
        var previousAudio = MLXArray.ones([1, model.config.numCodebooks, 1], dtype: .int32) * Int32(model.config.card)
        var previousText = model.config.textCard
        for (position, expected) in reference.frame_token_ids.enumerated() {
            let codes = reference.mimi_codebooks[position]
            XCTAssertEqual(codes.count, model.config.numCodebooks)
            let actual = model.decoder.step(
                text: previousText, audio: previousAudio, cache: cache, position: position)
                .argMax(axis: -1).item(Int.self)
            XCTAssertEqual(actual, expected, "Decoder token at frame \(position), with reference audio codes and text history")
            previousText = expected
            previousAudio = MLXArray(codes).reshaped([1, model.config.numCodebooks, 1])
        }
    }
}

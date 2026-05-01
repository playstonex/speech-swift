#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

#if os(macOS)
private typealias CoreMLFloat = Float
private let coreMLDataType: MLMultiArrayDataType = .float32
#else
private typealias CoreMLFloat = Float16
private let coreMLDataType: MLMultiArrayDataType = .float16
#endif

extension WeSpeakerModel {

    func embedCoreML(melSpec: [Float], nFrames: Int) throws -> [Float] {
        guard let model = coremlModel else {
            throw AudioModelError.inferenceFailed(
                operation: "SpeakerEmbedding", reason: "CoreML model not loaded")
        }

        let targetLength = Self.enumeratedMelLengths.first { $0 >= nFrames }
            ?? Self.enumeratedMelLengths.last!

        let melArray = try MLMultiArray(
            shape: [1, targetLength as NSNumber, 80],
            dataType: coreMLDataType
        )
        let melPtr = melArray.dataPointer.assumingMemoryBound(to: CoreMLFloat.self)

        let copyCount = min(nFrames, targetLength) * 80
        for i in 0..<copyCount {
            melPtr[i] = CoreMLFloat(melSpec[i])
        }
        let totalElements = targetLength * 80
        for i in copyCount..<totalElements {
            melPtr[i] = 0
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melArray),
        ])

        let result = try model.prediction(from: input)

        guard let embArray = result.featureValue(for: "embedding")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "SpeakerEmbedding", reason: "Missing 'embedding' output")
        }

        var embedding = [Float](repeating: 0, count: 256)
        let embPtr = embArray.dataPointer.assumingMemoryBound(to: CoreMLFloat.self)
        for i in 0..<256 {
            embedding[i] = Float(embPtr[i])
        }

        let norm = sqrt(embedding.reduce(Float(0)) { $0 + $1 * $1 })
        if norm > 1e-10 {
            for i in 0..<256 { embedding[i] /= norm }
        }

        return embedding
    }
}
#endif

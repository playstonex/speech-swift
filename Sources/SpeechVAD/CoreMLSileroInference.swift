#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

#if canImport(Foundation) && os(macOS)
private typealias CoreMLFloat = Float
private let coreMLDataType: MLMultiArrayDataType = .float32
#else
private typealias CoreMLFloat = Float16
private let coreMLDataType: MLMultiArrayDataType = .float16
#endif

extension SileroVADModel {

    func processChunkCoreML(_ fullSamples: [Float]) throws -> Float {
        guard let model = coremlModel else {
            throw AudioModelError.inferenceFailed(
                operation: "VAD", reason: "CoreML model not loaded")
        }

        let audioArray = try MLMultiArray(shape: [1, 1, 576], dataType: coreMLDataType)
        let audioPtr = audioArray.dataPointer.assumingMemoryBound(to: CoreMLFloat.self)
        for i in 0..<576 {
            audioPtr[i] = CoreMLFloat(fullSamples[i])
        }

        if coremlH == nil {
            coremlH = try MLMultiArray(shape: [1, 1, 128], dataType: coreMLDataType)
            coremlC = try MLMultiArray(shape: [1, 1, 128], dataType: coreMLDataType)
            zeroFillCoreML(coremlH!)
            zeroFillCoreML(coremlC!)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: audioArray),
            "h": MLFeatureValue(multiArray: coremlH!),
            "c": MLFeatureValue(multiArray: coremlC!),
        ])

        let result = try model.prediction(from: input)

        coremlH = result.featureValue(for: "h_out")!.multiArrayValue!
        coremlC = result.featureValue(for: "c_out")!.multiArrayValue!

        let probArray = result.featureValue(for: "probability")!.multiArrayValue!
        let probPtr = probArray.dataPointer.assumingMemoryBound(to: CoreMLFloat.self)
        return Float(probPtr[0])
    }

    private func zeroFillCoreML(_ array: MLMultiArray) {
        let ptr = UnsafeMutableBufferPointer(
            start: array.dataPointer.assumingMemoryBound(to: CoreMLFloat.self),
            count: array.count)
        for i in 0..<ptr.count {
            ptr[i] = 0
        }
    }
}
#endif

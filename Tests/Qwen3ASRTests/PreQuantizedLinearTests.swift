import XCTest
import MLX
import MLXNN
import MLXRandom
import MLXCommon
@testable import Qwen3ASR

/// `QuantizedLinear.preQuantized` stands in for the random-init
/// `QuantizedLinear(…)` in every layer a checkpoint fills in, so it must have
/// the same shapes, compute the same thing once loaded, and — the reason it
/// exists — leave nothing behind when a model is loaded again. Synthetic:
/// no model download.
final class PreQuantizedLinearTests: XCTestCase {

    private let inputDimensions = 256
    private let outputDimensions = 64

    func testPlaceholderArraysMatchTheRandomInitLayer() {
        let reference = QuantizedLinear(inputDimensions, outputDimensions, bias: true, groupSize: 64, bits: 4)
        let placeholder = QuantizedLinear.preQuantized(inputDimensions, outputDimensions, bias: true, groupSize: 64, bits: 4)

        XCTAssertEqual(placeholder.weight.shape, reference.weight.shape)
        XCTAssertEqual(placeholder.weight.dtype, reference.weight.dtype)
        XCTAssertEqual(placeholder.scales.shape, reference.scales.shape)
        XCTAssertEqual(placeholder.biases?.shape, reference.biases?.shape)
        XCTAssertEqual(placeholder.bias?.shape, reference.bias?.shape)
        XCTAssertEqual(placeholder.groupSize, reference.groupSize)
        XCTAssertEqual(placeholder.bits, reference.bits)
        XCTAssertEqual(placeholder.shape.0, reference.shape.0)
        XCTAssertEqual(placeholder.shape.1, reference.shape.1)
        XCTAssertNil(QuantizedLinear.preQuantized(inputDimensions, outputDimensions).bias)
    }

    func testLoadedPlaceholderComputesTheSameOutputAsTheLoadedArrays() {
        let checkpoint = makeCheckpoint(prefix: "proj")
        let loaded = QuantizedLinear.preQuantized(inputDimensions, outputDimensions, groupSize: 64, bits: 4)
        CommonWeightLoader.applyQuantizedLinearWeights(to: loaded, prefix: "proj", from: checkpoint)
        let reference = QuantizedLinear(
            weight: checkpoint["proj.weight"]!, bias: nil,
            scales: checkpoint["proj.scales"]!, biases: checkpoint["proj.biases"]!,
            groupSize: 64, bits: 4)

        let input = MLXRandom.normal([3, inputDimensions])
        XCTAssertTrue(arrayEqual(loaded(input), reference(input)).item(Bool.self))
    }

    /// Regression: loading a checkpoint into a random-init `QuantizedLinear`
    /// overwrote its lazy quantize outputs in place, which kept the whole init
    /// graph — and the scalar buffers feeding it — alive for the lifetime of
    /// the process, once per layer per model load.
    func testReloadingIntoPlaceholderLayersLeavesNoBuffersBehind() {
        let checkpoint = makeCheckpoint(prefix: "proj")
        func reload(times: Int) {
            for _ in 0..<times {
                let layer = QuantizedLinear.preQuantized(inputDimensions, outputDimensions, groupSize: 64, bits: 4)
                CommonWeightLoader.applyQuantizedLinearWeights(to: layer, prefix: "proj", from: checkpoint)
                eval(layer.parameters())
            }
        }
        // Optimized builds may release the checkpoint after its last reload.
        // Keep it alive so both readings include the same fixture buffers.
        withExtendedLifetime(checkpoint) {
            reload(times: 5)
            let activeMemoryBefore = Memory.activeMemory
            reload(times: 100)
            XCTAssertEqual(Memory.activeMemory, activeMemoryBefore)
        }
    }

    private func makeCheckpoint(prefix: String) -> [String: MLXArray] {
        let weight = MLXRandom.uniform(low: Float(-1), high: Float(1), [outputDimensions, inputDimensions])
        let (quantizedWeight, scales, biases) = MLX.quantized(weight, groupSize: 64, bits: 4)
        let checkpoint = [
            "\(prefix).weight": quantizedWeight,
            "\(prefix).scales": scales,
            "\(prefix).biases": biases!,
        ]
        eval(Array(checkpoint.values))
        return checkpoint
    }
}

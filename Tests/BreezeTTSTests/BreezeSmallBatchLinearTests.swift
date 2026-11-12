@testable import BreezeTTS
import MLX
import XCTest

final class BreezeSmallBatchLinearTests: XCTestCase {
    func testProjectionPreservesLeadingDimensionsAndPrecision() {
        Device.withDefaultDevice(.cpu) {
            let weight = MLXArray([Float(1), 2, 3, -1, 0, 1]).reshaped([2, 3])
            let pair: [Float] = [1, 2, 3, -2, 0, 2]
            let expectedPair: [Float] = [14, 2, 4, 4]
            for dtype in [DType.bfloat16, .float16, .float32] {
                let layer = BreezeSmallBatchLinear(weight: weight.asType(dtype))
                for shape in [[2, 3], [1, 2, 3], [2, 1, 3], [4, 3], [2, 2, 3], [8, 3]] {
                    let rows = shape.dropLast().reduce(1, *)
                    let values = Array(repeating: pair, count: rows / 2).flatMap { $0 }
                    let input = MLXArray(values).asType(dtype).reshaped(shape)
                    let actual = layer(input)
                    XCTAssertEqual(actual.shape, Array(shape.dropLast()) + [2])
                    XCTAssertEqual(actual.dtype, dtype)
                    XCTAssertEqual(actual.asType(.float32).asArray(Float.self),
                        Array(repeating: expectedPair, count: rows / 2).flatMap { $0 })
                }
            }
        }
    }

    func testNoncontiguousInputsKeepTheirElementOrder() {
        Device.withDefaultDevice(.cpu) {
            for dtype in [DType.bfloat16, .float16] {
                let weight = MLXArray([Float(1), 2, 3, -1, 0, 1]).asType(dtype).reshaped([2, 3])
                let layer = BreezeSmallBatchLinear(weight: weight)
                let input = MLXArray([Float(1), -2, 2, 0, 3, 2]).asType(dtype)
                    .reshaped([3, 2]).transposed(1, 0)
                XCTAssertEqual(layer(input).asType(.float32).asArray(Float.self), [14, 2, 4, 4])
            }
        }
    }

    func testBiasAndSingleRowRetainLinearBehavior() {
        Device.withDefaultDevice(.cpu) {
            let weight = MLXArray([Float(1), 2, 3, -1, 0, 1]).asType(.bfloat16).reshaped([2, 3])
            let layer = BreezeSmallBatchLinear(weight: weight, bias: MLXArray([Float(0.5), -0.5]).asType(.bfloat16))
            let input = MLXArray([Float(1), 2, 3]).asType(.bfloat16).reshaped([1, 3])
            XCTAssertEqual(layer(input).asType(.float32).asArray(Float.self), [14.5, 1.5])
            let pair = concatenated([input, input], axis: 0)
            XCTAssertEqual(layer(pair).asType(.float32).asArray(Float.self), [14.5, 1.5, 14.5, 1.5])
        }
    }
}

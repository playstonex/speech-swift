import Foundation
import MLX
import MLXNN

final class BreezeSmallBatchLinear: Linear {
    private let useVectorKernel = ProcessInfo.processInfo.environment["SPEECH_SWIFT_BREEZE_GEMV"] != "0"

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard useVectorKernel, bias == nil, x.ndim >= 2, x.dim(-1) > 0,
              x.size / x.dim(-1) == 2, weight.dtype == .bfloat16 else {
            return super.callAsFunction(x)
        }
        // Keep the two guidance vectors in the batch dimension. A conventional
        // x @ weight.T collapses them into a two-row matrix multiplication;
        // this orientation lets Metal use its batched matrix-vector kernel.
        let output = matmul(weight, x.reshaped([2, -1, 1]))
        let shape = Array(x.shape.dropLast()) + [weight.dim(0)]
        return output.reshaped(shape)
    }
}

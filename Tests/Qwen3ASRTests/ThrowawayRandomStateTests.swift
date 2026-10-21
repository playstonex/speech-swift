import XCTest
import MLX
import MLXNN
import MLXCommon

final class ThrowawayRandomStateTests: XCTestCase {
    func testRepeatedCheckpointLoadsPreserveCallerRandomKey() {
        let caller = MLXRandom.RandomState(seed: 42)
        let originalKey = caller.innerState()[0]
        let checkpoint = ModuleParameters.unflattened([
            "weight": MLXArray.zeros([16, 32]),
        ])

        withRandomState(caller) {
            for _ in 0..<100 {
                let layer = withThrowawayRandomState { Linear(32, 16, bias: false) }
                layer.update(parameters: checkpoint)
            }
        }

        // Key identity detects retention of the lazy split chain even when
        // Memory.activeMemory reports no additional GPU buffers.
        XCTAssertTrue(caller.innerState()[0] === originalKey)
    }

    func testScopeRestoresCallerAfterThrowing() throws {
        enum LoadingError: Error { case invalidCheckpoint }
        let caller = MLXRandom.RandomState(seed: 42)
        let originalKey = caller.innerState()[0]

        try withRandomState(caller) {
            XCTAssertThrowsError(try withThrowawayRandomState {
                _ = MLXRandom.normal([16])
                throw LoadingError.invalidCheckpoint
            })
            XCTAssertTrue(caller.innerState()[0] === originalKey)

            _ = MLXRandom.normal([1])
            XCTAssertFalse(caller.innerState()[0] === originalKey,
                           "Random draws after the scope must use the caller's state")
        }
    }

    func testScopePreservesGlobalRandomKey() {
        let originalKey = MLXRandom.globalState.innerState()[0]
        let layer = withThrowawayRandomState { Linear(32, 16, bias: false) }

        XCTAssertEqual(layer.weight.shape, [16, 32])
        XCTAssertTrue(MLXRandom.globalState.innerState()[0] === originalKey)
    }
}

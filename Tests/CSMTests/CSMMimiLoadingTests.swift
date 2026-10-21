import Foundation
import XCTest
import MLX
@testable import CSM

final class CSMMimiLoadingTests: XCTestCase {
    func testMissingCheckpointDoesNotAdvanceCallerRandomKey() throws {
        let caller = MLXRandom.RandomState(seed: 42)
        let originalKey = caller.innerState()[0]
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("mimi.safetensors")

        try withRandomState(caller) {
            XCTAssertThrowsError(try CSMMimi.load(from: missing))
        }

        XCTAssertTrue(caller.innerState()[0] === originalKey,
                      "Failed codec loads must not retain initialization in the caller's random state")
    }
}

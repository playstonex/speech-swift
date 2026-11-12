import AudioCommon
import XCTest

final class InferenceCancellationTokenTests: XCTestCase {
    func testCancellationIsVisibleAcrossThreads() throws {
        let token = InferenceCancellationToken()
        try token.checkCancellation()
        let done = expectation(description: "cancelled")
        DispatchQueue.global().async { token.cancel(); done.fulfill() }
        wait(for: [done], timeout: 2)
        XCTAssertTrue(token.isCancelled)
        XCTAssertThrowsError(try token.checkCancellation()) { XCTAssertTrue($0 is CancellationError) }
    }
}

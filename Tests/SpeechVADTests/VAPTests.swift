#if canImport(CoreML)
import Foundation
import XCTest
@testable import SpeechVAD

final class VAPTests: XCTestCase {
    func testStereoChannelsKeepTheirOrderAndSharePadding() throws {
        let window = try VAPModel.preparedWindow(user: [1, 2], agent: [3, 4])
        XCTAssertEqual(window.count, 640_000)
        XCTAssertEqual(Array(window[319_997..<320_000]), [0, 1, 2])
        XCTAssertEqual(Array(window[639_997..<640_000]), [0, 3, 4])
    }

    func testLongWindowsRetainTheSameRecentInterval() throws {
        let user = [Float](repeating: 0.1, count: 320_001)
        var agent = [Float](repeating: 0.2, count: 320_001)
        agent[1] = 0.3
        let window = try VAPModel.preparedWindow(user: user, agent: agent)
        XCTAssertEqual(window[320_000], 0.3)
        XCTAssertEqual(window.last, 0.2)
    }

    func testUnsynchronizedEmptyAndNonfiniteChannelsAreRejected() {
        XCTAssertThrowsError(try VAPModel.preparedWindow(user: [], agent: []))
        XCTAssertThrowsError(try VAPModel.preparedWindow(user: [0], agent: [0, 0]))
        XCTAssertThrowsError(try VAPModel.preparedWindow(user: [.nan], agent: [0]))
        XCTAssertThrowsError(try VAPModel.preparedWindow(user: [0], agent: [.infinity]))
    }

    func testReversedChannelManifestIsRejected() throws {
        var config: [String: Any] = [
            "model_type": "vap-oto-coreml", "sample_rate": 16_000, "channels": 2,
            "channel_order": ["user", "agent"], "window_samples": 320_000,
            "frame_hz": 50, "compiled_model": "vap.mlmodelc",
        ]
        XCTAssertNoThrow(try VAPModel.decodeConfiguration(JSONSerialization.data(withJSONObject: config)))
        config["channel_order"] = ["agent", "user"]
        XCTAssertThrowsError(try VAPModel.decodeConfiguration(JSONSerialization.data(withJSONObject: config)))
    }
}
#endif

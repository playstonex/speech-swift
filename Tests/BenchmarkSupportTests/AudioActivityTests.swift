import BenchmarkSupport
import XCTest

final class AudioActivityTests: XCTestCase {
    func testSilenceAndQuietNoiseRemainInactive() {
        var activity = BenchmarkAudioActivity(windowSamples: 4, rmsThreshold: 0.1)
        activity.append([0, 0, 0, 0, 0.01, -0.01, 0.01, -0.01])
        XCTAssertNil(activity.firstActiveSample)
        XCTAssertEqual(activity.processedSamples, 8)
    }

    func testWindowSpansChunkBoundariesAndReportsGlobalOffset() {
        var activity = BenchmarkAudioActivity(windowSamples: 4, rmsThreshold: 0.1)
        activity.append([0, 0, 0])
        activity.append([0, 0.2, -0.2])
        XCTAssertNil(activity.firstActiveSample)
        activity.append([0.2, -0.2, 0])
        XCTAssertEqual(activity.firstActiveSample, 4)
        XCTAssertEqual(activity.processedSamples, 9)
        activity.append([1, 1, 1])
        XCTAssertEqual(activity.firstActiveSample, 4)
        XCTAssertEqual(activity.processedSamples, 12)
    }

    func testRMSUsesEntireWindowRatherThanSinglePeak() {
        var activity = BenchmarkAudioActivity(windowSamples: 4, rmsThreshold: 0.1)
        activity.append([0.15, 0, 0, 0])
        XCTAssertNil(activity.firstActiveSample)
        activity.append([0.2, 0, 0, 0])
        XCTAssertEqual(activity.firstActiveSample, 4)
    }
}

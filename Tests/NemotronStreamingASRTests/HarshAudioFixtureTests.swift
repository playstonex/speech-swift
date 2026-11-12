import CryptoKit
import XCTest
@testable import AudioCommon

final class HarshAudioFixtureTests: XCTestCase {
    func testOverlapFixtureIsPinnedAudio() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "overlapped_speech", withExtension: "wav"))
        let data = try Data(contentsOf: url)
        let checksum = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(
            checksum,
            "6bba4d96f4ca458a436d87f7a45a1268532c91a79b5cafdd896509222d703d0c",
            "Keep the overlap regression's audio fixed across runs and model updates"
        )

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: url)
        XCTAssertEqual(sampleRate, 24_000)
        XCTAssertEqual(samples.count, 104_880)
        XCTAssertTrue(samples.allSatisfy { $0.isFinite })
        let rms = sqrt(samples.reduce(Float(0)) { $0 + $1 * $1 } / Float(samples.count))
        XCTAssertGreaterThan(rms, 0.01, "Overlap fixture must contain audible speech")
    }
}

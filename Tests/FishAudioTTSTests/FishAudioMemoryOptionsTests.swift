import Foundation
import XCTest
@testable import FishAudioTTS

final class FishAudioMemoryOptionsTests: XCTestCase {
    func testBalancedDefaultsAndUnrestrictedOptOut() {
        XCTAssertEqual(FishAudioMemoryOptions.balanced.cacheTrimThresholdBytes, 512 * 1024 * 1024)
        XCTAssertTrue(FishAudioMemoryOptions.balanced.lazyReferenceEncoder)
        XCTAssertNil(FishAudioMemoryOptions.unrestricted.cacheTrimThresholdBytes)
        XCTAssertFalse(FishAudioMemoryOptions.unrestricted.lazyReferenceEncoder)
    }

    func testRejectsNegativeThresholdBeforeLoading() throws {
        XCTAssertThrowsError(try FishAudioMemoryOptions(cacheTrimThresholdBytes: -1).validate())
        try FishAudioMemoryOptions(cacheTrimThresholdBytes: 0).validate()
        try FishAudioMemoryOptions(cacheTrimThresholdBytes: nil).validate()
    }

    func testTrimsOnlyAboveThresholdWithoutChangingGlobalLimits() {
        let options = FishAudioMemoryOptions(cacheTrimThresholdBytes: 100)
        var trims = 0
        for bytes in [0, 99, 100] {
            options.trimCache(cacheBytes: { bytes }, clearCache: { trims += 1 })
        }
        XCTAssertEqual(trims, 0)
        options.trimCache(cacheBytes: { 101 }, clearCache: { trims += 1 })
        XCTAssertEqual(trims, 1)
    }

    func testUnrestrictedDoesNotInspectOrTrimCache() {
        FishAudioMemoryOptions.unrestricted.trimCache(cacheBytes: {
            XCTFail("Unrestricted policy must preserve caller memory management")
            return Int.max
        }, clearCache: { XCTFail("Unexpected cache trim") })
    }

    func testInvalidPolicyFailsBeforeOpeningModelFiles() async {
        do {
            _ = try await FishAudioDrama3Model.fromBundle(URL(fileURLWithPath: "/missing/drama3"),
                memoryOptions: .init(cacheTrimThresholdBytes: -1))
            XCTFail("Expected invalid memory policy")
        } catch FishAudioError.invalidInput { }
        catch { XCTFail("Expected policy validation before file access: \(error)") }
    }
}

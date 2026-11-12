import XCTest
@testable import SourceSeparation

final class StemMixerTests: XCTestCase {
    func testSumsBackingStemsWithoutVocalsOrQuietGainBoost() throws {
        let stems: [SeparationTarget: [[Float]]] = [
            .drums: [[0.125, -0.25], [0.25, -0.125]],
            .bass: [[0.25, 0.125], [-0.125, 0.25]],
            .other: [[0.125, 0.125], [0.125, -0.125]],
            .vocals: [[100, 100], [100, 100]],
        ]
        let mix = try StemMixer.instrumental(from: stems)
        XCTAssertEqual(mix, [[0.5, 0], [0.25, 0]])
        XCTAssertEqual(stems[.drums], [[0.125, -0.25], [0.25, -0.125]])
    }

    func testPeakProtectionUsesOneGainForBothChannels() throws {
        let stems: [SeparationTarget: [[Float]]] = [
            .drums: [[1, -0.5], [0.5, 0]],
            .bass: [[1, -0.5], [0.5, 0]],
            .other: [[0, 0], [0, 0]],
        ]
        XCTAssertEqual(try StemMixer.instrumental(from: stems), [[1, -0.5], [0.5, 0]])
    }

    func testFiniteInputsCannotOverflowBeforeAttenuation() throws {
        let maximum = Float.greatestFiniteMagnitude
        let stems: [SeparationTarget: [[Float]]] = [
            .drums: [[maximum], [maximum]],
            .bass: [[maximum], [maximum]],
            .other: [[maximum], [0]],
        ]
        let mix = try StemMixer.instrumental(from: stems)
        XCTAssertEqual(mix[0][0], 1)
        XCTAssertEqual(mix[1][0], 2 / 3, accuracy: 1e-6)
    }

    func testSilenceRemainsFiniteSilence() throws {
        let stems: [SeparationTarget: [[Float]]] = [
            .drums: [[0, 0], [0, 0]],
            .bass: [[0, 0], [0, 0]],
            .other: [[0, 0], [0, 0]],
        ]
        XCTAssertEqual(try StemMixer.instrumental(from: stems), [[0, 0], [0, 0]])
    }

    func testRejectsMissingBackingStem() {
        XCTAssertThrowsError(try StemMixer.instrumental(from: [
            .drums: [[0], [0]], .bass: [[0], [0]],
        ])) { error in
            XCTAssertEqual(error as? StemMixer.MixingError, .missingStem(.other))
        }
    }

    func testRejectsEmptyOrNonStereoInputs() {
        let malformedInputs: [[[Float]]] = [[], [[]], [[], []], [[0]], [[0], [0, 0]], [[0], [0], [0]]]
        for malformed in malformedInputs {
            XCTAssertThrowsError(try StemMixer.instrumental(from: [
                .drums: malformed, .bass: [[0], [0]], .other: [[0], [0]],
            ])) { error in
                XCTAssertEqual(error as? StemMixer.MixingError, .invalidStereoShape(.drums))
            }
        }
    }

    func testRejectsDifferentFrameCountsAcrossStems() {
        XCTAssertThrowsError(try StemMixer.instrumental(from: [
            .drums: [[0, 0], [0, 0]], .bass: [[0], [0]], .other: [[0, 0], [0, 0]],
        ])) { error in
            XCTAssertEqual(error as? StemMixer.MixingError, .mismatchedLength(.bass))
        }
    }

    func testRejectsNonFiniteBackingSamples() {
        for sample in [Float.nan, .infinity, -.infinity] {
            XCTAssertThrowsError(try StemMixer.instrumental(from: [
                .drums: [[0], [0]], .bass: [[0], [sample]], .other: [[0], [0]],
            ])) { error in
                XCTAssertEqual(error as? StemMixer.MixingError, .nonFiniteSamples(.bass))
            }
        }
    }
}

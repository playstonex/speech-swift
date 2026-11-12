import ArgumentParser
import XCTest
@testable import AudioCLILib

final class AnalyzeMusicCommandTests: XCTestCase {
    func testParsesDefaults() throws {
        let root = try AudioCLI.parseAsRoot(["analyze-music", "song.wav"])
        let command = try XCTUnwrap(root as? AnalyzeMusicCommand)
        XCTAssertEqual(command.input, "song.wav")
        XCTAssertEqual(command.start, 0)
        XCTAssertEqual(command.maxSeconds, 120)
        XCTAssertFalse(command.json)
    }

    func testParsesExplicitWindowAndJSON() throws {
        let root = try AudioCLI.parseAsRoot([
            "analyze-music", "Your Song.m4a", "--start", "12.5", "--max-seconds", "30", "--json"
        ])
        let command = try XCTUnwrap(root as? AnalyzeMusicCommand)
        XCTAssertEqual(command.input, "Your Song.m4a")
        XCTAssertEqual(command.start, 12.5)
        XCTAssertEqual(command.maxSeconds, 30)
        XCTAssertTrue(command.json)
    }

    func testRejectsInvalidWindowsBeforeReadingAFile() {
        for options in [
            ["--start", "-1"], ["--start", "nan"], ["--start", "inf"],
            ["--max-seconds", "0"], ["--max-seconds", "-1"],
            ["--max-seconds", "121"], ["--max-seconds", "nan"], ["--max-seconds", "inf"]
        ] {
            XCTAssertThrowsError(try AudioCLI.parseAsRoot(["analyze-music", "missing.wav"] + options)) { error in
                XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
            }
        }
    }
}

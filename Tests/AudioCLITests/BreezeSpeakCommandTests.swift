import ArgumentParser
@testable import AudioCLILib
import XCTest

final class BreezeSpeakCommandTests: XCTestCase {
    func testDesignAndStreamingFlags() throws {
        let command = try SpeakCommand.parse(["Hello", "--engine", "breeze", "--instruct", "A calm voice",
            "--stream", "--breeze-cfg-scale", "4", "--breeze-seed", "0", "--breeze-model-id", "owner/bundle"])
        try command.validate()
        XCTAssertTrue(command.stream)
        XCTAssertEqual(command.breezeSeed, 0)
        XCTAssertEqual(command.breezeModelId, "owner/bundle")
    }

    func testCloneRequiresExactTranscriptPair() throws {
        for args in [["--voice-sample", "ref.wav"], ["--breeze-reference-text", "Reference"],
                     ["--voice-sample", "ref.wav", "--breeze-reference-text", " "]] {
            XCTAssertThrowsError(try SpeakCommand.parse(["Hello", "--engine", "breeze"] + args).validate())
        }
        try SpeakCommand.parse(["Hello", "--engine", "breeze", "--voice-sample", "ref.wav",
            "--breeze-reference-text", "Reference speech.", "--instruct", "Speak warmly"]).validate()
    }

    func testCacheBudgetIsOptionalAndValidatedBeforeLoading() throws {
        XCTAssertNil(try SpeakCommand.parse(["Hello", "--engine", "breeze"]).breezeCacheLimitMiB)
        for budget in [0, 1024] {
            let command = try SpeakCommand.parse(["Hello", "--engine", "breeze", "--breeze-cache-limit-mib", String(budget)])
            try command.validate()
            XCTAssertEqual(command.breezeCacheLimitMiB, budget)
        }
        for budget in [-1, Int.max] {
            XCTAssertThrowsError(try SpeakCommand.parse(["Hello", "--engine", "breeze",
                "--breeze-cache-limit-mib", String(budget)]).validate())
        }
    }

    func testInvalidControlsAndUnsupportedOptions() throws {
        for args in [["--breeze-top-p", "0"], ["--breeze-temperature", "nan"], ["--breeze-cfg-scale", "0"],
                     ["--breeze-max-frames", "0"], ["--first-chunk-frames", "0"], ["--language", "de"],
                     ["--list-speakers"], ["--speaker", "alice"], ["--batch-file", "lines.txt"], ["--clean-reference"]] {
            XCTAssertThrowsError(try SpeakCommand.parse(["Hello", "--engine", "breeze"] + args).validate())
        }
    }
}

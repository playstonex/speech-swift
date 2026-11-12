import ArgumentParser
@testable import AudioCLILib
import XCTest

final class InstructionTTSSpeakCommandTests: XCTestCase {
    func testExplicitBundlesAndSeed() throws {
        for engine in ["fireredtts3", "moss-voicegenerator"] {
            let command = try SpeakCommand.parse(["Hello.", "--engine", engine, "--instruct", "A calm voice.",
                "--\(engine)-bundle-dir", "/tmp/bundle", "--seed", "0"])
            try command.validate()
            XCTAssertEqual(command.seed, 0)
        }
    }

    func testRejectsMissingOrAmbiguousBundleAndUnsupportedControls() throws {
        for engine in ["fireredtts3", "moss-voicegenerator"] {
            let base = ["Hello.", "--engine", engine, "--instruct", "A calm voice."]
            XCTAssertThrowsError(try SpeakCommand.parse(base).validate())
            for extra in [["--\(engine)-model-id", "owner/bundle"], ["--voice-sample", "ref.wav"],
                          ["--speaker", "ryan"], ["--clean-reference"], ["--batch-file", "text.txt"]] {
                XCTAssertThrowsError(try SpeakCommand.parse(base + ["--\(engine)-bundle-dir", "/tmp/bundle"] + extra).validate())
            }
            XCTAssertThrowsError(try SpeakCommand.parse(["Hello.", "--engine", engine,
                "--\(engine)-bundle-dir", "/tmp/bundle"]).validate())
        }
    }

    func testStreamingAndLimits() throws {
        let moss = ["Hello.", "--engine", "moss-voicegenerator", "--instruct", "A calm voice.",
                    "--moss-voicegenerator-model-id", "owner/bundle"]
        try SpeakCommand.parse(moss + ["--stream"]).validate()
        for extra in [["--moss-voicegenerator-max-frames", "0"], ["--moss-voicegenerator-chunk-frames", "33"], ["--language", "en"]] {
            XCTAssertThrowsError(try SpeakCommand.parse(moss + extra).validate())
        }
        let fireRed = ["Hello.", "--engine", "fireredtts3", "--instruct", "A calm voice.",
                       "--fireredtts3-model-id", "owner/bundle"]
        for extra in [["--stream"], ["--fireredtts3-flow-steps", "0"], ["--fireredtts3-max-patches", "0"], ["--language", "de"]] {
            XCTAssertThrowsError(try SpeakCommand.parse(fireRed + extra).validate())
        }
    }
}

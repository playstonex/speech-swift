import ArgumentParser
import XCTest
@testable import AudioCLILib

final class MossNanoCommandTests: XCTestCase {
    func testCloningStreamingArguments() throws {
        let command = try XCTUnwrap(AudioCLI.parseAsRoot(["speak", "Hello", "--engine", "moss-nano", "--voice-sample", "reference.wav",
            "--stream", "--moss-nano-model", "/tmp/model", "--moss-nano-compute-units", "gpu", "--moss-nano-max-frames", "120", "--moss-nano-seed", "7"]) as? SpeakCommand)
        XCTAssertEqual(command.mossNanoModel, "/tmp/model")
        XCTAssertEqual(command.mossNanoComputeUnits, "gpu")
        XCTAssertEqual(command.mossNanoMaxFrames, 120)
        XCTAssertEqual(command.mossNanoSeed, 7)
        XCTAssertTrue(command.stream)
        XCTAssertEqual(command.voiceSample, "reference.wav")
    }

    func testDefaultsAndUnsupportedOptions() throws {
        let command = try XCTUnwrap(AudioCLI.parseAsRoot(["speak", "Hello", "--engine", "moss-nano"]) as? SpeakCommand)
        XCTAssertEqual(command.mossNanoComputeUnits, "cpu")
        XCTAssertEqual(command.mossNanoMaxFrames, 300)
        for extra in [["--moss-nano-compute-units", "ane"], ["--moss-nano-max-frames", "0"],
                      ["--speaker", "alice"], ["--instruct", "happy"], ["--language", "english"],
                      ["--batch-file", "texts.txt"], ["--clean-reference"]] {
            XCTAssertThrowsError(try AudioCLI.parseAsRoot(["speak", "Hello", "--engine", "moss-nano"] + extra))
        }
    }
}

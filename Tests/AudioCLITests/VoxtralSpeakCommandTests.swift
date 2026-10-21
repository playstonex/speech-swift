import XCTest
@testable import AudioCLILib

final class VoxtralSpeakCommandTests: XCTestCase {
    func testDefaultsAndExplicitVoiceOptions() throws {
        let command = try SpeakCommand.parse(["Hello.", "--engine", "voxtral", "--voxtral-voice", "fr_female",
                                            "--voxtral-bundle-dir", "/tmp/voxtral", "--voxtral-seed", "42"])
        XCTAssertEqual(command.voxtralVoice, "fr_female")
        XCTAssertEqual(command.voxtralBundleDir, "/tmp/voxtral")
        XCTAssertEqual(command.voxtralSeed, 42)
        XCTAssertEqual(command.voxtralMaxFrames, 500)
        XCTAssertEqual(command.voxtralModelId, "aufklarer/Voxtral-4B-TTS-2603-MLX-4bit")
    }

    func testListSpeakersNeedsNoTextOrDownloadedModel() throws {
        let command = try SpeakCommand.parse(["--engine", "voxtral", "--list-speakers"])
        XCTAssertTrue(command.listSpeakers)
    }

    func testRejectsUnsupportedModesAndInvalidInputsBeforeLoading() {
        for flags in [["--batch-file", "batch.txt"], ["--voice-sample", "reference.wav"], ["--speaker", "alice"],
                      ["--instruct", "happy"], ["--voxtral-voice", "missing"],
                      ["--voxtral-max-frames", "0"], ["--voxtral-chunk-frames", "0"],
                      ["--voxtral-chunk-frames", "126"], ["--language", "japanese"]] {
            XCTAssertThrowsError(try SpeakCommand.parse(["Hello.", "--engine", "voxtral"] + flags), "flags=\(flags)")
        }
        XCTAssertThrowsError(try SpeakCommand.parse([" ", "--engine", "voxtral"]))
    }

    func testStreamingAndChunkSizeAreAccepted() throws {
        let command = try SpeakCommand.parse(["Hello.", "--engine", "voxtral", "--stream",
                                             "--voxtral-chunk-frames", "1"])
        XCTAssertTrue(command.stream)
        XCTAssertEqual(command.voxtralChunkFrames, 1)
    }

    func testExistingDefaultEngineIsUnchanged() throws {
        let command = try SpeakCommand.parse(["Hello."])
        XCTAssertEqual(command.engine, "qwen3")
        XCTAssertEqual(command.model, "base")
    }
}

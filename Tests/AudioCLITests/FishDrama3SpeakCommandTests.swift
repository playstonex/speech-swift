import XCTest
@testable import AudioCLILib

final class FishDrama3SpeakCommandTests: XCTestCase {
    func testDefaultAndLocalBundleOptions() throws {
        let command = try SpeakCommand.parse(["Hello.", "--engine", "fish-drama3", "--fish-drama3-bundle-dir", "/tmp/drama3", "--seed", "42"])
        XCTAssertEqual(command.fishDrama3ModelId, "mlx-community/fish-audio-drama-3-4bit")
        XCTAssertEqual(command.fishDrama3BundleDir, "/tmp/drama3")
        XCTAssertEqual(command.fishDrama3MaxFrames, 2048)
        XCTAssertEqual(command.fishDrama3Temperature, 1)
        XCTAssertEqual(command.fishDrama3MemoryMode, "balanced")
        XCTAssertEqual(command.seed, 42)
    }

    func testCloningRequiresAudioAndExactTranscript() throws {
        XCTAssertNoThrow(try SpeakCommand.parse(["Hello.", "--engine", "fish-drama3", "--voice-sample", "reference.wav",
                                                "--fish-drama3-reference-text", "The reference words."]))
        XCTAssertThrowsError(try SpeakCommand.parse(["Hello.", "--engine", "fish-drama3", "--voice-sample", "reference.wav"]))
        XCTAssertThrowsError(try SpeakCommand.parse(["Hello.", "--engine", "fish-drama3", "--fish-drama3-reference-text", "Reference."]))
    }

    func testRejectsUnsupportedModesAndInvalidTextBeforeLoading() {
        for flags in [["--batch-file", "batch.txt"], ["--list-speakers"], ["--speaker", "alice"],
                      ["--instruct", "happy"], ["--fish-drama3-max-frames", "0"], ["--fish-drama3-top-k", "0"],
                      ["--fish-drama3-top-p", "1.1"], ["--fish-drama3-temperature", "nan"]] {
            XCTAssertThrowsError(try SpeakCommand.parse(["Hello.", "--engine", "fish-drama3"] + flags), "\(flags)")
        }
        XCTAssertThrowsError(try SpeakCommand.parse(["<|speaker:16|> Hello.", "--engine", "fish-drama3"]))
    }

    func testStreamingAcceptsBoundedChunkSizesAndCloning() throws {
        let command = try SpeakCommand.parse(["Hello.", "--engine", "fish-drama3", "--stream"])
        XCTAssertTrue(command.stream)
        XCTAssertEqual(command.fishDrama3ChunkFrames, 3)
        XCTAssertNoThrow(try SpeakCommand.parse(["Hello.", "--engine", "fish-drama3", "--stream", "--fish-drama3-chunk-frames", "1",
            "--voice-sample", "reference.wav", "--fish-drama3-reference-text", "The reference words."]))
        for count in ["0", "129", "-1"] {
            XCTAssertThrowsError(try SpeakCommand.parse(["Hello.", "--engine", "fish-drama3", "--stream", "--fish-drama3-chunk-frames", count]))
        }
    }

    func testMemoryPolicyCanOptOutAndRejectsUnknownValues() throws {
        let command = try SpeakCommand.parse(["Hello.", "--engine", "fish-drama3", "--fish-drama3-memory-mode", "unrestricted"])
        XCTAssertEqual(command.fishDrama3MemoryMode, "unrestricted")
        XCTAssertThrowsError(try SpeakCommand.parse(["Hello.", "--engine", "fish-drama3", "--fish-drama3-memory-mode", "unknown"]))
    }
}

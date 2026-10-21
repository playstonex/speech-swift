import XCTest
@testable import AudioCLILib

final class PocketSpeakCommandTests: XCTestCase {
    func testStreamingLocalBundleAndSeedZero() throws {
        let command = try SpeakCommand.parse(["Hello.", "--engine", "pocket", "--stream", "--pocket-bundle-dir", "/tmp/pocket", "--pocket-seed", "0"])
        XCTAssertTrue(command.stream)
        XCTAssertEqual(command.pocketSeed,0)
        XCTAssertEqual(command.pocketBundleDir,"/tmp/pocket")
        XCTAssertEqual(command.pocketMaxFrames,350)
    }
    func testListPresetNeedsNoTextOrWeights() throws {
        XCTAssertTrue(try SpeakCommand.parse(["--engine","pocket","--list-speakers"]).listSpeakers)
    }
    func testUnsupportedInputsFailBeforeLoading() {
        for flags in [["--language","french"],["--voice-sample","a.wav"],["--batch-file","a.txt"],
                      ["--speaker","other"],["--pocket-max-frames","0"]] {
            XCTAssertThrowsError(try SpeakCommand.parse(["Hello.","--engine","pocket"]+flags))
        }
        XCTAssertThrowsError(try SpeakCommand.parse([" ","--engine","pocket"]))
    }
    func testDefaultEngineRemainsQwen3() throws {
        XCTAssertEqual(try SpeakCommand.parse(["Hello."]).engine,"qwen3")
    }
}

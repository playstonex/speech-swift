import AudioCommon
import Foundation
import XCTest

final class E2EFireRedTTS3CLITests: XCTestCase {
    func testExplicitLocalBundleProducesCompleteWAV() throws {
        let env = ProcessInfo.processInfo.environment
        guard let binary = env["SPEECH_CLI"], let bundle = env["FIREREDTTS3_E2E_BUNDLE"] else {
            throw XCTSkip("Set SPEECH_CLI and FIREREDTTS3_E2E_BUNDLE; this test never downloads weights")
        }
        let output = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: output) }
        let script = "The weather is beautiful today."
        let process = Process(), pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: binary)
        process.arguments = ["speak", script, "--engine", "fireredtts3",
            "--fireredtts3-bundle-dir", bundle, "--instruct", "A calm, clear female voice speaking warmly.",
            "--seed", "2", "--fireredtts3-max-patches", "150", "--output", output.path]
        process.standardOutput = pipe; process.standardError = pipe
        try process.run()
        let log = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0, log)
        XCTAssertFalse(process.isRunning)
        try InstructionTTSCLITestSupport.assertEnglishContent(binary: binary, wav: output,
            expectedText: script, expectedWordCount: 5, synthesisLog: log)
        let samples = try AudioFileLoader.load(url: output, targetSampleRate: 24000)
        XCTAssertGreaterThan(samples.count, 12000)
        XCTAssertTrue(samples.count.isMultiple(of: 3840))
        XCTAssertTrue(samples.allSatisfy(\.isFinite))
        XCTAssertTrue(log.contains("Complete PCM:"), log)
        XCTAssertTrue(log.contains("Saved audio"), log)
    }
}

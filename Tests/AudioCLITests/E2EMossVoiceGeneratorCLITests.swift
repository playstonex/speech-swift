import AudioCommon
import Foundation
import XCTest

final class E2EMossVoiceGeneratorCLITests:XCTestCase {
    func testExplicitLocalBundleStreamingCLIProducesCompleteWAV()throws {
        let env=ProcessInfo.processInfo.environment
        guard let binary=env["SPEECH_CLI"],let bundle=env["MOSS_VOICEGEN_BUNDLE"] else {
            throw XCTSkip("Set SPEECH_CLI and MOSS_VOICEGEN_BUNDLE; this test never downloads weights")
        }
        let output=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".wav")
        defer {try? FileManager.default.removeItem(at:output)}
        let process=Process(),pipe=Pipe()
        process.executableURL=URL(fileURLWithPath:binary)
        process.arguments=["speak","The little boat crossed the quiet lake before sunset.",
            "--engine","moss-voicegenerator","--moss-voicegenerator-bundle-dir",bundle,
            "--instruct","A warm adult female voice speaking clearly at a natural pace.",
            "--stream","--seed","42","--moss-voicegenerator-max-frames","150","--output",output.path]
        process.standardOutput=pipe;process.standardError=pipe
        try process.run()
        let log=String(decoding:pipe.fileHandleForReading.readDataToEndOfFile(),as:UTF8.self)
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus,0,log)
        let samples=try AudioFileLoader.load(url:output,targetSampleRate:24000)
        XCTAssertGreaterThan(samples.count,24000)
        XCTAssertTrue(samples.count.isMultiple(of:1920))
        XCTAssertTrue(samples.allSatisfy(\.isFinite))
        XCTAssertTrue(log.contains("First PCM:"),log)
        XCTAssertTrue(log.contains("Saved audio"),log)
        // The speech process has exited before Parakeet starts. Correct finite
        // PCM alone cannot detect a spoken instruction or missing final word.
        XCTAssertFalse(process.isRunning)
        try InstructionTTSCLITestSupport.assertEnglishContent(binary:binary,wav:output,
            expectedText:"The little boat crossed the quiet lake before sunset.",
            expectedWordCount:9,synthesisLog:log)
    }
}

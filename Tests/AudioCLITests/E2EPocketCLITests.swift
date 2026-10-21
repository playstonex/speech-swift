import Foundation
import XCTest

final class E2EPocketCLITests: XCTestCase {
    func testNativeCLIStreamingProducesCompleteWAV() throws {
        guard let binary = ProcessInfo.processInfo.environment["SPEECH_CLI"],
              let bundle = ProcessInfo.processInfo.environment["POCKET_TTS_BUNDLE"] else {
            throw XCTSkip("Set SPEECH_CLI and POCKET_TTS_BUNDLE for the CLI round trip")
        }
        let output = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".wav")
        defer { try? FileManager.default.removeItem(at: output) }
        let process = Process(); let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: binary)
        process.arguments = ["speak","Hello world.","--engine","pocket","--stream","--pocket-bundle-dir",bundle,"--pocket-seed","42","-o",output.path]
        process.standardOutput = pipe; process.standardError = pipe
        try process.run()
        let log = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus,0,log)
        let data = try Data(contentsOf: output)
        XCTAssertEqual(String(decoding: data.prefix(4),as: UTF8.self),"RIFF")
        XCTAssertGreaterThan(data.count,24000)
        XCTAssertTrue(log.contains("First audio:"),log)
    }
}

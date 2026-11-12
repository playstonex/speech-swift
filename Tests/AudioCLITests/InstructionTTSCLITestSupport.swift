import Foundation
import XCTest

/// Shared content check for instruction-TTS CLI tests. Call only after the
/// synthesis child has exited, so ASR and synthesis never hold models together.
enum InstructionTTSCLITestSupport {
    struct ChildResult {
        let status: Int32
        let log: String
    }

    static func run(binary: String, arguments: [String]) throws -> ChildResult {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: binary)
        process.arguments = arguments
        // CLI JSON may be written to either stream. One pipe also avoids a
        // child filling an unread stderr pipe while its stdout is consumed.
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let output = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return ChildResult(status: process.terminationStatus,
                           log: String(decoding: output, as: UTF8.self))
    }

    /// Punctuation/case normalization for the fixed alphabetic English scripts.
    /// These tests deliberately contain no numbers requiring word expansion.
    static func normalizedEnglishWords(_ text: String) -> [String] {
        text.lowercased()
            .replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "[^\\w\\s']", with: " ", options: .regularExpression)
            .split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
    }

    /// Runs a new, independent Parakeet process and checks every spoken word.
    /// JSON lines are preferred; the per-clip text output is a robust fallback.
    @discardableResult
    static func assertEnglishContent(
        binary: String, wav: URL, expectedText: String, expectedWordCount: Int = 9,
        synthesisLog: String = "", file: StaticString = #filePath, line: UInt = #line
    ) throws -> String {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("instruction-tts-asr-" + UUID().uuidString)
        let input = root.appendingPathComponent("audio")
        let output = root.appendingPathComponent("text")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let stem = "clip"
        try FileManager.default.createSymbolicLink(
            at: input.appendingPathComponent(stem + ".wav"), withDestinationURL: wav)

        let result = try run(binary: binary, arguments: [
            "transcribe-batch", input.path, "--engine", "parakeet", "--jsonl",
            "--output-dir", output.path
        ])
        let diagnostic = "Synthesis log:\n\(synthesisLog)\nIndependent Parakeet log:\n\(result.log)"
        XCTAssertEqual(result.status, 0, diagnostic, file: file, line: line)
        guard result.status == 0 else { return "" }

        var transcript: String?
        for raw in result.log.split(whereSeparator: { $0.isNewline }) {
            let text = String(raw).trimmingCharacters(in: .whitespacesAndNewlines)
            guard text.hasPrefix("{"), let bytes = text.data(using: .utf8),
                  let row = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any],
                  let name = row["file"] as? String,
                  URL(fileURLWithPath: name).deletingPathExtension().lastPathComponent == stem else { continue }
            if let value = row["text"] as? String { transcript = value; break }
        }
        if transcript == nil {
            transcript = try? String(contentsOf: output.appendingPathComponent(stem + ".txt"), encoding: .utf8)
        }
        guard let transcript, !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            XCTFail("Parakeet produced no transcript.\n\(diagnostic)", file: file, line: line)
            return ""
        }
        let expected = normalizedEnglishWords(expectedText)
        XCTAssertEqual(expected.count, expectedWordCount,
                       "The fixed content-test script changed.", file: file, line: line)
        XCTAssertEqual(normalizedEnglishWords(transcript), expected,
                       "Transcript: \(transcript)\n\(diagnostic)", file: file, line: line)
        return transcript
    }
}

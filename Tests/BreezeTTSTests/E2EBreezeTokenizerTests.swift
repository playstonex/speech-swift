@testable import BreezeTTS
import Hub
import Tokenizers
import XCTest

final class E2EBreezeTokenizerTests: XCTestCase {
    func testPublishedControlsAndPlaceholderFallbackMatchFullTokenizer() async throws {
        guard let path = ProcessInfo.processInfo.environment["BREEZE_E2E_BUNDLE"] else {
            throw XCTSkip("Set BREEZE_E2E_BUNDLE to compare the complete exported tokenizer")
        }
        let directory = URL(fileURLWithPath: path)
        let original = try await AutoTokenizer.from(modelFolder: directory)
        let compact = try await BreezeTextTokenizer.fromBundle(directory)
        let data = try HubApi.shared.configuration(fileURL: directory.appendingPathComponent("tokenizer.json"))
        let tokens = data["added_tokens"].array(or: []).compactMap { $0.content.string() }
        // Cover every retained control, including newlines, literal SentencePiece
        // spaces and speaker/instruction tags, plus representative full fallbacks.
        let controls = tokens.filter { !$0.hasPrefix("<unused") }
        let unused = tokens.filter { $0.hasPrefix("<unused") }
        let placeholders = unused.isEmpty ? [] : [unused[0], unused[unused.count / 2], unused[unused.count - 1]]
        for selected in [controls, placeholders] {
            for start in stride(from: 0, to: selected.count, by: 64) {
                let text = selected[start..<min(start + 64, selected.count)].joined(separator: "abc ")
                try autoreleasepool {
                    XCTAssertEqual(try compact.encode(text), original.encode(text: text))
                }
            }
        }
        for text in ["[S0]A different sentence, with 123 numbers and punctuation!",
                     "[S0]<ins_bos>温柔的女声。<ins_eos>今天下午三点，会议开始。",
                     "  Leading spaces\n\nnewlines\tand café e\u{301} 😀 ▁▁",
                     "[S0]<unused999999>literal unknown placeholder", "<unused"] {
            XCTAssertEqual(try compact.encode(text), original.encode(text: text), text)
        }
    }
}

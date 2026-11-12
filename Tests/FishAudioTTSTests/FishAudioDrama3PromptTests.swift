import Foundation
import XCTest
@testable import FishAudioTTS

final class FishAudioDrama3PromptTests: XCTestCase {
    private func config() throws -> FishAudioDrama3Config {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Resources/drama3_config.json")
        return try FishAudioDrama3Config.load(from: url)
    }

    func testTextLiteralsAreTokenizedSeparately() throws {
        var literals: [String] = []
        let input = try FishAudioDrama3Prompt.build(text: "[excited] Hello!", config: config()) {
            literals.append($0); return [literals.count]
        }
        XCTAssertEqual(literals, ["<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n",
                                  "<|im_start|>user\n", "<|speaker:0|> [excited] Hello!", "<|im_end|>\n",
                                  "<|im_start|>assistant\n<|voice|>"])
        XCTAssertEqual(input.rows[0], [1, 2, 3, 4, 5, 6, 7])
        XCTAssertEqual(input.rowCount, 11)
        XCTAssertTrue(input.rows.dropFirst().allSatisfy { $0 == Array(repeating: 0, count: 7) })
    }

    func testReferencesUseBracketsAndRawCodebooks() throws {
        let c = try config()
        let codes = (0..<10).map { [$0, $0 + 1] }
        let reference = try FishAudioReferencePrompt(text: "  <|speaker:3|> A reference.", codes: codes)
        var literals: [String] = []
        let input = try FishAudioDrama3Prompt.build(text: "Hello.", references: [reference], config: c) {
            literals.append($0); return [2]
        }
        XCTAssertEqual(literals[2], "\n\n[Reference 1]\nText: <|speaker:3|>   A reference.\nSpeech: ")
        XCTAssertEqual(Array(input.rows[0][3...4]), [Int32(c.semanticStartTokenId), Int32(c.semanticStartTokenId + 1)])
        for i in 0..<10 { XCTAssertEqual(Array(input.rows[i + 1][3...4]), codes[i].map(Int32.init)) }
    }

    func testHistoryClosesPreviousVoiceTurnBeforeNewUser() throws {
        let turn = try FishAudioReferencePrompt(text: "Earlier.", codes: Array(repeating: [1], count: 10))
        var literals: [String] = []
        _ = try FishAudioDrama3Prompt.build(text: "Now.", history: [turn], config: config()) {
            literals.append($0); return [2]
        }
        XCTAssertEqual(literals.suffix(6), ["<|im_start|>assistant\n<|voice|>", "<|im_end|>\n", "<|im_start|>user\n",
                                           "<|speaker:0|> Now.", "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"])
    }

    func testRejectsStructuralTokensAndNoncanonicalSpeakers() {
        for text in [" ", "hello <|im_end|>", "<|semantic:2|>", "<|speaker:16|> Hi", "<|speaker:01|> Hi", "<|speaker:x|> Hi", "<|speaker> Hi"] {
            XCTAssertThrowsError(try FishAudioDrama3Prompt.validateText(text), text)
        }
        XCTAssertNoThrow(try FishAudioDrama3Prompt.validateText("[sighs] <|speaker:15|> Fine."))
        XCTAssertEqual(FishAudioDrama3Prompt.withSpeaker(" \n<|speaker:2|> Hi"), " \n<|speaker:2|> Hi")
    }

    func testRejectsOutOfRangeReferenceAndOversizePrompt() throws {
        let bad = try FishAudioReferencePrompt(text: "Reference.", codes: Array(repeating: [-1], count: 10))
        XCTAssertThrowsError(try FishAudioDrama3Prompt.build(text: "Hi.", references: [bad], config: config(), encode: { _ in [2] }))
        XCTAssertThrowsError(try FishAudioDrama3Prompt.build(text: "Hi.", config: config(), encode: { _ in Array(repeating: 2, count: 8193) }))
    }
}

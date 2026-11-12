import XCTest
@testable import KokoroTTS

/// Mirrors speech-core's `test_text_chunker.cpp`. Token counts use the same
/// stand-in: one per non-space character plus two for BOS/EOS.
final class KokoroTextChunkerTests: XCTestCase {

    private func count(_ text: String) -> Int {
        2 + text.filter { $0 != " " && $0 != "\t" }.count
    }

    private func chunk(_ text: String, budget: Int, hardCap: Int = 128, minTail: Int = 12) -> [String] {
        KokoroTextChunker.chunk(text, countTokens: count, maxTokens: budget,
                                hardCapTokens: hardCap, minTailTokens: minTail)
    }

    private func retry(_ text: String, min: Int, max: Int) -> [String] {
        KokoroTextChunker.splitForRetry(text, countTokens: count, minTokens: min, maxTokens: max)
    }

    // MARK: - Synthesizer plan

    func testPlanKeepsShortTextWhole() {
        let text = "Hello there. How are you today?"
        XCTAssertEqual(KokoroTTSModel.chunkPlan(text, countTokens: count), [text])
    }

    func testPlanMergesShortTailWithinHardCap() {
        let a = "I moved the planning meeting with the design team to Thursday."
        XCTAssertEqual(count(a), 54)
        // 54 + "See you." is 61 tokens: past the 56 budget, within the 64 cap.
        XCTAssertEqual(KokoroTTSModel.chunkPlan(a + " See you.", countTokens: count), [a + " See you."])
    }

    func testPlanKeepsTailWhenMergeExceedsHardCap() {
        let a = "I moved the planning meetings with the design teams to Thursday."
        XCTAssertEqual(count(a), 56)
        // Merged with "Thank you." it would be 65 tokens, past the 64 cap.
        XCTAssertEqual(KokoroTTSModel.chunkPlan(a + " Thank you.", countTokens: count), [a, "Thank you."])
    }

    // MARK: - Packing

    func testEmptyAndWhitespace() {
        XCTAssertEqual(chunk("", budget: 72), [])
        XCTAssertEqual(chunk("   \n\t  ", budget: 72), [])
    }

    func testShortTextIsOneChunk() {
        XCTAssertEqual(chunk("Music stopped.", budget: 72), ["Music stopped."])
    }

    func testSentencesPackTogether() {
        XCTAssertEqual(chunk("Calling Anna. Please wait.", budget: 72), ["Calling Anna. Please wait."])
    }

    func testSplitsOnSentenceBoundary() {
        let a = "The first sentence is fairly long already."
        let b = "The second sentence is also fairly long."
        XCTAssertEqual(chunk(a + " " + b, budget: count(a) + 4), [a, b])
    }

    func testBudgetRespected() {
        let text = "I can call your contacts, dial numbers, look up phone numbers, "
            + "play or stop music, and set the volume, all offline, on this "
            + "device. Ask me anything you like. There is more where that came from!"
        let chunks = chunk(text, budget: 72)
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        for piece in chunks {
            XCTAssertLessThanOrEqual(count(piece), 72)
            XCTAssertFalse(piece.isEmpty)
        }
    }

    func testClauseFallbackForRunOnSentence() {
        let text = "first clause goes here, second clause goes here, third clause goes here, fourth clause goes here"
        let chunks = chunk(text, budget: 30)
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        for piece in chunks { XCTAssertLessThanOrEqual(count(piece), 30) }
    }

    func testWordFallback() {
        let chunks = chunk("one two three four five six seven eight nine ten", budget: 12, minTail: 3)
        XCTAssertGreaterThanOrEqual(chunks.count, 3)
        for piece in chunks { XCTAssertLessThanOrEqual(count(piece), 12) }
    }

    func testLongSentencePreservesWholeWords() {
        let text = "The new short-turn graph is ready for quick replies."
        let chunks = chunk(text, budget: 14, hardCap: 14, minTail: 0)
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        for piece in chunks { XCTAssertLessThanOrEqual(count(piece), 14) }
        XCTAssertEqual(chunks.joined(separator: " "), text)
        XCTAssertTrue(chunks.contains("short-turn"))
    }

    func testMonsterWordCharacterSplit() {
        let word = String(repeating: "x", count: 50)
        let chunks = chunk(word, budget: 12, minTail: 3)
        XCTAssertGreaterThanOrEqual(chunks.count, 4)
        for piece in chunks { XCTAssertLessThanOrEqual(count(piece), 12) }
        XCTAssertEqual(chunks.joined(), word)
    }

    func testCombiningMarksStayWithTheirLetter() {
        // "é" spelled as e + U+0301 is one Character; a cut must never strand
        // the accent at the start of a chunk.
        let word = String(repeating: "e\u{301}中", count: 12)
        let chunks = chunk(word, budget: 10, minTail: 2)
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        for piece in chunks {
            XCTAssertNotEqual(piece.unicodeScalars.first, "\u{301}")
        }
        XCTAssertEqual(chunks.joined(), word)
    }

    func testTinyTailMerges() {
        let a = "This is a reasonably sized first sentence here."
        let b = "Ok."
        XCTAssertEqual(chunk(a + " " + b, budget: count(a) + 2), [a + " " + b])
    }

    func testTinyTailKeptWhenHardCapBlocks() {
        let a = "This is a reasonably sized first sentence here."
        let b = "Ok."
        let budget = count(a) + 2
        XCTAssertEqual(chunk(a + " " + b, budget: budget, hardCap: budget), [a, b])
    }

    func testNewlinesAreBoundaries() {
        XCTAssertEqual(chunk("Line one has words\nLine two has words", budget: 18).count, 2)
        XCTAssertEqual(chunk("Line one has words\r\nLine two has words", budget: 18).count, 2)
    }

    // MARK: - Inner punctuation

    func testInnerPunctuationIsNotABoundary() {
        // Decimals, versions, domains, times, dates and German number formats
        // must reach the phonemizer exactly as written.
        let texts = [
            "It costs 3.5 euros.",
            "Version 1.2.3 is live at soniqo.audio today.",
            "Der Zug kommt um 18:30 Uhr, nicht um 19.15 Uhr.",
            "Es kostet 1.000 Euro, z.B. heute.",
            "Am 03.10.2026 kostet es 12,80 Euro.",
        ]
        for text in texts {
            XCTAssertEqual(chunk(text, budget: 72), [text])
        }
    }

    func testClauseSplitKeepsNumbersWhole() {
        let chunks = chunk("Um 18:30 Uhr kostet das Ticket 12,80 Euro, danach kostet es 15,20 Euro", budget: 40)
        XCTAssertEqual(chunks, ["Um 18:30 Uhr kostet das Ticket 12,80 Euro,", "danach kostet es 15,20 Euro"])
    }

    // MARK: - Retry split

    func testRetrySplitProgressFor13To42Tokens() {
        for parent in 13...42 {
            let word = String(repeating: "x", count: parent - 2)
            let pieces = retry(word, min: 6, max: parent - 1)
            XCTAssertEqual(pieces.count, 2, "parent \(parent)")
            for piece in pieces {
                XCTAssertFalse(piece.isEmpty)
                XCTAssertGreaterThanOrEqual(count(piece), 6)
                XCTAssertLessThan(count(piece), parent)
            }
            XCTAssertEqual(pieces.joined(), word)
        }
    }

    func testRetrySplitPrefersWordBoundaryWhenBalanced() {
        XCTAssertEqual(retry("alpha bravo", min: 6, max: 12), ["alpha", "bravo"])
    }

    func testRetrySplitDoesNotTradeWordBoundaryForBalance() {
        XCTAssertEqual(retry("alpha bravo charlie", min: 6, max: 16), ["alpha bravo", "charlie"])
    }

    func testRetrySplitKeepsPunctuationOnLeft() {
        XCTAssertEqual(retry("aaaaa. bbbbb", min: 6, max: 12), ["aaaaa.", "bbbbb"])
    }

    func testRetrySplitKeepsClosingQuoteWithSentence() {
        XCTAssertEqual(retry("aaaaa.\" bbbbb", min: 6, max: 13), ["aaaaa.\"", "bbbbb"])
    }

    func testRetrySplitKeepsFullSentenceEnderRun() {
        XCTAssertEqual(retry("aaaaa...?!\" bbbbb", min: 6, max: 18), ["aaaaa...?!\"", "bbbbb"])
    }

    func testRetrySplitDoesNotCutInsideANumber() {
        let pieces = retry("aaaaa 3.5 bbbbb", min: 6, max: 13)
        XCTAssertEqual(pieces.count, 2)
        XCTAssertEqual(pieces.joined(separator: " "), "aaaaa 3.5 bbbbb")
    }

    func testRetrySplitImpossibleReturnsEmpty() {
        XCTAssertEqual(retry("tiny", min: 6, max: 6), [])
    }
}

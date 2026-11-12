import Foundation

/// Splits text into pieces the fixed-capacity Kokoro graph can render.
///
/// Mirrors speech-core's `chunk_text_for_synthesis` and
/// `split_text_for_synthesis_retry`, so Apple and Android cut the same text
/// in the same places. Pieces are packed greedily up to a token budget,
/// preferring sentence boundaries, then clause boundaries (`,` `;` `:`), then
/// words, then characters for a single oversized word. Punctuation is a
/// boundary only when whitespace or the end of the text follows it, so
/// "3.5", "1.000", "z.B." or "18:30" reach the phonemizer intact.
enum KokoroTextChunker {

    /// Packs `text` into chunks of at most `maxTokens`. A last chunk below
    /// `minTailTokens` merges into its predecessor when the result stays
    /// within `hardCapTokens`, the model's absolute input capacity.
    /// Whitespace-only text yields no chunks.
    static func chunk(
        _ text: String,
        countTokens: (String) -> Int,
        maxTokens: Int,
        hardCapTokens: Int,
        minTailTokens: Int
    ) -> [String] {
        guard maxTokens > 0 else { return [] }
        var chunks: [String] = []
        pack(splitUnits(text, isEnd: isSentenceEnd), countTokens: countTokens,
             maxTokens: maxTokens, level: 0, into: &chunks)
        if chunks.count >= 2, countTokens(chunks[chunks.count - 1]) < minTailTokens {
            let merged = chunks[chunks.count - 2] + " " + chunks[chunks.count - 1]
            if countTokens(merged) <= hardCapTokens {
                chunks.removeLast()
                chunks[chunks.count - 1] = merged
            }
        }
        return chunks
    }

    /// Two strictly smaller pieces of a chunk the model could not render
    /// safely, or `[]` when no cut keeps both within `minTokens...maxTokens`.
    /// The boundary class (sentence, clause, word, character) outranks
    /// balance; a cut inside a punctuation run or before a closing quote is
    /// never taken.
    static func splitForRetry(
        _ text: String,
        countTokens: (String) -> Int,
        minTokens: Int,
        maxTokens: Int
    ) -> [String] {
        guard minTokens > 0, minTokens <= maxTokens else { return [] }
        let chars = Array(text)
        guard chars.count >= 2 else { return [] }
        // Every sentence, clause and word boundary touches whitespace, and any
        // of them beats a character cut, so character cuts are only tried
        // when no boundary works. This keeps the phonemizer calls per retry
        // proportional to the number of words.
        let boundaries = (1..<chars.count).filter {
            chars[$0 - 1].isWhitespace || chars[$0].isWhitespace
        }
        if let pieces = bestSplit(chars, cuts: boundaries, countTokens: countTokens,
                                  minTokens: minTokens, maxTokens: maxTokens) {
            return pieces
        }
        return bestSplit(chars, cuts: Array(1..<chars.count), countTokens: countTokens,
                         minTokens: minTokens, maxTokens: maxTokens) ?? []
    }

    // MARK: - Boundaries

    static func isSentenceEnd(_ c: Character) -> Bool { c == "." || c == "!" || c == "?" }
    static func isClauseEnd(_ c: Character) -> Bool { c == "," || c == ";" || c == ":" }
    private static func isQuote(_ c: Character) -> Bool { c == "\"" || c == "'" }

    /// Units ending at `isEnd` punctuation that whitespace or the end of the
    /// text follows; newlines always end a unit. A run of enders and closing
    /// quotes ("...", `?!"`) stays with its unit.
    static func splitUnits(_ text: String, isEnd: (Character) -> Bool) -> [String] {
        var units: [String] = []
        var current = ""
        func flush() {
            let unit = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !unit.isEmpty { units.append(unit) }
            current = ""
        }
        let chars = Array(text)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if c.isNewline {
                flush()
                i += 1
                continue
            }
            current.append(c)
            if isEnd(c) {
                while i + 1 < chars.count, isEnd(chars[i + 1]) || isQuote(chars[i + 1]) {
                    i += 1
                    current.append(chars[i])
                }
                if i + 1 == chars.count || chars[i + 1].isWhitespace { flush() }
            }
            i += 1
        }
        flush()
        return units
    }

    static func splitWords(_ text: String) -> [String] {
        text.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
    }

    /// Last resort for one word the budget cannot hold: cut between
    /// characters, never inside one.
    static func splitCharacters(_ word: String, countTokens: (String) -> Int, maxTokens: Int) -> [String] {
        var parts: [String] = []
        var current = ""
        for c in word {
            let candidate = current + String(c)
            if !current.isEmpty && countTokens(candidate) > maxTokens {
                parts.append(current)
                current = String(c)
            } else {
                current = candidate
            }
        }
        if !current.isEmpty { parts.append(current) }
        return parts
    }

    /// Greedily packs `units` into chunks of at most `maxTokens`; a unit too
    /// large on its own is refined at the next level (clauses, then words,
    /// then characters).
    private static func pack(
        _ units: [String],
        countTokens: (String) -> Int,
        maxTokens: Int,
        level: Int,
        into out: inout [String]
    ) {
        var acc = ""
        for unit in units {
            let candidate = acc.isEmpty ? unit : acc + " " + unit
            if countTokens(candidate) <= maxTokens {
                acc = candidate
                continue
            }
            if !acc.isEmpty {
                out.append(acc)
                acc = ""
            }
            if countTokens(unit) <= maxTokens {
                acc = unit
                continue
            }
            var finer: [String] = []
            if level == 0 {
                finer = splitUnits(unit, isEnd: isClauseEnd)
                // A long sentence without clause punctuation still falls back
                // to whole words before any character cut.
                if finer.count <= 1 { finer = splitWords(unit) }
            } else if level == 1 {
                finer = splitWords(unit)
            }
            if level >= 2 || finer.count <= 1 {
                out.append(contentsOf: splitCharacters(unit, countTokens: countTokens, maxTokens: maxTokens))
            } else {
                pack(finer, countTokens: countTokens, maxTokens: maxTokens, level: level + 1, into: &out)
            }
        }
        if !acc.isEmpty { out.append(acc) }
    }

    // MARK: - Retry split

    private static func bestSplit(
        _ chars: [Character],
        cuts: [Int],
        countTokens: (String) -> Int,
        minTokens: Int,
        maxTokens: Int
    ) -> [String]? {
        var best: [String]?
        var bestScore = (Int.max, Int.max, Int.max)
        for cut in cuts {
            guard let rank = boundaryRank(chars, cut: cut) else { continue }
            let left = String(chars[..<cut]).trimmingCharacters(in: .whitespacesAndNewlines)
            let right = String(chars[cut...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if left.isEmpty || right.isEmpty { continue }
            let leftTokens = countTokens(left)
            let rightTokens = countTokens(right)
            if leftTokens < minTokens || rightTokens < minTokens ||
                leftTokens > maxTokens || rightTokens > maxTokens {
                continue
            }
            let score = (rank, max(leftTokens, rightTokens), abs(leftTokens - rightTokens))
            if score < bestScore {
                bestScore = score
                best = [left, right]
            }
        }
        return best
    }

    /// 0 sentence, 1 clause, 2 word, 3 character; `nil` for a cut that would
    /// leave part of a punctuation or closing-quote run on the right.
    private static func boundaryRank(_ chars: [Character], cut: Int) -> Int? {
        let before = chars[cut - 1]
        let after = chars[cut]
        var sawSentenceEnd = false
        var sawClauseEnd = false
        var i = cut
        while i > 0 {
            let c = chars[i - 1]
            if isSentenceEnd(c) {
                sawSentenceEnd = true
            } else if isClauseEnd(c) {
                sawClauseEnd = true
            } else if !isQuote(c) {
                break
            }
            i -= 1
        }
        if sawSentenceEnd || sawClauseEnd {
            if isSentenceEnd(after) || isClauseEnd(after) || isQuote(after) { return nil }
            // "3.|5" or "18:|30" is a cut inside a token, not a boundary.
            if after.isWhitespace { return sawSentenceEnd ? 0 : 1 }
            return 3
        }
        return before.isWhitespace || after.isWhitespace ? 2 : 3
    }
}

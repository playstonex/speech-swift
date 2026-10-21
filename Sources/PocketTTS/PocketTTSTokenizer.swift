import AudioCommon
import Foundation

/// SentencePiece unigram encoding for the pinned English tokenizer.
struct PocketTTSTokenizer: Sendable {
    let pieces: [SentencePieceModel.Piece]
    private let lookup: [String: Int]
    private let maxLength: Int
    init(data: Data) throws {
        let pieces = try SentencePieceModel(data: data).pieces
        guard !pieces.isEmpty, pieces.count <= 10000 else {
            throw PocketTTSError.invalidConfiguration("invalid SentencePiece vocabulary")
        }
        self.pieces = pieces
        var lookup: [String: Int] = [:]
        for (id, piece) in pieces.enumerated() where piece.type == 1 || piece.type == 4 {
            lookup[piece.text] = id
        }
        self.lookup = lookup
        maxLength = lookup.keys.map { $0.unicodeScalars.count }.max() ?? 1
    }
    func encode(_ text: String) throws -> [Int] {
        guard !text.isEmpty else { return [] }
        let scalars = Array(("▁" + text.replacingOccurrences(of: " ", with: "▁")).unicodeScalars)
        var scores = [Float](repeating: -.infinity, count: scalars.count + 1)
        var previous = [(Int, Int)](repeating: (-1, -1), count: scalars.count + 1)
        scores[0] = 0
        let unknownScore = (pieces.map(\.score).min() ?? -100) - 10
        for i in scalars.indices where scores[i].isFinite {
            var coversScalar = false
            for length in 1...min(maxLength, scalars.count-i) {
                let end = i+length
                let part = String(String.UnicodeScalarView(scalars[i..<end]))
                if let id = lookup[part] {
                    if length == 1 { coversScalar = true }
                    if scores[i] + pieces[id].score > scores[end] {
                        scores[end] = scores[i] + pieces[id].score; previous[end] = (i,id)
                    }
                }
            }
            if !coversScalar, scores[i] + unknownScore > scores[i+1] {
                scores[i+1] = scores[i] + unknownScore; previous[i+1] = (i,-2)
            }
        }
        guard scores[scalars.count].isFinite else {
            throw PocketTTSError.invalidInput("text contains characters unsupported by the English tokenizer")
        }
        var ids: [Int] = [], end = scalars.count
        while end > 0 {
            let (start,id) = previous[end]
            guard start >= 0, start < end else { throw PocketTTSError.processingFailed("invalid token path") }
            if id == -2 {
                let fallback = try String(scalars[start]).utf8.map { byte -> Int in
                    let pieceName = String(format: "<0x%02X>", Int(byte))
                    guard let id = pieces.firstIndex(where: { $0.text == pieceName && $0.type == 6 }) else {
                        throw PocketTTSError.invalidConfiguration("missing byte fallback")
                    }
                    return id
                }
                ids.append(contentsOf: fallback.reversed())
            } else { ids.append(id) }
            end = start
        }
        return ids.reversed()
    }
    static func normalized(_ text: String) throws -> String {
        guard text.count <= 4096 else { throw PocketTTSError.invalidInput("text exceeds 4096 characters") }
        var value = text.precomposedStringWithCompatibilityMapping
            .replacingOccurrences(of: "’", with: "'").replacingOccurrences(of: "‘", with: "'")
            .replacingOccurrences(of: "“", with: "\"").replacingOccurrences(of: "”", with: "\"")
        value = value.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        guard !value.isEmpty else { throw PocketTTSError.invalidInput("text is empty") }
        return value
    }
    /// Split at words before prefill/cache limits. Every word is retained.
    func chunks(_ text: String) throws -> [String] {
        var result: [String] = [], current = ""
        for word in try Self.normalized(text).split(separator: " ").map(String.init) {
            let candidate = current.isEmpty ? word : current + " " + word
            if try encode(candidate).count > 50 {
                guard !current.isEmpty, try encode(word).count <= 50 else {
                    throw PocketTTSError.invalidInput("one word exceeds the 50-token chunk limit")
                }
                result.append(current); current = word
            } else { current = candidate }
            if current.hasSuffix(".") || current.hasSuffix("!") || current.hasSuffix("?") {
                result.append(current); current = ""
            }
        }
        if !current.isEmpty { result.append(current) }
        return result
    }
}

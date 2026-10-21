import Foundation

/// Tekken byte-pair encoding and the preset-voice speech prompt. Text is always
/// encoded as ordinary text; control-token spellings in the input are not executed.
public final class VoxtralTTSTokenizer: @unchecked Sendable {
    private struct File: Decodable {
        struct Config: Decodable {
            let pattern: String
            let defaultVocabSize: Int
            let defaultNumSpecialTokens: Int
        }
        struct Entry: Decodable {
            let rank: Int
            let tokenBytes: String
        }
        struct Special: Decodable {
            let rank: Int
            let tokenStr: String
        }
        struct Audio: Decodable { let voiceNumAudioTokens: [String: Int] }
        let config: Config
        let vocab: [Entry]
        let specialTokens: [Special]
        let audio: Audio
    }

    private let ranks: [Data: Int]
    private let pattern: NSRegularExpression
    private let specialOffset: Int
    private let bos: Int
    private let beginAudio: Int
    private let audio: Int
    private let textToAudio: Int
    private let audioToText: Int
    public let voiceTokenCounts: [String: Int]
    public var voices: [String] { voiceTokenCounts.keys.sorted() }

    public init(tekkenURL: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let file = try decoder.decode(File.self, from: Data(contentsOf: tekkenURL))
        specialOffset = file.config.defaultNumSpecialTokens
        let vocabularyCount = file.config.defaultVocabSize - specialOffset
        guard specialOffset > 0, vocabularyCount >= 256 else {
            throw VoxtralTTSError.invalidCheckpoint("invalid Tekken vocabulary size")
        }
        var lookup: [Data: Int] = [:]
        for entry in file.vocab where entry.rank < vocabularyCount {
            guard entry.rank >= 0, let bytes = Data(base64Encoded: entry.tokenBytes),
                  !bytes.isEmpty, lookup.updateValue(entry.rank, forKey: bytes) == nil else {
                throw VoxtralTTSError.invalidCheckpoint("invalid or duplicate Tekken vocabulary entry")
            }
        }
        guard lookup.count == vocabularyCount,
              Set(lookup.values) == Set(0..<vocabularyCount),
              (0...255).allSatisfy({ lookup[Data([UInt8($0)])] != nil }) else {
            throw VoxtralTTSError.invalidCheckpoint("incomplete Tekken byte vocabulary")
        }
        var specials: [String: Int] = [:]
        for token in file.specialTokens {
            guard token.rank >= 0, token.rank < specialOffset,
                  specials.updateValue(token.rank, forKey: token.tokenStr) == nil else {
                throw VoxtralTTSError.invalidCheckpoint("invalid Tekken control token")
            }
        }
        func required(_ name: String) throws -> Int {
            guard let id = specials[name] else {
                throw VoxtralTTSError.invalidCheckpoint("missing Tekken token \(name)")
            }
            return id
        }
        bos = try required("<s>")
        beginAudio = try required("[BEGIN_AUDIO]")
        audio = try required("[AUDIO]")
        textToAudio = try required("[NEXT_AUDIO_TEXT]")
        audioToText = try required("[REPEAT_AUDIO_TEXT]")
        guard !file.audio.voiceNumAudioTokens.isEmpty,
              file.audio.voiceNumAudioTokens.values.allSatisfy({ $0 > 0 && $0 <= 4096 }) else {
            throw VoxtralTTSError.invalidCheckpoint("missing voice prompt lengths")
        }
        ranks = lookup
        pattern = try NSRegularExpression(pattern: file.config.pattern)
        voiceTokenCounts = file.audio.voiceNumAudioTokens
    }

    public func encode(_ text: String) throws -> [Int] {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let source = text as NSString
        let matches = pattern.matches(in: text, range: range)
        var end = 0
        var tokens: [Int] = []
        for match in matches {
            guard match.range.location == end, match.range.length > 0 else {
                throw VoxtralTTSError.invalidInput("text cannot be split by the Tekken pattern")
            }
            let piece = Array(source.substring(with: match.range).utf8)
            tokens.append(contentsOf: try encodeBytes(piece))
            end = NSMaxRange(match.range)
        }
        guard end == range.length else {
            throw VoxtralTTSError.invalidInput("text cannot be split by the Tekken pattern")
        }
        return tokens
    }

    private func encodeBytes(_ bytes: [UInt8]) throws -> [Int] {
        if let rank = ranks[Data(bytes)] { return [rank + specialOffset] }
        var pieces = bytes.map { Data([$0]) }
        while pieces.count > 1 {
            var best: (index: Int, rank: Int)?
            for index in 0..<(pieces.count - 1) {
                if let rank = ranks[pieces[index] + pieces[index + 1]], rank < (best?.rank ?? Int.max) {
                    best = (index, rank)
                }
            }
            guard let best else { break }
            pieces[best.index].append(pieces[best.index + 1])
            pieces.remove(at: best.index + 1)
        }
        return try pieces.map {
            guard let rank = ranks[$0] else {
                throw VoxtralTTSError.invalidCheckpoint("missing Tekken byte token")
            }
            return rank + specialOffset
        }
    }

    public func speechPrompt(text: String, voice: String) throws -> [Int] {
        guard let count = voiceTokenCounts[voice] else { throw VoxtralTTSError.unknownVoice(voice) }
        return [bos, beginAudio] + Array(repeating: audio, count: count)
            + [textToAudio] + (try encode(text)) + [audioToText, beginAudio]
    }

    public static func normalizedText(_ text: String) throws -> String {
        let normalized = text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        guard !normalized.isEmpty, normalized.utf8.count <= 16_000,
              !normalized.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) else {
            throw VoxtralTTSError.invalidInput("text must contain 1...16000 UTF-8 bytes without control characters")
        }
        return normalized
    }
}

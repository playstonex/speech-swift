import Foundation

public enum FishAudioDrama3Prompt {
    public static let maximumRows = 8_192
    static let messageEnd = "<|im_end|>\n"
    static let generationStart = "<|im_start|>assistant\n<|voice|>"
    private static let structural = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"]

    public static func validateText(_ text: String) throws {
        guard text.unicodeScalars.contains(where: { !$0.properties.isWhitespace }) else {
            throw FishAudioError.invalidInput("Drama 3 text must not be empty")
        }
        guard text.utf8.count <= 131_072, !structural.contains(where: text.contains) else {
            throw FishAudioError.invalidInput("Drama 3 text is too long or contains a structural token")
        }
        let markers = try NSRegularExpression(pattern: "<\\|speaker")
        let tag = try NSRegularExpression(pattern: "<\\|speaker:([0-9]+)\\|>")
        let full = NSRange(text.startIndex..., in: text)
        for marker in markers.matches(in: text, range: full) {
            let range = NSRange(location: marker.range.location, length: full.length - marker.range.location)
            guard let match = tag.firstMatch(in: text, options: .anchored, range: range),
                  let digitsRange = Range(match.range(at: 1), in: text),
                  let number = Int(text[digitsRange]), (0...15).contains(number),
                  String(number) == String(text[digitsRange]) else {
                throw FishAudioError.invalidInput("Drama 3 speaker tags must be <|speaker:0|> through <|speaker:15|>")
            }
        }
    }

    static func withSpeaker(_ text: String) -> String {
        let leading = text.unicodeScalars.prefix(while: { $0.properties.isWhitespace })
        let stripped = String(text.unicodeScalars.dropFirst(leading.count))
        return stripped.hasPrefix("<|speaker:") ? text : "<|speaker:0|> \(text)"
    }

    static func build(text: String, references: [FishAudioReferencePrompt] = [], history: [FishAudioReferencePrompt] = [],
                      config: FishAudioDrama3Config, encode: (String) -> [Int]) throws -> FishAudioModelInput {
        try validateText(text)
        guard references.count <= 16, history.count <= 64 else { throw FishAudioError.invalidInput("Too many Drama 3 references or history turns") }
        var rows = Array(repeating: [Int32](), count: config.tower.codebookCount + 1)
        func appendText(_ string: String) throws {
            let ids = encode(string)
            guard ids.allSatisfy({ $0 >= 0 && $0 < config.text.vocabularySize }),
                  rows[0].count + ids.count <= maximumRows else { throw FishAudioError.invalidInput("Drama 3 prompt exceeds token limits") }
            rows[0].append(contentsOf: ids.map(Int32.init))
            for i in 1..<rows.count { rows[i].append(contentsOf: repeatElement(0, count: ids.count)) }
        }
        func appendCodes(_ reference: FishAudioReferencePrompt) throws {
            try validateText(reference.text)
            let codes = reference.codes
            guard codes.count == config.tower.codebookCount, let count = codes.first?.count, count > 0,
                  rows[0].count + count <= maximumRows,
                  codes.allSatisfy({ $0.count == count && $0.allSatisfy({ (0..<config.tower.vocabularySize).contains($0) }) }) else {
                throw FishAudioError.invalidCodebookShape("Drama 3 references require matching, nonempty codebooks with in-range ids")
            }
            rows[0].append(contentsOf: codes[0].map { Int32(config.semanticStartTokenId + $0) })
            for i in codes.indices { rows[i + 1].append(contentsOf: codes[i].map(Int32.init)) }
        }
        // Each literal is encoded separately: BPE merges across these seams change the prompt.
        try appendText("<|im_start|>system\n")
        try appendText(references.isEmpty ? "convert the provided text to speech" : "convert the provided text to speech reference to the following:")
        for (index, reference) in references.enumerated() {
            try validateText(reference.text)
            let tagged = withSpeaker(reference.text)
            let leading = tagged.unicodeScalars.prefix(while: { $0.properties.isWhitespace })
            var transcript = String(tagged.unicodeScalars.dropFirst(leading.count))
            if let end = transcript.range(of: "|>")?.upperBound {
                transcript.insert(contentsOf: String(leading), at: end)
            }
            try appendText("\n\n[Reference \(index + 1)]\nText: \(transcript)\nSpeech: ")
            try appendCodes(reference)
        }
        try appendText(messageEnd)
        for turn in history {
            try validateText(turn.text)
            try appendText("<|im_start|>user\n")
            try appendText(withSpeaker(turn.text))
            try appendText(messageEnd)
            try appendText(generationStart)
            try appendCodes(turn)
            try appendText(messageEnd)
        }
        try appendText("<|im_start|>user\n")
        try appendText(withSpeaker(text))
        try appendText(messageEnd)
        try appendText(generationStart)
        return try FishAudioModelInput(rows: rows)
    }
}

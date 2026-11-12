import Foundation
import Hub
import Tokenizers

/// Avoids scanning thousands of unused Gemma placeholders on ordinary text.
/// Calls are serialized by the model's generation guard.
final class BreezeTextTokenizer {
    private let compact: Tokenizer
    private let loadFull: () throws -> Tokenizer
    private let hasUnusedTokens: Bool
    private var full: Tokenizer?

    init(config: Config, data: Config, loadFull: @escaping () throws -> Tokenizer) throws {
        let added = data["added_tokens"].array(or: [])
        let used = added.filter { !$0.content.string(or: "").hasPrefix("<unused") }
        hasUnusedTokens = used.count != added.count
        var compactData = data.dictionary(or: [:])
        compactData["added_tokens"] = Config(used)
        compact = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: Config(compactData))
        self.loadFull = loadFull
    }

    static func fromBundle(_ directory: URL) async throws -> BreezeTextTokenizer {
        let configuration = LanguageModelConfigurationFromHub(modelFolder: directory)
        guard let config = try await configuration.tokenizerConfig else { throw TokenizerError.missingConfig }
        let data = try await configuration.tokenizerData
        return try BreezeTextTokenizer(config: config, data: data) {
            let api = HubApi.shared
            return try AutoTokenizer.from(
                tokenizerConfig: config,
                tokenizerData: api.configuration(fileURL: directory.appendingPathComponent("tokenizer.json")))
        }
    }

    func encode(_ text: String) throws -> [Int] {
        // Keep every original added-token rule when a placeholder could occur.
        // This also handles overlapping placeholders and their whitespace flags.
        if hasUnusedTokens && text.contains("<unused") {
            if full == nil { full = try loadFull() }
            return full!.encode(text: text, addSpecialTokens: true)
        }
        return compact.encode(text: text, addSpecialTokens: true)
    }
}

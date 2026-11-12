import Foundation

extension FishAudioDrama3Prompt {
    public static func build(text: String, references: [FishAudioReferencePrompt] = [],
                             history: [FishAudioReferencePrompt] = [], tokenizer: FishAudioTokenizer,
                             config: FishAudioDrama3Config) throws -> FishAudioModelInput {
        for code in 0..<config.tower.vocabularySize {
            guard tokenizer.tokenId(FishAudioToken.semantic(code)) == config.semanticStartTokenId + code else {
                throw FishAudioError.invalidConfig("Drama 3 tokenizer semantic ids do not match config")
            }
        }
        guard tokenizer.tokenId(FishAudioToken.imEnd) == config.eosTokenId else {
            throw FishAudioError.invalidConfig("Drama 3 tokenizer EOS does not match config")
        }
        return try build(text: text, references: references, history: history, config: config, encode: tokenizer.encode)
    }

}

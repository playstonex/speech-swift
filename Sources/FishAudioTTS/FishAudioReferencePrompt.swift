import Foundation

public struct FishAudioReferencePrompt: Sendable, Equatable {
    public let text: String
    public let codes: [[Int]]

    public init(text: String, codes: [[Int]]) throws {
        guard !codes.isEmpty else {
            throw FishAudioError.invalidCodebookShape("reference codes must not be empty")
        }
        let width = codes[0].count
        guard width > 0 else {
            throw FishAudioError.invalidCodebookShape("reference codebook frames must not be empty")
        }
        guard codes.allSatisfy({ $0.count == width }) else {
            throw FishAudioError.invalidCodebookShape("all codebooks must have the same frame count")
        }
        self.text = text
        self.codes = codes
    }
}

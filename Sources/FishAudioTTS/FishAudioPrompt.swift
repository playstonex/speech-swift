import Foundation

public enum FishAudioEmotionMarker {
    public static let supported: [String] = [
        "[pause]",
        "[emphasis]",
        "[laughing]",
        "[excited]",
        "[angry]",
        "[whisper]",
        "[screaming]",
        "[shouting]",
        "[surprised]",
        "[sad]",
    ]
}

public enum FishAudioPrompt {
    public static func systemPrompt(referenceTexts: [String]) -> String {
        guard !referenceTexts.isEmpty else {
            return "convert the provided text to speech"
        }
        let referenceText = referenceTexts.enumerated().map { index, text in
            text.contains("<|speaker:") ? text : "<|speaker:\(index)|>\(text)"
        }.joined(separator: "\n")
        return """
        convert the provided text to speech reference to the following:

        Text:
        \(referenceText)

        Speech:
        """
    }

    public static func userPrompt(_ text: String) -> String {
        text
    }

    public static func chatTemplate(system: String, user: String) -> String {
        """
        \(FishAudioToken.imStart)system
        \(system)\(FishAudioToken.imEnd)
        \(FishAudioToken.imStart)user
        \(user)\(FishAudioToken.imEnd)
        \(FishAudioToken.imStart)assistant
        \(FishAudioToken.voiceModality)
        """
    }
}

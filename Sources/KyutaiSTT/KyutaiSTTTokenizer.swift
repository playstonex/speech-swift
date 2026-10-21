import AudioCommon
import Foundation

struct KyutaiSTTTokenizer {
    let vocabulary: SentencePieceModel
    private var bytes: [UInt8] = []

    init(vocabulary: SentencePieceModel) { self.vocabulary = vocabulary }

    mutating func append(_ id: Int) -> String {
        guard id > 3, let piece = vocabulary[id] else { return "" }
        if piece.pieceType == .byte {
            guard piece.text.hasPrefix("<0x"), piece.text.hasSuffix(">"),
                  let byte = UInt8(piece.text.dropFirst(3).dropLast(), radix: 16) else { return "" }
            bytes.append(byte)
            guard let text = String(bytes: bytes, encoding: .utf8) else { return "" }
            bytes.removeAll(keepingCapacity: true)
            return text
        }
        guard piece.pieceType == .normal || piece.pieceType == .userDefined else { return "" }
        let prefix = String(decoding: bytes, as: UTF8.self)
        bytes.removeAll(keepingCapacity: true)
        return prefix + piece.text.replacingOccurrences(of: "▁", with: " ")
    }
}

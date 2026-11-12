import Foundation

/// Storage precision for unquantized network parameters. AudioVAE stays FP32.
public enum VoxCPM2WeightPrecision: String, Sendable, Codable {
    case automatic
    case float16
    case float32

    func validate(quantized: Bool) throws {
        if self == .float16 && quantized {
            throw NSError(domain: "VoxCPM2TTSModel", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "FP16 requires an unquantized checkpoint; quantized weights are not an FP16 model"
            ])
        }
    }
}

import Foundation
import MLX
import XCTest
@testable import MossTranscribe

/// Exercises the real loader with a tiny, deliberately incompatible bundle.
/// No downloaded weights or large model allocations are needed.
final class E2EMossMLXCacheLimitTests: XCTestCase {
    func testWeightLoadFailureAppliesCeilingAndRestoresCallerBudget() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("moss-cache-limit-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try makeIncompatibleBundle(in: directory)

        let previousLimit = Memory.cacheLimit
        defer {
            Memory.clearCache()
            Memory.cacheLimit = previousLimit
        }
        let gibibyte = 1024 * 1024 * 1024
        let modelLimit = min(2 * gibibyte, Int(ProcessInfo.processInfo.physicalMemory) / 4)
        for callerLimit in [8 * gibibyte, gibibyte / 2, 0] {
            Memory.cacheLimit = callerLimit
            var limitBeforeWeights: Int?
            do {
                _ = try await MossMLXModel.fromDirectory(directory) { _, message in
                    if message == "Loading MOSS MLX audio encoder..." {
                        limitBeforeWeights = Memory.cacheLimit
                    }
                }
                XCTFail("The incomplete weight file must be rejected")
            } catch MossTranscribeError.invalidConfiguration(let reason) {
                XCTAssertTrue(reason.contains("incompatible MLX audio weights"), reason)
            }
            XCTAssertEqual(limitBeforeWeights, min(callerLimit, modelLimit))
            XCTAssertEqual(Memory.cacheLimit, callerLimit)
        }
    }

    private func makeIncompatibleBundle(in directory: URL) throws {
        func write(_ name: String, _ object: [String: Any]) throws {
            try JSONSerialization.data(withJSONObject: object)
                .write(to: directory.appendingPathComponent(name))
        }

        try write("config.json", [
            "audio_config": [
                "hidden_size": 16, "intermediate_size": 32,
                "layer_norm_eps": 0.00001, "max_source_positions": 1500,
                "merge_size": 4, "num_heads": 2, "num_layers": 1, "num_mel_bins": 80,
            ],
            "audio_token_id": 1, "backend": "mlx",
            "decoder_config": [
                "head_dim": 8, "hidden_size": 16, "intermediate_size": 32,
                "num_heads": 2, "num_kv_heads": 1, "num_layers": 1,
                "rms_norm_eps": 0.000001, "rope_theta": 1000000, "vocab_size": 32,
            ],
            "files": ["audio_encoder": "audio_encoder.safetensors", "decoder": "decoder.safetensors"],
            "model_type": "moss-transcribe-diarize-mlx",
            "precision": "fp16-audio-int8-decoder",
            "quantization_config": ["bits": 8, "group_size": 8, "mode": "affine"],
        ])
        try write("processor_config.json", [
            "audio_tokens_per_second": 12.5, "audio_merge_size": 4,
            "time_marker_every_seconds": 1, "enable_time_marker": false,
        ])
        try write("preprocessor_config.json", [
            "feature_size": 80, "hop_length": 160, "n_fft": 400,
            "n_samples": 480000, "nb_max_frames": 3000, "sampling_rate": 16000,
        ])
        try write("generation_config.json", [:])
        try write("tokenizer_config.json", [
            "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>",
        ])
        let tokens = ["<unk>", "<|audio_pad|>", "<|im_end|>"] + "0123456789".map(String.init)
        let vocabulary = Dictionary(uniqueKeysWithValues: tokens.enumerated().map { ($0.element, $0.offset) })
        let specialTokens: [[String: Any]] = tokens.prefix(3).enumerated().map {
            ["id": $0.offset, "content": $0.element, "special": true,
             "single_word": false, "lstrip": false, "rstrip": false, "normalized": false]
        }
        try write("tokenizer.json", [
            "version": "1.0", "added_tokens": specialTokens,
            "model": ["type": "BPE", "vocab": vocabulary, "merges": [], "unk_token": "<unk>"],
        ])
        for name in ["audio_encoder.safetensors", "decoder.safetensors"] {
            try MLX.save(
                arrays: ["unexpected_weight": MLXArray([Float.zero])],
                url: directory.appendingPathComponent(name)
            )
        }
    }
}

import ArgumentParser
import XCTest
@testable import AudioCLILib

final class KyutaiSTTCommandTests: XCTestCase {
    func testDefaultAndExplicitMemoryCacheLimit() throws {
        let defaults = try TranscribeCommand.parse(["speech.wav", "--engine", "kyutai"])
        XCTAssertEqual(defaults.effectiveKyutaiCacheLimitMB, 256)
        for limit in [0, 256, 1024] {
            let command = try TranscribeCommand.parse([
                "speech.wav", "--engine", "kyutai", "--mlx-cache-limit-mb", "\(limit)"])
            XCTAssertEqual(command.mlxCacheLimitMB, limit)
            XCTAssertEqual(command.effectiveKyutaiCacheLimitMB, limit)
        }
        for limit in ["-1", String(Int.max)] {
            XCTAssertThrowsError(try TranscribeCommand.parse([
                "speech.wav", "--engine", "kyutai", "--mlx-cache-limit-mb", limit]))
        }
        XCTAssertThrowsError(try TranscribeCommand.parse(["speech.wav", "--mlx-cache-limit-mb", "256"]))
    }

    func testNativeStreamParses() throws {
        let command = try TranscribeCommand.parse(["speech.wav", "--engine", "kyutai", "--stream", "--partial"])
        XCTAssertTrue(command.stream)
        XCTAssertNoThrow(try command.validate())
    }

    func testAliasesAndCustomModel() throws {
        XCTAssertEqual(try resolveKyutaiModelId("0.6B"), "aufklarer/Kyutai-STT-1B-MLX-8bit")
        XCTAssertEqual(try resolveKyutaiModelId("1B"), "aufklarer/Kyutai-STT-1B-MLX-8bit")
        XCTAssertEqual(try resolveKyutaiModelId("INT8"), "aufklarer/Kyutai-STT-1B-MLX-8bit")
        XCTAssertEqual(try resolveKyutaiModelId("4bit"), "aufklarer/Kyutai-STT-1B-MLX-4bit")
        XCTAssertEqual(try resolveKyutaiModelId("fp16"), "aufklarer/Kyutai-STT-1B-MLX-FP16")
        XCTAssertEqual(try resolveKyutaiModelId("upstream"), "kyutai/stt-1b-en_fr")
        XCTAssertEqual(try resolveKyutaiModelId("2.6b"), "kyutai/stt-2.6b-en")
        XCTAssertEqual(try resolveKyutaiModelId("org/kyutai-export"), "org/kyutai-export")
        XCTAssertEqual(try resolveKyutaiModelId("."), ".")
        XCTAssertThrowsError(try resolveKyutaiModelId("int7"))
        XCTAssertThrowsError(try resolveKyutaiModelId("/missing/export"))
    }

    func testRejectsUnsupportedPromptOptions() {
        for args in [["--language", "fr"], ["--context", "names"], ["--partial"]] {
            XCTAssertThrowsError(try TranscribeCommand.parse(["speech.wav", "--engine", "kyutai"] + args))
        }
    }
}

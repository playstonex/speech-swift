import Foundation
import XCTest
@testable import AsrBenchmark

final class AsrBenchmarkTests: XCTestCase {
    func testEmptyTranscriptCountsEveryReferenceWordAsDeletion() {
        let result = WER.compute(reference: "one two three", hypothesis: "")
        XCTAssertEqual(result.deletions, 3)
        XCTAssertEqual(result.insertions, 0)
        XCTAssertEqual(result.substitutions, 0)
        XCTAssertEqual(result.wer, 1)
        let characters = CER.compute(reference: "one two", hypothesis: "")
        XCTAssertEqual(characters.editDistance, 6)
        XCTAssertEqual(characters.cer, 1)
    }

    func testEmptyReferenceAndNonemptyAlignment() {
        XCTAssertEqual(WER.compute(reference: "", hypothesis: "one two").insertions, 2)
        XCTAssertEqual(WER.compute(reference: "", hypothesis: "").totalErrors, 0)
        let result = WER.compute(reference: "one two three", hypothesis: "one five three four")
        XCTAssertEqual(result.substitutions, 1)
        XCTAssertEqual(result.insertions, 1)
        XCTAssertEqual(result.deletions, 0)
        XCTAssertEqual(result.totalErrors, 2)
        XCTAssertEqual(result.referenceWords, 3)
    }

    func testCacheLimitArgumentBounds() throws {
        for limit in [0, 256] {
            let command = try AsrBench.parse(["--dataset", "fixture.tsv", "--mlx-cache-limit-mb", "\(limit)"])
            XCTAssertEqual(command.mlxCacheLimitMB, limit)
        }
        for value in ["-1", String(Int.max)] {
            XCTAssertThrowsError(try AsrBench.parse(["--dataset", "fixture.tsv", "--mlx-cache-limit-mb", value]))
        }
    }

    func testKyutaiVariantsHaveDistinctEngineIDs() {
        for suffix in ["fp16", "int4", "int8"] {
            let id = EngineID(rawValue: "kyutai-stt-mlx-\(suffix)")
            XCTAssertEqual(id?.make().name, "kyutai-stt-mlx-\(suffix)")
        }
    }

    func testMemoryHighWaterMarksIncludeCurrentUsage() {
        let usage = currentProcessMemoryUsage()
        XCTAssertGreaterThan(usage.residentBytes, 0)
        XCTAssertGreaterThanOrEqual(usage.peakResidentBytes, usage.residentBytes)
        XCTAssertGreaterThanOrEqual(usage.peakPhysicalFootprintBytes, usage.physicalFootprintBytes)
    }

    func testUtteranceTimingsDecodeWithoutStreamingFields() throws {
        let data = Data("""
        {"id":"fixture","reference":"Hello","hypothesis":"Hello", "audioSeconds":1,
         "elapsedSeconds":0.2,"werPercent":0}
        """.utf8)
        let result = try JSONDecoder().decode(UtteranceResult.self, from: data)
        XCTAssertNil(result.firstTokenSeconds)
        XCTAssertNil(result.frameLatencyP95Seconds)
        XCTAssertEqual(result.elapsedSeconds, 0.2)
    }
}

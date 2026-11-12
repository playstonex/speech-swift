#if canImport(CoreML)
import CoreML
import Foundation
import XCTest
@testable import SpeechVAD

final class E2ETurnModelsTests: XCTestCase {
    func testPublishedBundlesDownloadAndWorkOffline() async throws {
        guard ProcessInfo.processInfo.environment["RUN_PUBLISHED_TURN_MODELS"] == "1" else {
            throw XCTSkip("Set RUN_PUBLISHED_TURN_MODELS=1 to check fresh public downloads")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let p99Directory = root.appendingPathComponent("p99")
        let vapDirectory = root.appendingPathComponent("vap")
        let p99 = try await P99TurnModel.fromPretrained(
            cacheDir: p99Directory, downloadOptions: .init(retryDelaysSeconds: []))
        let zero = [Float](repeating: 0, count: 2_560)
        let online = try p99.createSession().pushAudio(zero)
        XCTAssertEqual(online.count, 8)
        let cached = try await P99TurnModel.fromPretrained(cacheDir: p99Directory, offlineMode: true)
        let offline = try cached.createSession().pushAudio(zero)
        XCTAssertEqual(offline.count, online.count)
        for (left, right) in zip(online, offline) {
            XCTAssertEqual(left.probability, right.probability, accuracy: 0.0002)
        }
        let vap = try await VAPModel.fromPretrained(
            cacheDir: vapDirectory, downloadOptions: .init(retryDelaysSeconds: []))
        let silence = [Float](repeating: 0, count: 320_000)
        let forecast = try vap.forecast(userAudio: silence, agentAudio: silence)
        XCTAssertEqual(forecast.pNow[0], 0.44378302, accuracy: 0.0005)
        let offlineVap = try await VAPModel.fromPretrained(cacheDir: vapDirectory, offlineMode: true)
        let cachedForecast = try offlineVap.forecast(userAudio: silence, agentAudio: silence)
        XCTAssertEqual(cachedForecast.pNow[0], forecast.pNow[0], accuracy: 0.0005)
        for directory in [p99Directory, vapDirectory] {
            for file in ["config.json", "LICENSE", "NOTICE"] {
                XCTAssertFalse(try Data(contentsOf: directory.appendingPathComponent(file)).isEmpty)
            }
        }
    }

    private func directory(_ key: String) throws -> URL {
        guard let path = ProcessInfo.processInfo.environment[key] else {
            throw XCTSkip("Set \(key) to a downloaded compiled model bundle")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    func testP99MatchesUpstreamReferenceAcrossChunkSizesAndReset() throws {
        let model = try P99TurnModel.load(from: directory("P99_TURN_MODEL_DIR"), computeUnits: .cpuOnly)
        let audio = (0..<5_120).map { Float(0.05 * sin(2 * Double.pi * 173 * Double($0) / 16_000)) }
        let expected: [Float] = [
            0.42350024, 0.18286484, 0.11129376, 0.10120583,
            0.08812279, 0.07812408, 0.07061130, 0.06588340,
            0.06441975, 0.06236923, 0.06329080, 0.06167912,
            0.05885968, 0.05599582, 0.05373377, 0.05230293,
        ]
        let session = try model.createSession()
        let whole = try session.pushAudio(audio)
        XCTAssertEqual(whole.count, expected.count)
        for (actual, target) in zip(whole, expected) {
            XCTAssertEqual(actual.probability, target, accuracy: 0.0002)
        }
        try session.reset()
        var chunked = [P99TurnFrame]()
        for start in stride(from: 0, to: audio.count, by: 317) {
            chunked += try session.pushAudio(Array(audio[start..<min(start + 317, audio.count)]))
        }
        XCTAssertEqual(chunked.count, whole.count)
        for (left, right) in zip(whole, chunked) {
            XCTAssertEqual(left.probability, right.probability, accuracy: 1e-6)
            XCTAssertEqual(left.availableAfterSeconds, right.availableAfterSeconds)
        }
    }

    func testVAPSilenceMatchesOriginalCheckpoint() throws {
        let model = try VAPModel.load(from: directory("VAP_MODEL_DIR"), computeUnits: .cpuOnly)
        let zero = [Float](repeating: 0, count: 320_000)
        let result = try model.forecast(userAudio: zero, agentAudio: zero)
        XCTAssertEqual(result.pNow[0], 0.44378302, accuracy: 0.0005)
        XCTAssertEqual(result.pNow[1], 0.55563056, accuracy: 0.0005)
        XCTAssertEqual(result.pFuture[0], 0.34941691, accuracy: 0.0005)
        XCTAssertEqual(result.pFuture[1], 0.65039903, accuracy: 0.0005)
        XCTAssertEqual(result.voiceActivity[0], 0.00018012, accuracy: 0.00001)
        XCTAssertEqual(result.voiceActivity[1], 0.00018012, accuracy: 0.00001)
    }

    func testVAPChannelsMatchOriginalCheckpoint() throws {
        let model = try VAPModel.load(from: directory("VAP_MODEL_DIR"), computeUnits: .cpuOnly)
        let zero = [Float](repeating: 0, count: 320_000)
        let signal = (0..<320_000).map { index -> Float in
            if (index / 2_400) % 3 == 0 { return 0 }
            let time = Double(index) / 16_000
            return Float(0.05 * sin(2 * Double.pi * 173 * time) + 0.02 * sin(2 * Double.pi * 587 * time))
        }
        let result = try model.forecast(userAudio: signal, agentAudio: zero)
        XCTAssertEqual(result.pNow[0], 0.98009735, accuracy: 0.0005)
        XCTAssertEqual(result.pNow[1], 0.01986181, accuracy: 0.0005)
        let swapped = try model.forecast(userAudio: zero, agentAudio: signal)
        XCTAssertEqual(swapped.pNow[0], 0.01861823, accuracy: 0.0005)
        XCTAssertEqual(swapped.pNow[1], 0.98133320, accuracy: 0.0005)
    }

    func testDefaultComputeUnitsMatchTheReference() throws {
        let p99 = try P99TurnModel.load(from: directory("P99_TURN_MODEL_DIR"))
        let audio = (0..<2_560).map { Float(0.05 * sin(2 * Double.pi * 173 * Double($0) / 16_000)) }
        let frames = try p99.createSession().pushAudio(audio)
        XCTAssertEqual(frames[0].probability, 0.42350024, accuracy: 0.0002)
        let vap = try VAPModel.load(from: directory("VAP_MODEL_DIR"))
        let zero = [Float](repeating: 0, count: 320_000)
        let result = try vap.forecast(userAudio: zero, agentAudio: zero)
        XCTAssertEqual(result.pNow[0], 0.44378302, accuracy: 0.0005)
    }
}
#endif

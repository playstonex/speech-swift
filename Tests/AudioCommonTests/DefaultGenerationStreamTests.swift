import XCTest
@testable import AudioCommon

final class DefaultGenerationStreamTests: XCTestCase {
    private final class Generator: SpeechGenerationModel {
        let started: XCTestExpectation
        let cancelled: XCTestExpectation
        var sampleRate: Int { 24000 }
        init(started: XCTestExpectation, cancelled: XCTestExpectation) {
            self.started = started
            self.cancelled = cancelled
        }
        func generate(text: String, language: String?) async throws -> [Float] {
            started.fulfill()
            do { try await Task.sleep(nanoseconds: 30_000_000_000) }
            catch { cancelled.fulfill(); throw error }
            return [1]
        }
    }

    func testCancellingConsumerCancelsDefaultProducer() async {
        let started = expectation(description: "producer starts")
        let cancelled = expectation(description: "producer cancels")
        let generator = Generator(started: started, cancelled: cancelled)
        let task = Task {
            do {
                for try await _ in generator.generateStream(text: "hello", language: nil) {
                    XCTFail("Cancelled producer must not emit audio")
                }
            } catch is CancellationError { }
            catch { XCTFail("Unexpected error: \(error)") }
        }
        await fulfillment(of: [started], timeout: 2)
        task.cancel()
        await fulfillment(of: [cancelled], timeout: 2)
        await task.value
    }
}

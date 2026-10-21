import XCTest
import AudioCommon
@testable import Qwen3Chat

final class Qwen35PipelineLLMTests: XCTestCase {
    private enum StubError: Error {
        case generationFailed
    }

    private final class StubBackend: Qwen35ChatBackend {
        enum Behavior {
            case chunks([String])
            case failure(Error)
        }

        let tokenizer = ChatTokenizer()
        let config = Qwen3ChatConfig.qwen35_08B
        private let behavior: Behavior

        init(_ behavior: Behavior) {
            self.behavior = behavior
        }

        func generateStream(
            messages: [ChatMessage],
            sampling: ChatSamplingConfig
        ) -> AsyncThrowingStream<String, Error> {
            _ = messages
            _ = sampling
            return AsyncThrowingStream { continuation in
                switch behavior {
                case .chunks(let chunks):
                    for chunk in chunks {
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                case .failure(let error):
                    continuation.finish(throwing: error)
                }
            }
        }

        func resetState() {}
    }

    func testEmptyGenerationStillFinalizesExactlyOnce() {
        let events = run(behavior: .chunks([]))

        XCTAssertEqual(events, [.init(text: "", isFinal: true)])
    }

    func testFailureBeforeFirstTokenStillFinalizesExactlyOnce() {
        let events = run(behavior: .failure(StubError.generationFailed))

        XCTAssertEqual(events, [.init(text: "", isFinal: true)])
    }

    func testTokensAreFollowedByExactlyOneFinalMarker() {
        let events = run(behavior: .chunks(["Hello", " world"]))

        XCTAssertEqual(events, [
            .init(text: "Hello", isFinal: false),
            .init(text: " world", isFinal: false),
            .init(text: "", isFinal: true),
        ])
    }

    func testCancellationStillFinalizesExactlyOnce() {
        let llm = Qwen35PipelineLLM(model: StubBackend(.chunks(["Hello", " world"])))
        var events: [Event] = []
        llm.onToken = { _ in llm.cancel() }

        llm.chat(messages: [(role: .user, content: "Hello")]) { text, isFinal in
            events.append(.init(text: text, isFinal: isFinal))
        }

        XCTAssertEqual(events, [
            .init(text: "Hello", isFinal: false),
            .init(text: "", isFinal: true),
        ])
    }

    private struct Event: Equatable {
        let text: String
        let isFinal: Bool
    }

    private func run(behavior: StubBackend.Behavior) -> [Event] {
        let llm = Qwen35PipelineLLM(model: StubBackend(behavior))
        var events: [Event] = []

        llm.chat(messages: [(role: .user, content: "Hello")]) { text, isFinal in
            events.append(.init(text: text, isFinal: isFinal))
        }

        return events
    }
}

import BenchmarkSupport
import XCTest

final class StreamingLatencyTests: XCTestCase {
    func testPrebufferStartsWhenEnoughPCMArrives() throws {
        var timing = BenchmarkStreamingLatency(sampleRate: 100, prebufferSeconds: 0.2)
        timing.append(samples: 10, arrivalSeconds: 0.1)
        XCTAssertNil(timing.prebufferReadySeconds)
        timing.append(samples: 20, arrivalSeconds: 0.4)
        XCTAssertEqual(try XCTUnwrap(timing.prebufferReadySeconds), 0.4, accuracy: 1e-9)
        XCTAssertEqual(timing.minimumContinuousStartSeconds, 0.3, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(timing.additionalBufferSeconds), 0, accuracy: 1e-9)
        XCTAssertEqual(timing.scheduledSamples, 30)
    }

    func testEarlyPacketCanStillRequireMoreBufferToAvoidAGap() throws {
        var timing = BenchmarkStreamingLatency(sampleRate: 100, prebufferSeconds: 0.1)
        timing.append(samples: 10, arrivalSeconds: 0.1)
        timing.append(samples: 10, arrivalSeconds: 0.45)
        timing.append(samples: 20, arrivalSeconds: 0.5)
        timing.append(samples: 0, arrivalSeconds: 0.9)
        timing.finish(at: 1)
        XCTAssertEqual(try XCTUnwrap(timing.prebufferReadySeconds), 0.1, accuracy: 1e-9)
        XCTAssertEqual(timing.minimumContinuousStartSeconds, 0.35, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(timing.additionalBufferSeconds), 0.25, accuracy: 1e-9)
        XCTAssertEqual(timing.scheduledSamples, 40)
    }

    func testShortClipStartsOnCompletionAndEmptyStreamStaysUnscheduled() throws {
        var timing = BenchmarkStreamingLatency(sampleRate: 100, prebufferSeconds: 0.5)
        timing.append(samples: 10, arrivalSeconds: 0.1)
        timing.finish(at: 0.3)
        XCTAssertEqual(try XCTUnwrap(timing.prebufferReadySeconds), 0.3, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(timing.additionalBufferSeconds), 0, accuracy: 1e-9)
        var empty = BenchmarkStreamingLatency(sampleRate: 100, prebufferSeconds: 0)
        empty.append(samples: 0, arrivalSeconds: 0.1)
        empty.finish(at: 0.2)
        XCTAssertNil(empty.prebufferReadySeconds)
        XCTAssertNil(empty.additionalBufferSeconds)
    }
}

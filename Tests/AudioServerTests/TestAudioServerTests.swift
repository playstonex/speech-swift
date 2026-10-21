import XCTest
@testable import AudioServer

final class TestAudioServerTests: XCTestCase {
    func testWaitsForListenerBeyondPreviousStartupSleep() async throws {
        let port = 19_392
        let fixture = TestAudioServer(port: port) {
            // The old 1.5-second setup sleep would connect before this listener exists.
            try await Task.sleep(for: .seconds(2))
            let server = AudioServer(
                host: "127.0.0.1", port: port,
                realtimeState: FailingRealtimeModelLoading())
            try await server.run()
        }
        addTeardownBlock { await fixture.stop() }

        try await fixture.start()
        let socket = URLSession.shared.webSocketTask(
            with: URL(string: "ws://127.0.0.1:\(port)/v1/realtime")!)
        socket.resume()
        defer { socket.cancel(with: .normalClosure, reason: nil) }

        let message = try await socket.receive()
        guard case .string(let text) = message else {
            return XCTFail("Expected a session.created message")
        }
        let json = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
        XCTAssertEqual(json?["type"] as? String, "session.created")
    }

    func testPropagatesServerStartupFailure() async throws {
        enum ForcedError: Error { case startup }
        let fixture = TestAudioServer(port: 19_393) { throw ForcedError.startup }
        addTeardownBlock { await fixture.stop() }

        do {
            try await fixture.start(timeout: .seconds(1))
            XCTFail("Expected the server's startup error")
        } catch ForcedError.startup {
            // Startup errors must reach setup instead of becoming connection failures.
        }
    }

    func testReadinessTimeoutStopsTheServerTask() async throws {
        let lifecycle = ServerLifecycle()
        let port = 19_394
        let fixture = TestAudioServer(port: port) {
            do {
                try await Task.sleep(for: .seconds(60))
            } catch {
                await lifecycle.markStopped()
                throw error
            }
        }
        addTeardownBlock { await fixture.stop() }

        do {
            try await fixture.start(timeout: .milliseconds(200))
            XCTFail("Expected a bounded readiness timeout")
        } catch TestAudioServer.StartupError.timedOut(let failedPort, let lastFailure) {
            XCTAssertEqual(failedPort, port)
            XCTAssertFalse(lastFailure.isEmpty)
        }
        let stopped = await lifecycle.stopped
        XCTAssertTrue(stopped, "Failed setup must await the cancelled server task")
    }

    func testCancelledSetupStopsTheServerTask() async throws {
        let started = expectation(description: "Server task started")
        let lifecycle = ServerLifecycle()
        let fixture = TestAudioServer(port: 19_396) {
            started.fulfill()
            do {
                try await Task.sleep(for: .seconds(60))
            } catch {
                await lifecycle.markStopped()
                throw error
            }
        }
        let startup = Task { try await fixture.start(timeout: .seconds(5)) }
        addTeardownBlock {
            startup.cancel()
            await fixture.stop()
        }

        await fulfillment(of: [started], timeout: 5)
        startup.cancel()
        do {
            try await startup.value
            XCTFail("Expected cancelled setup to throw")
        } catch is CancellationError {
            // Cancellation must remain distinguishable from a readiness timeout.
        }
        let stopped = await lifecycle.stopped
        XCTAssertTrue(stopped, "Cancelled setup must await the server task")
    }

    func testServerExitingBeforeReadinessFailsSetup() async throws {
        let port = 19_397
        let fixture = TestAudioServer(port: port) {}
        addTeardownBlock { await fixture.stop() }

        do {
            try await fixture.start(timeout: .seconds(1))
            XCTFail("Expected setup to fail when the server exits")
        } catch TestAudioServer.StartupError.stoppedBeforeReady(let failedPort) {
            XCTAssertEqual(failedPort, port)
        }
    }

    func testShutdownCompletesBeforePortIsReused() async throws {
        let port = 19_395
        let run: @Sendable () async throws -> Void = {
            let server = AudioServer(
                host: "127.0.0.1", port: port,
                realtimeState: FailingRealtimeModelLoading())
            try await server.run()
        }
        let first = TestAudioServer(port: port, run: run)
        let second = TestAudioServer(port: port, run: run)
        addTeardownBlock {
            await second.stop()
            await first.stop()
        }

        try await first.start()
        await first.stop()
        try await second.start()
    }
}

private actor ServerLifecycle {
    private(set) var stopped = false

    func markStopped() { stopped = true }
}

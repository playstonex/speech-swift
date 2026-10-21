import Foundation

/// Owns a test server until its listener is ready and its shutdown completes.
actor TestAudioServer {
    enum StartupError: Error, LocalizedError {
        case stoppedBeforeReady(port: Int)
        case timedOut(port: Int, lastFailure: String)

        var errorDescription: String? {
            switch self {
            case .stoppedBeforeReady(let port):
                return "Test server on port \(port) stopped before /health was ready"
            case .timedOut(let port, let lastFailure):
                return "Timed out waiting for test server on port \(port): \(lastFailure)"
            }
        }
    }

    private let port: Int
    private let run: @Sendable () async throws -> Void
    private var serverTask: Task<Void, Never>?
    private var completion: Result<Void, Error>?

    init(port: Int, run: @escaping @Sendable () async throws -> Void) {
        self.port = port
        self.run = run
    }

    func start(timeout: Duration = .seconds(10)) async throws {
        precondition(serverTask == nil, "Test server is already running")
        completion = nil
        serverTask = Task {
            do {
                try await run()
                completion = .success(())
            } catch {
                completion = .failure(error)
            }
        }

        do {
            try await waitUntilReady(timeout: timeout)
        } catch {
            await stop()
            throw error
        }
    }

    func stop() async {
        guard let serverTask else { return }
        serverTask.cancel()
        await serverTask.value
        self.serverTask = nil
    }

    private func checkServerTask() throws {
        if let completion {
            try completion.get()
            throw StartupError.stoppedBeforeReady(port: port)
        }
    }

    private func waitUntilReady(timeout: Duration) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 1
        config.timeoutIntervalForResource = 1
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }

        let url = URL(string: "http://127.0.0.1:\(port)/health")!
        var lastFailure = "No successful health response"
        while clock.now < deadline {
            try Task.checkCancellation()
            try checkServerTask()

            let remaining = clock.now.duration(to: deadline).components
            var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData)
            request.timeoutInterval = min(
                1, max(0.01, Double(remaining.seconds) + Double(remaining.attoseconds) / 1e18))
            var ready = false
            do {
                let (data, response) = try await session.data(for: request)
                let status = (response as? HTTPURLResponse)?.statusCode
                let body = try JSONSerialization.jsonObject(with: data) as? [String: String]
                ready = status == 200 && body?["status"] == "ok"
                if !ready { lastFailure = "Unexpected /health response (HTTP \(status ?? 0))" }
            } catch {
                lastFailure = String(describing: error)
            }

            try Task.checkCancellation()
            try checkServerTask()
            if ready { return }
            if clock.now < deadline {
                try await clock.sleep(until: min(deadline, clock.now.advanced(by: .milliseconds(50))))
            }
        }
        try Task.checkCancellation()
        try checkServerTask()
        throw StartupError.timedOut(port: port, lastFailure: lastFailure)
    }
}

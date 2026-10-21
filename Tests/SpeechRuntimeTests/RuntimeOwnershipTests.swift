import XCTest
@testable import SpeechRuntime

private actor Counts {
    var loads = 0
    var active = 0
    var maximum = 0
    func loaded() -> Int { loads += 1; return loads }
    func begin() { active += 1; maximum = max(maximum, active) }
    func end() { active -= 1 }
    func snapshot() -> (Int, Int) { (loads, maximum) }
}

final class RuntimeOwnershipTests: XCTestCase {
    func testConcurrentRequestsLoadOnceAndNeverOverlap() async throws {
        let counts = Counts()
        let runtime = ModelRuntime<Int>(maximumPending: 64) { await counts.loaded() }
        let results = try await withThrowingTaskGroup(of: Int.self) { group in
            for _ in 0..<20 {
                group.addTask {
                    try await runtime.withModel { model in
                        await counts.begin()
                        try await Task.sleep(nanoseconds: 1_000_000)
                        await counts.end()
                        return model
                    }
                }
            }
            var results: [Int] = []
            for try await result in group { results.append(result) }
            return results
        }
        XCTAssertEqual(results, Array(repeating: 1, count: 20))
        let snapshot = await counts.snapshot()
        XCTAssertEqual(snapshot.0, 1)
        XCTAssertEqual(snapshot.1, 1)
        try await runtime.unload()
        let reloaded = try await runtime.withModel { $0 }
        XCTAssertEqual(reloaded, 2)
    }

    func testFailureDoesNotPoisonModelLoadOrExecutionLane() async throws {
        enum Failure: Error { case expected }
        let counts = Counts()
        let runtime = ModelRuntime<Int> {
            let count = await counts.loaded()
            if count == 1 { throw Failure.expected }
            return count
        }
        do { _ = try await runtime.withModel { $0 }; XCTFail("Load must fail") }
        catch Failure.expected { }
        let loaded = try await runtime.withModel { $0 }
        XCTAssertEqual(loaded, 2)
        do {
            _ = try await runtime.withModel { _ -> Int in throw Failure.expected }
            XCTFail("Inference must fail")
        } catch Failure.expected { }
        let reused = try await runtime.withModel { $0 }
        XCTAssertEqual(reused, 2)
    }

    func testQueueIsBoundedAndCancellationRemovesWaiter() async throws {
        let gate = InferenceGate(maximumPending: 1)
        let first = try await gate.acquire()
        let waiting = Task { try await gate.acquire() }
        while gate.pendingCount == 0 { await Task.yield() }
        do { _ = try await gate.acquire(); XCTFail("Queue must reject excess admission") }
        catch InferenceGate.AdmissionError.queueFull { }
        waiting.cancel()
        do { _ = try await waiting.value; XCTFail("Waiter must cancel without acquiring") }
        catch is CancellationError { }
        XCTAssertEqual(gate.pendingCount, 0)
        first.release()
        first.release()
        let next = try await gate.acquire()
        next.release()
    }

    func testCancelledRequestNeverLoadsModel() async throws {
        let counts = Counts()
        let runtime = ModelRuntime<Int> { await counts.loaded() }
        let started = expectation(description: "request created")
        let task = Task {
            started.fulfill()
            while !Task.isCancelled { await Task.yield() }
            return try await runtime.withModel { $0 }
        }
        await fulfillment(of: [started], timeout: 2)
        task.cancel()
        do { _ = try await task.value; XCTFail("Must cancel") }
        catch is CancellationError { }
        let snapshot = await counts.snapshot()
        XCTAssertEqual(snapshot.0, 0)
    }

    func testCacheCoalescesSameIdentityAndRetriesFailure() async throws {
        final class Token { }
        let counts = Counts()
        let cache = ModelCache()
        let tokens = try await withThrowingTaskGroup(of: ObjectIdentifier.self) { group in
            for _ in 0..<20 {
                group.addTask {
                    let token = try await cache.load(key: "same") {
                        _ = await counts.loaded()
                        try await Task.sleep(nanoseconds: 20_000_000)
                        return Token()
                    }
                    return ObjectIdentifier(token)
                }
            }
            var ids = Set<ObjectIdentifier>()
            for try await id in group { ids.insert(id) }
            return ids
        }
        XCTAssertEqual(tokens.count, 1)
        let snapshot = await counts.snapshot()
        XCTAssertEqual(snapshot.0, 1)
        enum Failure: Error { case expected }
        do {
            let _: Int = try await cache.load(key: "retry") { throw Failure.expected }
            XCTFail("Must fail")
        } catch Failure.expected { }
        let recovered: Int = try await cache.load(key: "retry") { 42 }
        XCTAssertEqual(recovered, 42)
        // Equal string identities with different concrete types cannot collide.
        let other: String = try await cache.load(key: "retry") { "other" }
        XCTAssertEqual(other, "other")
    }

    func testUnloadDrainsAnActiveRequestBeforeReleasingTheModel() async throws {
        let entered = expectation(description: "inference entered")
        let counts = Counts()
        let runtime = ModelRuntime<Int> { await counts.loaded() }
        let request = Task {
            try await runtime.withModel { model in
                entered.fulfill()
                try await Task.sleep(nanoseconds: 50_000_000)
                return model
            }
        }
        await fulfillment(of: [entered], timeout: 2)
        try await runtime.unload()
        let result = try await request.value
        XCTAssertEqual(result, 1)
        let reloaded = try await runtime.withModel { $0 }
        XCTAssertEqual(reloaded, 2)
    }
}

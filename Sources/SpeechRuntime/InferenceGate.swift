import Foundation

/// A bounded FIFO execution lane for stateful inference. Waiting requests are
/// cancellable; a permit remains held until its owner explicitly releases it.
public final class InferenceGate: @unchecked Sendable {
    public enum AdmissionError: Error, LocalizedError {
        case queueFull
        public var errorDescription: String? { "Inference queue is full; retry the request later" }
    }

    // Every field is accessed under the enclosing gate's lock.
    private final class Waiter: @unchecked Sendable {
        enum State { case new, waiting, granted, cancelled }
        var state: State = .new
        var continuation: CheckedContinuation<InferencePermit, Error>?
    }

    private let lock = NSLock()
    private var occupied = false
    private var waiters: [Waiter] = []
    private let maximumPending: Int

    public init(maximumPending: Int = 32) {
        precondition(maximumPending >= 0)
        self.maximumPending = maximumPending
    }

    public var pendingCount: Int {
        lock.lock(); defer { lock.unlock() }
        return waiters.count
    }

    public func acquire() async throws -> InferencePermit {
        let waiter = Waiter()
        let permit = try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                enqueue(waiter, continuation: continuation)
            }
        } onCancel: {
            self.cancel(waiter)
        }
        do {
            try Task.checkCancellation()
            return permit
        } catch {
            permit.release()
            throw error
        }
    }

    private func enqueue(_ waiter: Waiter, continuation: CheckedContinuation<InferencePermit, Error>) {
        lock.lock()
        if waiter.state == .cancelled {
            lock.unlock()
            continuation.resume(throwing: CancellationError())
        } else if !occupied {
            occupied = true
            waiter.state = .granted
            lock.unlock()
            continuation.resume(returning: InferencePermit(gate: self))
        } else if waiters.count >= maximumPending {
            waiter.state = .cancelled
            lock.unlock()
            continuation.resume(throwing: AdmissionError.queueFull)
        } else {
            waiter.state = .waiting
            waiter.continuation = continuation
            waiters.append(waiter)
            lock.unlock()
        }
    }

    private func cancel(_ waiter: Waiter) {
        lock.lock()
        guard waiter.state != .granted else { lock.unlock(); return }
        waiter.state = .cancelled
        waiters.removeAll { $0 === waiter }
        let continuation = waiter.continuation
        waiter.continuation = nil
        lock.unlock()
        continuation?.resume(throwing: CancellationError())
    }

    fileprivate func release() {
        lock.lock()
        if waiters.isEmpty {
            occupied = false
            lock.unlock()
        } else {
            let waiter = waiters.removeFirst()
            waiter.state = .granted
            let continuation = waiter.continuation
            waiter.continuation = nil
            lock.unlock()
            continuation?.resume(returning: InferencePermit(gate: self))
        }
    }
}

/// Release at the end of the entire inference operation, including streaming.
public final class InferencePermit: @unchecked Sendable {
    private let lock = NSLock()
    private var gate: InferenceGate?

    fileprivate init(gate: InferenceGate) { self.gate = gate }

    public func release() {
        lock.lock()
        let gate = self.gate
        self.gate = nil
        lock.unlock()
        gate?.release()
    }

    deinit { release() }
}

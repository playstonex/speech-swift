import Foundation

private final class MutableModel: @unchecked Sendable {
    var active = 0
    var operations = 0
}

private final class Token: Sendable { }
private actor LoadCount {
    var value = 0
    func increment() { value += 1 }
}

@main
struct RuntimeOwnershipStress {
    static func main() async throws {
        let runtime = ModelRuntime<MutableModel>(maximumPending: 64) { MutableModel() }
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<16 {
                group.addTask {
                    for _ in 0..<1000 {
                        try await runtime.withModel { model in
                            model.active += 1
                            precondition(model.active == 1)
                            await Task.yield()
                            precondition(model.active == 1)
                            model.operations += 1
                            model.active -= 1
                        }
                    }
                }
            }
            try await group.waitForAll()
        }
        try await runtime.withModel { precondition($0.operations == 16000 && $0.active == 0) }
        let cache = ModelCache()
        let loads = LoadCount()
        let identities = try await withThrowingTaskGroup(of: ObjectIdentifier.self) { group in
            for _ in 0..<64 {
                group.addTask {
                    let token = try await cache.load(key: "shared") {
                        await loads.increment()
                        try await Task.sleep(for: .milliseconds(2))
                        return Token()
                    }
                    return ObjectIdentifier(token)
                }
            }
            var identities = Set<ObjectIdentifier>()
            for try await identity in group { identities.insert(identity) }
            return identities
        }
        let count = await loads.value
        precondition(identities.count == 1 && count == 1)

        for _ in 0..<100 {
            let gate = InferenceGate(maximumPending: 64)
            let held = try await gate.acquire()
            let waiting = (0..<32).map { _ in
                Task {
                    let permit = try await gate.acquire()
                    permit.release()
                }
            }
            let release = Task.detached { held.release() }
            for index in stride(from: 0, to: waiting.count, by: 2) { waiting[index].cancel() }
            for task in waiting {
                do { try await task.value } catch is CancellationError { }
            }
            await release.value
            precondition(gate.pendingCount == 0)
            let final = try await gate.acquire()
            final.release()
        }

        let playback = PlaybackSampleBuffer(capacity: 4096)
        let total = 960 * 2048
        let producer = Task.detached {
            for offset in stride(from: 0, to: total, by: 960) {
                playback.write((offset..<(offset + 960)).map(Float.init))
                if offset.isMultiple(of: 15360) { await Task.yield() }
            }
        }
        let consumer = Task.detached {
            let destination = UnsafeMutablePointer<Float>.allocate(capacity: 512)
            defer { destination.deallocate() }
            let deadline = ContinuousClock.now + .seconds(30)
            var consumed = 0
            while consumed < total {
                precondition(ContinuousClock.now < deadline, "Playback stalled or lost samples")
                let count = playback.read(into: destination, count: 512)
                for index in 0..<count {
                    precondition(destination[index] == Float(consumed + index), "Playback reordered or corrupted samples")
                }
                consumed += count
                if count == 0 { await Task.yield() }
            }
            return consumed
        }
        await producer.value
        let consumed = await consumer.value
        precondition(consumed == total)
        playback.reclaimConsumed()
        print("PASS: 16000 exclusive operations, 64 coalesced loads, 3200 cancellation races, \(total) ordered playback samples")
    }
}

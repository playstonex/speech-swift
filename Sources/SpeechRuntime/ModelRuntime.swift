import Foundation

/// Optional ownership for a single model. All access, including load and
/// unload, passes through the same execution lane. Model-specific APIs remain
/// available independently of this wrapper.
public final class ModelRuntime<Model>: @unchecked Sendable {
    private let gate: InferenceGate
    private let loader: @Sendable () async throws -> Model
    private let unloader: @Sendable (Model) -> Void
    // Accessed exclusively while holding gate's permit.
    private var model: Model?

    public init(
        maximumPending: Int = 32,
        unloader: @escaping @Sendable (Model) -> Void = { _ in },
        loader: @escaping @Sendable () async throws -> Model
    ) {
        gate = InferenceGate(maximumPending: maximumPending)
        self.loader = loader
        self.unloader = unloader
    }

    public func withModel<Result: Sendable>(
        _ operation: @Sendable (Model) async throws -> Result
    ) async throws -> Result {
        let permit = try await gate.acquire()
        defer { permit.release() }
        if model == nil { model = try await loader() }
        try Task.checkCancellation()
        let result = try await operation(model!)
        try Task.checkCancellation()
        return result
    }

    public func unload() async throws {
        let permit = try await gate.acquire()
        defer { permit.release() }
        if let model { unloader(model) }
        model = nil
    }

    deinit { if let model { unloader(model) } }
}

import Foundation

/// Coalesces loads by concrete model type and identity. It protects cache
/// metadata, not inference: callers must hold an execution permit when using
/// a cached model that does not support concurrent inference.
public actor ModelCache {
    private struct Key: Hashable {
        let type: ObjectIdentifier
        let identity: String
    }
    private struct Box: @unchecked Sendable { let value: Any }
    private var loads: [Key: Task<Box, Error>] = [:]

    public init() {}

    public func load<Model>(
        key: String,
        loader: @escaping @Sendable () async throws -> Model
    ) async throws -> Model {
        let key = Key(type: ObjectIdentifier(Model.self), identity: key)
        if let existing = loads[key] {
            return try await existing.value.value as! Model
        }
        let task = Task { Box(value: try await loader()) }
        loads[key] = task
        do {
            return try await task.value.value as! Model
        } catch {
            loads[key] = nil
            throw error
        }
    }
}

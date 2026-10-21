/// Runtime capabilities describe execution rather than inferring a backend
/// from a repository name or treating a buffered result as incremental output.
public struct ModelDescriptor: Sendable, Equatable, Codable {
    public enum Backend: String, Sendable, Codable {
        case mlx
        case coreML = "coreml"
        case hybrid
    }
    public enum Streaming: String, Sendable, Codable { case buffered, incremental }

    public let id: String
    public let modelID: String
    public let backend: Backend
    public let inputSampleRate: Int
    public let streaming: Streaming

    public init(id: String, modelID: String, backend: Backend, inputSampleRate: Int, streaming: Streaming) {
        self.id = id
        self.modelID = modelID
        self.backend = backend
        self.inputSampleRate = inputSampleRate
        self.streaming = streaming
    }
}

#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

struct P99TurnConfiguration: Decodable {
    let modelType: String
    let sampleRate: Int
    let stepSamples: Int
    let frameSamples: Int
    let framesPerStep: Int
    let compiledModel: String
    let stateShapes: [String: [Int]]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", sampleRate = "sample_rate"
        case stepSamples = "step_samples", frameSamples = "frame_samples"
        case framesPerStep = "frames_per_step", compiledModel = "compiled_model"
        case stateShapes = "state_shapes"
    }
}

/// Streaming audio-only end-of-turn probabilities from p99lab turn-1-mini.
/// The compiled graph includes the required 4 kHz filtering and log-mel frontend.
public final class P99TurnModel {
    public static let defaultModelId = "aufklarer/P99-Turn-1-Mini-CoreML"
    public static let sampleRate = 16_000
    public static let stepSamples = 2_560
    static let compiledModel = "p99_turn_step.mlmodelc"
    static let stateShapes = [
        "audio": [1, 304], "conv1": [1, 80, 2], "conv2": [1, 256, 2],
        "k": [6, 4, 60, 64], "v": [6, 4, 60, 64],
        "k_valid": [60], "h6": [60, 256],
    ]
    private let model: MLModel

    private init(model: MLModel) { self.model = model }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        downloadOptions: DownloadOptions = .init(),
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> P99TurnModel {
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId, to: directory,
            additionalFiles: ["\(compiledModel)/**", "config.json", "LICENSE", "NOTICE"],
            offlineMode: offlineMode, downloadOptions: downloadOptions,
            progressHandler: { progressHandler?($0, "Downloading p99 turn model...") })
        return try load(from: directory)
    }

    /// Load a downloaded compiled bundle without network access.
    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) throws -> P99TurnModel {
        _ = try decodeConfiguration(Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = try CoreMLLoader.load(
            url: directory.appendingPathComponent(compiledModel),
            computeUnits: computeUnits, name: "p99-turn-1-mini")
        let expectedInputs = ["audio": [1, stepSamples]].merging(
            stateShapes.reduce(into: [String: [Int]]()) { $0["state_" + $1.key] = $1.value },
            uniquingKeysWith: { first, _ in first })
        let expectedOutputs = ["probs": [8]].merging(
            stateShapes.reduce(into: [String: [Int]]()) { $0["new_state_" + $1.key] = $1.value },
            uniquingKeysWith: { first, _ in first })
        for (descriptions, expected) in [
            (model.modelDescription.inputDescriptionsByName, expectedInputs),
            (model.modelDescription.outputDescriptionsByName, expectedOutputs),
        ] {
            guard Set(descriptions.keys) == Set(expected.keys) else {
                throw invalid("compiled graph names do not match config.json")
            }
            for (name, shape) in expected {
                guard let constraint = descriptions[name]?.multiArrayConstraint,
                    constraint.dataType == .float32,
                    constraint.shape.map(\.intValue) == shape
                else { throw invalid("compiled graph tensor does not match: \(name)") }
            }
        }
        return P99TurnModel(model: model)
    }

    static func decodeConfiguration(_ data: Data) throws -> P99TurnConfiguration {
        let config = try JSONDecoder().decode(P99TurnConfiguration.self, from: data)
        guard config.modelType == "p99-turn-1-mini-coreml",
            config.sampleRate == sampleRate, config.stepSamples == stepSamples,
            config.frameSamples == 320, config.framesPerStep == 8,
            config.compiledModel == compiledModel, config.stateShapes == stateShapes
        else { throw invalid("incompatible p99 turn model configuration") }
        return config
    }

    static func invalid(_ reason: String) -> AudioModelError {
        .invalidConfiguration(model: "p99 turn-1-mini", reason: reason)
    }

    /// Each conversation needs its own session. Serialize calls within a session.
    public func createSession() throws -> P99TurnSession {
        let predictor = try P99TurnPredictor(model: model)
        return P99TurnSession(reset: { try predictor.reset() }, step: { try predictor.step($0) })
    }
}

/// A model frame and the amount of audio required before it became available.
public struct P99TurnFrame: Sendable {
    public let probability: Float
    public let frameEndSeconds: Double
    public let availableAfterSeconds: Double
}

/// Buffers arbitrary 16 kHz mono chunks into the model's 160 ms steps.
/// Partial steps remain buffered; reset discards them for the next conversation.
public final class P99TurnSession {
    private let resetPredictor: () throws -> Void
    private let predictStep: ([Float]) throws -> [Float]
    private var pending = [Float]()
    public private(set) var processedSamples: Int64 = 0

    init(reset: @escaping () throws -> Void, step: @escaping ([Float]) throws -> [Float]) {
        resetPredictor = reset
        predictStep = step
    }

    public func reset() throws {
        try resetPredictor()
        pending.removeAll(keepingCapacity: true)
        processedSamples = 0
    }

    public func pushAudio(_ samples: [Float]) throws -> [P99TurnFrame] {
        guard samples.allSatisfy(\.isFinite) else {
            throw P99TurnModel.invalid("audio contains a non-finite sample")
        }
        pending.append(contentsOf: samples)
        var consumed = 0
        defer { if consumed > 0 { pending.removeFirst(consumed) } }
        var frames = [P99TurnFrame]()
        while pending.count - consumed >= P99TurnModel.stepSamples {
            let end = consumed + P99TurnModel.stepSamples
            let probabilities = try autoreleasepool {
                try predictStep(Array(pending[consumed..<end]))
            }
            guard probabilities.count == 8,
                probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) })
            else { throw P99TurnModel.invalid("model returned invalid probabilities") }
            for (index, probability) in probabilities.enumerated() {
                frames.append(P99TurnFrame(
                    probability: probability,
                    frameEndSeconds: Double(processedSamples + Int64((index + 1) * 320)) / 16_000,
                    availableAfterSeconds: Double(processedSamples + 2_560) / 16_000))
            }
            processedSamples += 2_560
            consumed = end
        }
        return frames
    }
}

private final class P99TurnPredictor {
    private let model: MLModel
    private var state = [String: MLMultiArray]()

    init(model: MLModel) throws {
        self.model = model
        try reset()
    }

    func reset() throws {
        var next = [String: MLMultiArray]()
        for (name, shape) in P99TurnModel.stateShapes {
            let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
            array.dataPointer.assumingMemoryBound(to: Float.self).initialize(repeating: 0, count: array.count)
            next["state_" + name] = array
        }
        state = next
    }

    func step(_ samples: [Float]) throws -> [Float] {
        let audio = try MLMultiArray(shape: [1, 2_560], dataType: .float32)
        samples.withUnsafeBufferPointer {
            audio.dataPointer.assumingMemoryBound(to: Float.self).update(from: $0.baseAddress!, count: $0.count)
        }
        var inputs = state.mapValues { MLFeatureValue(multiArray: $0) }
        inputs["audio"] = MLFeatureValue(multiArray: audio)
        let result = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
        guard let probabilities = result.featureValue(for: "probs")?.multiArrayValue,
            probabilities.count == 8
        else { throw P99TurnModel.invalid("model omitted end-of-turn probabilities") }
        let values = (0..<8).map { probabilities[$0].floatValue }
        guard values.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
            throw P99TurnModel.invalid("model returned invalid probabilities")
        }
        var next = [String: MLMultiArray]()
        for (name, shape) in P99TurnModel.stateShapes {
            guard let array = result.featureValue(for: "new_state_" + name)?.multiArrayValue,
                array.shape.map(\.intValue) == shape, array.dataType == .float32
            else { throw P99TurnModel.invalid("model omitted streaming state: \(name)") }
            next["state_" + name] = array
        }
        state = next
        return values
    }
}
#endif

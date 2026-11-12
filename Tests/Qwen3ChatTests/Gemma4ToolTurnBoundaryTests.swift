import XCTest
@testable import Qwen3Chat

final class Gemma4ToolTurnBoundaryTests: XCTestCase {
    private var lookup: Gemma4ToolCall {
        .init(id: "lookup-1", name: "lookup_account", arguments: .object(["id": .string("account_7")]))
    }

    func testUserTurnAfterToolResultClosesThePreviousModelTurn() throws {
        for preserveText in [false, true] {
            let prompt = try Gemma4ToolCodec.prompt(messages: [
                .init(role: .user, content: "Look up my account."),
                .init(role: .assistant, toolCalls: [lookup]),
                .init(role: .tool, content: "{\"found\":true}", toolCallID: lookup.id),
                .init(role: .user, content: "Please continue.")
            ], tools: [], preserveToolResponseText: preserveText)
            XCTAssertTrue(prompt.contains("<tool_response|><turn|>\n<|turn>user\nPlease continue.<turn|>\n<|turn>model\n"), prompt)
            XCTAssertFalse(prompt.contains("<tool_response|><|turn>user"), prompt)
        }
    }

    func testUserTurnAfterChainedToolResultsClosesOnlyTheActiveModelTurn() throws {
        let details = Gemma4ToolCall(id: "details-2", name: "read_details", arguments: .object([:]))
        let prompt = try Gemma4ToolCodec.prompt(messages: [
            .init(role: .user, content: "Check my account and details."),
            .init(role: .assistant, toolCalls: [lookup]),
            .init(role: .tool, content: "{\"found\":true}", toolCallID: lookup.id),
            .init(role: .assistant, toolCalls: [details]),
            .init(role: .tool, content: "{\"status\":\"active\"}", toolCallID: details.id),
            .init(role: .user, content: "Thanks, continue.")
        ])
        XCTAssertTrue(prompt.contains("<tool_response|><|tool_call>call:read_details"), prompt)
        XCTAssertTrue(prompt.contains("<tool_response|><turn|>\n<|turn>user\nThanks, continue."), prompt)
        XCTAssertEqual(prompt.components(separatedBy: "<|turn>model\n").count - 1, 2)
    }

    func testAssistantAnswerAfterToolResultStillContinuesTheSameModelTurn() throws {
        let prompt = try Gemma4ToolCodec.prompt(messages: [
            .init(role: .user, content: "Look up my account."),
            .init(role: .assistant, toolCalls: [lookup]),
            .init(role: .tool, content: "{\"found\":true}", toolCallID: lookup.id),
            .init(role: .assistant, content: "I found your account.")
        ])
        XCTAssertTrue(prompt.contains("<tool_response|>I found your account.<turn|>\n"), prompt)
        XCTAssertEqual(prompt.components(separatedBy: "<|turn>model\n").count - 1, 2)
    }

    func testFinalToolResultStillLeavesModelTurnOpenForGeneration() throws {
        let prompt = try Gemma4ToolCodec.prompt(messages: [
            .init(role: .user, content: "Look up my account."),
            .init(role: .assistant, toolCalls: [lookup]),
            .init(role: .tool, content: "{\"found\":true}", toolCallID: lookup.id)
        ])
        XCTAssertTrue(prompt.hasSuffix("<tool_response|>"), prompt)
        XCTAssertEqual(prompt.components(separatedBy: "<|turn>model\n").count - 1, 1)
    }
}

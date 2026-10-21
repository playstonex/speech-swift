import Foundation
import XCTest

final class ShardIsolationScriptTests: XCTestCase {
    func testRunsExactSelectedMethodsInSeparateClassProcesses() throws {
        let fileManager = FileManager.default
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let temporaryRoot = fileManager.temporaryDirectory
            .appendingPathComponent("shard-isolation-\(UUID().uuidString)", isDirectory: true)
        let binDirectory = temporaryRoot.appendingPathComponent("bin", isDirectory: true)
        let fakeSwift = binDirectory.appendingPathComponent("swift")
        let invocationsFile = temporaryRoot.appendingPathComponent("invocations.txt")
        let logDirectory = temporaryRoot.appendingPathComponent("logs", isDirectory: true)

        try fileManager.createDirectory(at: binDirectory, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: temporaryRoot) }

        let fakeSwiftScript = #"""
        #!/bin/bash
        set -eu
        if [[ "${1:-}" == "test" && "${2:-}" == "list" ]]; then
          printf '%s\n' \
            'DemoTests.ConfigTests/testDefault' \
            'DemoTests.ConfigTests/testSkipped' \
            'DemoTests.E2EModelTests/testOnlyHeavy' \
            'OtherTests.E2EModelTests/testOnlyHeavy'
          exit 0
        fi
        printf '%s\n' "$*" >> "$FAKE_SWIFT_INVOCATIONS"
        """#
        try fakeSwiftScript.write(to: fakeSwift, atomically: true, encoding: .utf8)
        try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeSwift.path)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            packageRoot.appendingPathComponent("scripts/test_shard_isolated.sh").path,
            "DemoTests|testOnlyHeavy",
            "testSkipped|OtherTests",
        ]
        process.currentDirectoryURL = packageRoot
        var environment = ProcessInfo.processInfo.environment
        environment["PATH"] = "\(binDirectory.path):\(environment["PATH"] ?? "/usr/bin:/bin")"
        environment["FAKE_SWIFT_INVOCATIONS"] = invocationsFile.path
        environment["SHARD_TEST_LOG_DIR"] = logDirectory.path
        process.environment = environment

        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe
        try process.run()
        process.waitUntilExit()
        let output = String(
            data: outputPipe.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8) ?? ""

        XCTAssertEqual(process.terminationStatus, 0, output)
        XCTAssertTrue(output.contains("Selected 2 tests across 2 classes"), output)
        XCTAssertTrue(output.contains("Overall: GREEN (2 tests across 2 isolated classes)"), output)

        let invocations = try String(contentsOf: invocationsFile, encoding: .utf8)
            .split(separator: "\n")
            .map(String.init)
        XCTAssertEqual(
            invocations,
            [
                "test --skip-build --disable-sandbox --no-parallel --filter ^(DemoTests\\.ConfigTests/testDefault)$",
                "test --skip-build --disable-sandbox --no-parallel --filter ^(DemoTests\\.E2EModelTests/testOnlyHeavy)$",
            ],
            output)
    }
}

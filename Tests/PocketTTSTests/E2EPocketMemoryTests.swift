import Darwin
import Foundation
import XCTest
@testable import PocketTTS

final class E2EPocketMemoryTests: XCTestCase {
    func testRepeatedRequestsReleaseConditioningTemporaries() async throws {
        guard let directory = ProcessInfo.processInfo.environment["POCKET_TTS_BUNDLE"] else {
            throw XCTSkip("Set POCKET_TTS_BUNDLE to run the repeated-request memory regression")
        }
        let model = try PocketTTSModel.fromBundle(URL(fileURLWithPath: directory))
        for _ in 0..<3 { _ = try await model.generate(text: "Hello world.", options: .init()) }
        let baseline = try footprint()
        for _ in 0..<8 { _ = try await model.generate(text: "Hello world.", options: .init()) }
        let final = try footprint()
        // Missing the conditioning autorelease pool retained about 30 MiB/request.
        // Allow allocator/cache variation, while detecting that linear growth.
        XCTAssertLessThanOrEqual(final,baseline+64*1024*1024)
    }
    private func footprint() throws -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info)/4)
        let status = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
            }
        }
        guard status == KERN_SUCCESS else { throw PocketTTSError.processingFailed("task_info failed") }
        return info.phys_footprint
    }
}

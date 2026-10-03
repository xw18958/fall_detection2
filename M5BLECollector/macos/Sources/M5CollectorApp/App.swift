import SwiftUI
import AppKit

@main
struct M5CollectorApp: App {
    @StateObject private var collector = CollectorViewModel()

    var body: some Scene {
        WindowGroup("M5 Data Collector") {
            MainView(viewModel: collector)
                .onAppear { collector.start() }
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
                    collector.shutdown()
                }
        }
        .windowResizability(.contentSize)
    }
}

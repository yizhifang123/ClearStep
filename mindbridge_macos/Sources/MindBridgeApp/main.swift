import AppKit
import Foundation
import SwiftUI
import WebKit

private let defaultPort = "8504"
private let appBackground = Color(red: 0.96, green: 0.98, blue: 1.0)
private let ink = Color(red: 0.08, green: 0.12, blue: 0.18)
private let muted = Color(red: 0.38, green: 0.45, blue: 0.54)
private let accent = Color(red: 0.15, green: 0.39, blue: 0.92)

@main
struct MindBridgeDesktopApp: App {
    @StateObject private var server = StreamlitServer()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(server)
                .frame(minWidth: 1120, minHeight: 760)
                .task {
                    await server.start()
                }
                .onDisappear {
                    server.stop()
                }
        }
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}

@MainActor
final class StreamlitServer: ObservableObject {
    @Published var status = "Starting MindBridge..."
    @Published var appURL: URL?
    @Published var errorMessage: String?

    private var process: Process?

    var repoRoot: URL {
        if let resource = Bundle.main.url(forResource: "RepoRoot", withExtension: "txt"),
           let path = try? String(contentsOf: resource).trimmingCharacters(in: .whitespacesAndNewlines),
           !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true)
        }

        return Bundle.main.bundleURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    var port: String {
        ProcessInfo.processInfo.environment["MINDBRIDGE_DESKTOP_PORT"] ?? defaultPort
    }

    var url: URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }

    func start() async {
        guard process == nil else { return }
        appURL = nil
        errorMessage = nil
        status = "Checking local server..."

        if await isReachable(url) {
            status = "Connected to MindBridge."
            appURL = url
            return
        }

        let python = repoRoot.appendingPathComponent(".venv/bin/python")
        let app = repoRoot.appendingPathComponent("app.py")

        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            fail("Could not find executable Python at \(python.path). Create the project .venv first.")
            return
        }
        guard FileManager.default.fileExists(atPath: app.path) else {
            fail("Could not find MindBridge app.py at \(app.path).")
            return
        }

        status = "Starting Streamlit..."
        let process = Process()
        process.executableURL = python
        process.currentDirectoryURL = repoRoot
        process.arguments = [
            "-m", "streamlit", "run", "app.py",
            "--server.address", "127.0.0.1",
            "--server.port", port,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        process.environment = serverEnvironment()

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
            self.process = process
        } catch {
            fail("Could not start Streamlit: \(error.localizedDescription)")
            return
        }

        let ready = await waitUntilReady(url)
        if ready {
            status = "Connected to MindBridge."
            appURL = url
        } else {
            let log = String(data: pipe.fileHandleForReading.availableData, encoding: .utf8) ?? ""
            fail("Streamlit did not become ready on port \(port). \(log)")
        }
    }

    func stop() {
        process?.terminate()
        process = nil
    }

    private func serverEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = repoRoot.path
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        env["MINDBRIDGE_DESKTOP"] = "1"

        // Preserve live-provider keys from shell launches.
        _ = env["GEMINI_API_KEY"]
        _ = env["ANTHROPIC_API_KEY"]
        _ = env["OPENAI_API_KEY"]
        _ = env["MINDBRIDGE_PROVIDER"]
        _ = env["MINDBRIDGE_MODEL"]

        return env
    }

    private func waitUntilReady(_ url: URL) async -> Bool {
        for _ in 0 ..< 80 {
            if await isReachable(url) {
                return true
            }
            try? await Task.sleep(nanoseconds: 350_000_000)
        }
        return false
    }

    private func isReachable(_ url: URL) async -> Bool {
        var request = URLRequest(url: url)
        request.timeoutInterval = 1.2
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else { return false }
            return (200 ..< 500).contains(http.statusCode)
        } catch {
            return false
        }
    }

    private func fail(_ message: String) {
        status = "Could not start MindBridge."
        errorMessage = message
        stop()
    }
}

struct RootView: View {
    @EnvironmentObject private var server: StreamlitServer

    var body: some View {
        ZStack {
            appBackground.ignoresSafeArea()
            if let url = server.appURL {
                WebAppView(url: url)
                    .ignoresSafeArea(edges: .bottom)
            } else {
                LaunchStateView()
            }
        }
    }
}

struct LaunchStateView: View {
    @EnvironmentObject private var server: StreamlitServer

    var body: some View {
        VStack(spacing: 18) {
            Text("MindBridge")
                .font(.system(size: 44, weight: .bold))
                .foregroundStyle(ink)

            Text(server.status)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(muted)

            if let error = server.errorMessage {
                Text(error)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Color(red: 0.75, green: 0.12, blue: 0.12))
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 620)
                    .padding(14)
                    .background(Color(red: 1.0, green: 0.94, blue: 0.94))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                ProgressView()
                    .controlSize(.large)
                    .tint(accent)
            }

            Text("The app starts the local Streamlit website and displays it in this native window.")
                .font(.system(size: 13, weight: .regular))
                .foregroundStyle(muted)
        }
        .padding(30)
    }
}

struct WebAppView: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.allowsBackForwardNavigationGestures = true
        view.setValue(false, forKey: "drawsBackground")
        view.load(URLRequest(url: url))
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        if view.url?.absoluteString != url.absoluteString {
            view.load(URLRequest(url: url))
        }
    }
}

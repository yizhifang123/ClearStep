import Charts
import Foundation
import SwiftUI

private let signalColor = Color(red: 0.26, green: 0.84, blue: 0.78)
private let blueColor = Color(red: 0.54, green: 0.63, blue: 1.0)
private let panelColor = Color(red: 0.08, green: 0.11, blue: 0.13)
private let panelBorder = Color.white.opacity(0.12)
private let mutedText = Color(red: 0.66, green: 0.74, blue: 0.72)
private let decisionButtonTopSpacer: CGFloat = 8
private let geminiEndpointBase = "https://generativelanguage.googleapis.com/v1beta/models/"

struct RealCaseBundle: Decodable {
    let schemaVersion: Int
    let generatedAtUTC: String
    let condition: String
    let metrics: Metrics
    let subjects: [SubjectRecord]
}

struct Metrics: Decodable {
    let nSubjects: Int
    let nMdd: Int
    let aucLoso: Double
    let accuracyLoso: Double
    let sensitivity: Double
    let specificity: Double
}

struct SubjectRecord: Decodable, Identifiable, Hashable {
    let id: String
    let label: String
    let probabilityMDD: Double
    let contributions: [FeatureContribution]

    var displayName: String {
        "\(id) (\(label))"
    }
}

struct FeatureContribution: Decodable, Identifiable, Hashable {
    let feature: String
    let value: Double
    let direction: String

    var id: String { feature }
}

struct DecisionRecord: Identifiable {
    let id = UUID()
    let time: Date
    let subjectID: String
    let decision: String
}

enum GeminiDraftKind: String, CaseIterable, Identifiable {
    case summary
    case caveats
    case handoff

    var id: String { rawValue }

    var buttonTitle: String {
        switch self {
        case .summary:
            return "Generate Summary"
        case .caveats:
            return "Caveats"
        case .handoff:
            return "Handoff Note"
        }
    }

    var outputTitle: String {
        switch self {
        case .summary:
            return "AI case summary"
        case .caveats:
            return "AI caveats"
        case .handoff:
            return "AI handoff note"
        }
    }
}

enum GeminiError: LocalizedError {
    case missingAPIKey
    case invalidURL
    case invalidResponse
    case apiError(String)
    case emptyResponse

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "Add GEMINI_API_KEY to the launch environment or paste a session key."
        case .invalidURL:
            return "Gemini endpoint could not be constructed."
        case .invalidResponse:
            return "Gemini returned an invalid response."
        case .apiError(let message):
            return message
        case .emptyResponse:
            return "Gemini returned no text."
        }
    }
}

struct GeminiClient {
    let apiKey: String
    let model: String

    func generate(prompt: String) async throws -> String {
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else {
            throw GeminiError.missingAPIKey
        }

        guard var components = URLComponents(string: "\(geminiEndpointBase)\(model):generateContent") else {
            throw GeminiError.invalidURL
        }
        components.queryItems = [URLQueryItem(name: "key", value: trimmedKey)]

        guard let url = components.url else {
            throw GeminiError.invalidURL
        }

        let body = GeminiRequest(
            systemInstruction: GeminiContent(
                role: nil,
                parts: [
                    GeminiPart(
                        text: """
                        You are a concise clinical research demo assistant. Do not diagnose. Do not recommend medication. Do not claim clinical validation. Use the phrase research prototype when safety context is needed.
                        """
                    )
                ]
            ),
            contents: [
                GeminiContent(
                    role: "user",
                    parts: [GeminiPart(text: prompt)]
                )
            ],
            generationConfig: GeminiGenerationConfig(
                temperature: 0.2,
                maxOutputTokens: 420
            )
        )

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw GeminiError.invalidResponse
        }

        if !(200 ..< 300).contains(http.statusCode) {
            if let decoded = try? JSONDecoder().decode(GeminiAPIErrorResponse.self, from: data) {
                throw GeminiError.apiError(decoded.error.message)
            }
            throw GeminiError.apiError("Gemini request failed with HTTP \(http.statusCode).")
        }

        let decoded = try JSONDecoder().decode(GeminiResponse.self, from: data)
        let parts = decoded.candidates?.flatMap { $0.content.parts } ?? []
        let text = parts
            .compactMap { $0.text }
            .joined(separator: "\n\n")
            .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)

        guard !text.isEmpty else {
            throw GeminiError.emptyResponse
        }
        return text
    }
}

struct GeminiRequest: Encodable {
    let systemInstruction: GeminiContent
    let contents: [GeminiContent]
    let generationConfig: GeminiGenerationConfig
}

struct GeminiContent: Codable {
    let role: String?
    let parts: [GeminiPart]
}

struct GeminiPart: Codable {
    let text: String?
}

struct GeminiGenerationConfig: Encodable {
    let temperature: Double
    let maxOutputTokens: Int
}

struct GeminiResponse: Decodable {
    let candidates: [GeminiCandidate]?
}

struct GeminiCandidate: Decodable {
    let content: GeminiContent
}

struct GeminiAPIErrorResponse: Decodable {
    let error: GeminiAPIError
}

struct GeminiAPIError: Decodable {
    let message: String
}

enum GeminiPromptFactory {
    static func prompt(
        kind: GeminiDraftKind,
        subject: SubjectRecord,
        metrics: Metrics,
        condition: String
    ) -> String {
        let contributionLines = subject.contributions.map {
            "\($0.feature): \(String(format: "%+.3f", $0.value)) toward \($0.direction)"
        }.joined(separator: "\n")

        let sharedContext = """
        Selected held-out subject: \(subject.id)
        Demo transparency label: \(subject.label)
        P(MDD), leave-one-subject-out estimate: \(subject.probabilityMDD.formatted(.percent.precision(.fractionLength(0))))
        Validation context: AUC \(String(format: "%.2f", metrics.aucLoso)), sensitivity \(metrics.sensitivity.formatted(.percent.precision(.fractionLength(0)))), specificity \(metrics.specificity.formatted(.percent.precision(.fractionLength(0)))), \(metrics.nSubjects) subjects, condition \(condition).
        Top model feature contributions:
        \(contributionLines)

        Safety rules:
        - Do not diagnose.
        - Do not recommend medication.
        - Do not suggest treatment selection.
        - Say this is a research prototype and not for clinical use.
        - Keep the output concise and presentation-ready.
        """

        switch kind {
        case .summary:
            return """
            \(sharedContext)

            Write a plain-English case summary for a hackathon demo judge or clinician reviewer. Use 4 short bullets: model signal, confidence, main feature drivers, and next review action.
            """
        case .caveats:
            return """
            \(sharedContext)

            Write the main safety caveats for this selected case. Use 5 short bullets and focus on uncertainty, adult-only training data, subject-wise validation, and human review.
            """
        case .handoff:
            return """
            \(sharedContext)

            Draft a handoff note that a reviewer could paste into a demo chart. Use neutral language. Include signal, drivers, limitations, and reviewer action. Do not exceed 120 words.
            """
        }
    }
}

enum BundleLoader {
    static func load() throws -> RealCaseBundle {
        let decoder = JSONDecoder()
        let resourceURL = Bundle.main.url(forResource: "real_case_bundle", withExtension: "json")

        if let resourceURL {
            let data = try Data(contentsOf: resourceURL)
            return try decoder.decode(RealCaseBundle.self, from: data)
        }

        let fallback = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("native_macos/Resources/real_case_bundle.json")
        let data = try Data(contentsOf: fallback)
        return try decoder.decode(RealCaseBundle.self, from: data)
    }
}

@main
struct RealCaseApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
                .frame(minWidth: 1120, minHeight: 740)
        }
        .windowStyle(.hiddenTitleBar)
    }
}

struct RootView: View {
    @State private var bundle: RealCaseBundle?
    @State private var loadError: String?

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.04, green: 0.06, blue: 0.07),
                    Color(red: 0.07, green: 0.10, blue: 0.11),
                    Color(red: 0.05, green: 0.08, blue: 0.09)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            if let bundle {
                DashboardView(bundle: bundle)
            } else {
                LoadingView(message: loadError)
            }
        }
        .preferredColorScheme(.dark)
        .task {
            do {
                bundle = try BundleLoader.load()
            } catch {
                loadError = error.localizedDescription
            }
        }
    }
}

struct DashboardView: View {
    let bundle: RealCaseBundle
    @State private var selectedSubjectID: SubjectRecord.ID
    @State private var decisions: [DecisionRecord] = []
    @State private var geminiAPIKey = ProcessInfo.processInfo.environment["GEMINI_API_KEY"] ?? ""
    @State private var geminiModel = "gemini-2.0-flash"

    init(bundle: RealCaseBundle) {
        self.bundle = bundle
        _selectedSubjectID = State(initialValue: bundle.subjects.first?.id ?? "")
    }

    private var selectedSubject: SubjectRecord {
        bundle.subjects.first(where: { $0.id == selectedSubjectID }) ?? bundle.subjects[0]
    }

    var body: some View {
        ScrollView(.vertical) {
            VStack(alignment: .leading, spacing: 18) {
                HeaderView(generatedAt: bundle.generatedAtUTC)

                Picker("Held-out subject", selection: $selectedSubjectID) {
                    ForEach(bundle.subjects) { subject in
                        Text(subject.displayName).tag(subject.id)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(width: 280)

                HStack(alignment: .top, spacing: 18) {
                    VStack(spacing: 18) {
                        ProbabilityPanel(subject: selectedSubject)
                        MetricsPanel(metrics: bundle.metrics, condition: bundle.condition)
                        DecisionPanel(subject: selectedSubject, decisions: $decisions)
                    }
                    .frame(width: 420)

                    FeaturePanel(subject: selectedSubject)
                        .frame(maxWidth: .infinity)
                }

                GeminiPanel(
                    subject: selectedSubject,
                    metrics: bundle.metrics,
                    condition: bundle.condition,
                    apiKey: $geminiAPIKey,
                    model: $geminiModel
                )
            }
            .padding(26)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .scrollIndicators(.visible)
    }
}

struct HeaderView: View {
    let generatedAt: String

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Depression EEG Decision Support")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(.white)
                    Text("Real Case Console")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(mutedText)
                }
                Spacer()
                Text("Artifact \(generatedAt)")
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(mutedText)
            }
            .padding(14)
            .background(panelColor.opacity(0.78))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(panelBorder))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            VStack(alignment: .leading, spacing: 9) {
                Text("Real Case EEG Console")
                    .font(.system(size: 52, weight: .bold))
                    .foregroundStyle(.white)
                    .tracking(0)

                Text("Native review surface for the validated Layer A workflow: held-out EEG subject, calibrated MDD-vs-HC probability, feature drivers, and clinician action capture.")
                    .font(.system(size: 16, weight: .regular))
                    .foregroundStyle(Color(red: 0.75, green: 0.82, blue: 0.80))
                    .fixedSize(horizontal: false, vertical: true)

                Text("Research prototype. Not for clinical use. This app supports review of an EEG classifier signal and does not diagnose, prescribe, or predict treatment response.")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Color(red: 0.98, green: 0.86, blue: 0.64))
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(red: 0.35, green: 0.25, blue: 0.08).opacity(0.22))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(red: 0.94, green: 0.74, blue: 0.36).opacity(0.35)))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
    }
}

struct ProbabilityPanel: View {
    let subject: SubjectRecord

    private var confidenceText: String {
        if (0.40 ... 0.60).contains(subject.probabilityMDD) {
            return "Insufficient evidence. Send to clinician review before using this signal."
        }
        if subject.probabilityMDD >= 0.5 {
            return "Higher-confidence signal leaning toward MDD in this held-out estimate."
        }
        return "Higher-confidence signal leaning toward healthy control in this held-out estimate."
    }

    private var needsReview: Bool {
        (0.40 ... 0.60).contains(subject.probabilityMDD)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("P(MDD), honest LOSO estimate")
                .font(.system(size: 12, weight: .bold))
                .foregroundStyle(mutedText)

            Text(subject.probabilityMDD, format: .percent.precision(.fractionLength(0)))
                .font(.system(size: 72, weight: .heavy, design: .rounded))
                .foregroundStyle(.white)
                .monospacedDigit()

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.10))
                    Capsule()
                        .fill(LinearGradient(colors: [blueColor, signalColor], startPoint: .leading, endPoint: .trailing))
                        .frame(width: geometry.size.width * subject.probabilityMDD)
                }
            }
            .frame(height: 10)

            Text("Ground truth for demo transparency: \(subject.label)")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(Color(red: 0.72, green: 0.79, blue: 0.77))

            Text(confidenceText)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(needsReview ? Color(red: 0.98, green: 0.86, blue: 0.64) : .white)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(needsReview ? Color(red: 0.42, green: 0.31, blue: 0.10).opacity(0.26) : signalColor.opacity(0.10))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(needsReview ? Color(red: 0.94, green: 0.74, blue: 0.36).opacity(0.35) : signalColor.opacity(0.28)))
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .panelStyle()
    }
}

struct MetricsPanel: View {
    let metrics: Metrics
    let condition: String

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Validation context")
                .font(.system(size: 18, weight: .bold))
                .foregroundStyle(.white)

            HStack(spacing: 10) {
                MetricTile(label: "LOSO AUC", value: String(format: "%.2f", metrics.aucLoso))
                MetricTile(label: "Sensitivity", value: metrics.sensitivity.formatted(.percent.precision(.fractionLength(0))))
                MetricTile(label: "Specificity", value: metrics.specificity.formatted(.percent.precision(.fractionLength(0))))
            }

            Text("\(metrics.nSubjects) subjects, \(metrics.nMdd) MDD, condition \(condition). Subject-wise validation only.")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(mutedText)
        }
        .panelStyle()
    }
}

struct MetricTile: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(mutedText)
            Text(value)
                .font(.system(size: 22, weight: .bold, design: .monospaced))
                .foregroundStyle(.white)
                .minimumScaleFactor(0.75)
                .lineLimit(1)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.045))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(panelBorder))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct DecisionPanel: View {
    let subject: SubjectRecord
    @Binding var decisions: [DecisionRecord]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Clinician action")
                .font(.system(size: 18, weight: .bold))
                .foregroundStyle(.white)
            Text("The model suggests. The reviewer records the decision.")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(mutedText)

            HStack(spacing: 10) {
                DecisionButton(title: "Confirm") { record("confirmed") }
                DecisionButton(title: "Override") { record("overridden") }
                DecisionButton(title: "Review") { record("review_requested") }
            }
            .padding(.top, decisionButtonTopSpacer)

            if decisions.isEmpty {
                Text("No actions recorded in this session.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(mutedText)
            } else {
                VStack(spacing: 7) {
                    ForEach(decisions.suffix(3).reversed()) { decision in
                        HStack {
                            Text(decision.subjectID)
                                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                                .foregroundStyle(.white)
                            Spacer()
                            Text(decision.decision)
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(mutedText)
                        }
                        .padding(9)
                        .background(Color.white.opacity(0.045))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
        }
        .panelStyle()
    }

    private func record(_ decision: String) {
        decisions.append(DecisionRecord(time: Date(), subjectID: subject.id, decision: decision))
    }
}

struct DecisionButton: View {
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 13, weight: .bold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .background(signalColor.opacity(0.13))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(signalColor.opacity(0.38)))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct FeaturePanel: View {
    let subject: SubjectRecord

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Feature drivers")
                .font(.system(size: 22, weight: .bold))
                .foregroundStyle(.white)
            Text("Positive values push toward MDD. Negative values push toward HC.")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(mutedText)

            Chart(subject.contributions) { contribution in
                BarMark(
                    x: .value("Contribution", contribution.value),
                    y: .value("Feature", contribution.feature)
                )
                .foregroundStyle(contribution.value >= 0 ? signalColor : blueColor)
                .cornerRadius(4)
            }
            .chartXAxis {
                AxisMarks { _ in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                        .foregroundStyle(Color.white.opacity(0.08))
                    AxisValueLabel()
                        .foregroundStyle(mutedText)
                }
            }
            .chartYAxis {
                AxisMarks { _ in
                    AxisValueLabel()
                        .foregroundStyle(Color(red: 0.75, green: 0.82, blue: 0.80))
                }
            }
            .frame(minHeight: 360)
            .padding(.vertical, 8)

            VStack(spacing: 8) {
                ForEach(subject.contributions) { contribution in
                    HStack {
                        Text(contribution.feature)
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(.white)
                        Spacer()
                        Text(String(format: "%+.3f", contribution.value))
                            .font(.system(size: 13, weight: .bold, design: .monospaced))
                            .foregroundStyle(contribution.value >= 0 ? signalColor : blueColor)
                    }
                    .padding(.vertical, 7)
                    .padding(.horizontal, 9)
                    .background(Color.white.opacity(0.04))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }
        }
        .panelStyle()
    }
}

struct GeminiPanel: View {
    let subject: SubjectRecord
    let metrics: Metrics
    let condition: String
    @Binding var apiKey: String
    @Binding var model: String

    @State private var outputTitle = "AI explanation"
    @State private var outputText = "Choose a Gemini action to generate clinician-facing explanation text for the selected case."
    @State private var errorMessage: String?
    @State private var isGenerating = false

    private var hasKey: Bool {
        !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Gemini assistant")
                        .font(.system(size: 22, weight: .bold))
                        .foregroundStyle(.white)
                    Text("Explanation layer only. The local model remains the source of truth.")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(mutedText)
                }
                Spacer()
                if isGenerating {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            HStack(spacing: 12) {
                SecureField("GEMINI_API_KEY", text: $apiKey)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .background(Color.white.opacity(0.06))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(panelBorder))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .frame(maxWidth: .infinity)

                TextField("Model", text: $model)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .background(Color.white.opacity(0.06))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(panelBorder))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .frame(width: 190)
            }

            Text(hasKey ? "Using the launch environment key or the session key above. The key is not written into project files." : "Set GEMINI_API_KEY before launch or paste a session key above.")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(hasKey ? mutedText : Color(red: 0.98, green: 0.86, blue: 0.64))

            HStack(spacing: 10) {
                ForEach(GeminiDraftKind.allCases) { kind in
                    Button {
                        Task { await generate(kind) }
                    } label: {
                        Text(kind.buttonTitle)
                            .font(.system(size: 13, weight: .bold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.white)
                    .background(signalColor.opacity(0.13))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(signalColor.opacity(0.36)))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .disabled(isGenerating || !hasKey)
                    .opacity(isGenerating || !hasKey ? 0.48 : 1)
                }
            }

            if let errorMessage {
                Text(errorMessage)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Color(red: 0.98, green: 0.50, blue: 0.50))
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(red: 0.42, green: 0.08, blue: 0.08).opacity(0.22))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(red: 0.98, green: 0.50, blue: 0.50).opacity(0.28)))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            VStack(alignment: .leading, spacing: 10) {
                Text(outputTitle)
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.white)
                Text(outputText)
                    .font(.system(size: 14, weight: .regular))
                    .foregroundStyle(Color(red: 0.79, green: 0.86, blue: 0.84))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.white.opacity(0.045))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(panelBorder))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .panelStyle()
    }

    @MainActor
    private func generate(_ kind: GeminiDraftKind) async {
        isGenerating = true
        errorMessage = nil
        outputTitle = kind.outputTitle
        outputText = "Generating..."

        do {
            let prompt = GeminiPromptFactory.prompt(
                kind: kind,
                subject: subject,
                metrics: metrics,
                condition: condition
            )
            let client = GeminiClient(apiKey: apiKey, model: model)
            outputText = try await client.generate(prompt: prompt)
        } catch {
            outputText = "Generation did not complete."
            errorMessage = error.localizedDescription
        }

        isGenerating = false
    }
}

struct LoadingView: View {
    let message: String?

    var body: some View {
        VStack(spacing: 14) {
            Text("Real Case EEG Console")
                .font(.system(size: 32, weight: .bold))
                .foregroundStyle(.white)
            Text(message ?? "Loading native case bundle.")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(mutedText)
        }
        .padding(28)
        .panelStyle()
    }
}

private extension View {
    func panelStyle() -> some View {
        self
            .padding(18)
            .background(panelColor.opacity(0.82))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(panelBorder))
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

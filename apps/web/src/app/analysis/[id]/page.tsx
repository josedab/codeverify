import Link from "next/link";
import { ArrowLeft, CheckCircle, XCircle, Clock, FileCode, AlertTriangle } from "lucide-react";

// Mock data - would be fetched from API using params.id
const analysis = {
  id: "a1b2c3d4",
  repo: "acme/api-service",
  prNumber: 423,
  prTitle: "Add user authentication endpoints",
  headSha: "abc123def456",
  status: "completed",
  conclusion: "passed",
  startedAt: "2026-01-28T10:27:26Z",
  completedAt: "2026-01-28T10:30:00Z",
  stages: [
    { name: "fetch", status: "completed", durationMs: 1234 },
    { name: "parse", status: "completed", durationMs: 856 },
    { name: "semantic", status: "completed", durationMs: 45000 },
    { name: "verify", status: "completed", durationMs: 12000 },
    { name: "security", status: "completed", durationMs: 38000 },
    { name: "synthesize", status: "completed", durationMs: 5000 },
  ],
  findings: [
    {
      id: "f1",
      category: "security",
      severity: "medium",
      title: "Potential SQL injection in user query",
      description: "The user_id parameter is used directly in SQL query without proper sanitization.",
      filePath: "src/auth/users.py",
      lineStart: 42,
      lineEnd: 42,
      confidence: 0.87,
      verificationType: "ai",
      fixSuggestion: 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
    },
    {
      id: "f2",
      category: "logic_error",
      severity: "low",
      title: "Missing null check before access",
      description: "The user object may be null when accessing the email property.",
      filePath: "src/auth/session.py",
      lineStart: 78,
      lineEnd: 78,
      confidence: 0.72,
      verificationType: "formal",
      fixSuggestion: "if user is not None:\n    email = user.email",
    },
  ],
  summary: {
    total_issues: 2,
    critical: 0,
    high: 0,
    medium: 1,
    low: 1,
    pass: true,
  },
};

export default function AnalysisDetailPage({ params }: { params: { id: string } }) {
  const passed = analysis.summary.pass;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/dashboard/analyses" className="text-gray-500 hover:text-gray-700">
                <ArrowLeft className="h-5 w-5" />
              </Link>
              <div>
                <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                  {analysis.repo} #{analysis.prNumber}
                </h1>
                <p className="text-sm text-gray-500">{analysis.prTitle}</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {passed ? (
                <span className="flex items-center space-x-2 px-4 py-2 bg-green-100 text-green-800 rounded-lg">
                  <CheckCircle className="h-5 w-5" />
                  <span>Passed</span>
                </span>
              ) : (
                <span className="flex items-center space-x-2 px-4 py-2 bg-red-100 text-red-800 rounded-lg">
                  <XCircle className="h-5 w-5" />
                  <span>Failed</span>
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <SummaryCard
            label="Total Issues"
            value={analysis.summary.total_issues}
            icon={<AlertTriangle className="h-5 w-5" />}
          />
          <SummaryCard
            label="Critical/High"
            value={analysis.summary.critical + analysis.summary.high}
            className={analysis.summary.critical + analysis.summary.high > 0 ? "text-red-600" : "text-green-600"}
          />
          <SummaryCard
            label="Medium/Low"
            value={analysis.summary.medium + analysis.summary.low}
          />
          <SummaryCard
            label="Duration"
            value={formatDuration(analysis.stages.reduce((sum, s) => sum + s.durationMs, 0))}
            icon={<Clock className="h-5 w-5" />}
          />
        </div>

        {/* Pipeline Stages */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-8">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Analysis Pipeline
          </h2>
          <div className="flex items-center justify-between">
            {analysis.stages.map((stage, index) => (
              <div key={stage.name} className="flex items-center">
                <div className="flex flex-col items-center">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                    stage.status === "completed" ? "bg-green-100 text-green-600" : "bg-gray-100 text-gray-400"
                  }`}>
                    <CheckCircle className="h-5 w-5" />
                  </div>
                  <p className="mt-2 text-sm font-medium text-gray-900 dark:text-white capitalize">
                    {stage.name}
                  </p>
                  <p className="text-xs text-gray-500">
                    {formatDuration(stage.durationMs)}
                  </p>
                </div>
                {index < analysis.stages.length - 1 && (
                  <div className="w-24 h-0.5 bg-green-200 mx-2"></div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Findings */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Findings ({analysis.findings.length})
            </h2>
          </div>
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {analysis.findings.map((finding) => (
              <div key={finding.id} className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start space-x-3">
                    <SeverityBadge severity={finding.severity} />
                    <div>
                      <h3 className="font-medium text-gray-900 dark:text-white">
                        {finding.title}
                      </h3>
                      <p className="text-sm text-gray-500 flex items-center mt-1">
                        <FileCode className="h-4 w-4 mr-1" />
                        {finding.filePath}:{finding.lineStart}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">
                      {finding.verificationType}
                    </span>
                    <span className="text-xs text-gray-500">
                      {Math.round(finding.confidence * 100)}% confidence
                    </span>
                  </div>
                </div>
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  {finding.description}
                </p>
                {finding.fixSuggestion && (
                  <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Suggested fix:
                    </p>
                    <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
                      <code>{finding.fixSuggestion}</code>
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}

function SummaryCard({
  label,
  value,
  icon,
  className = "",
}: {
  label: string;
  value: number | string;
  icon?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-center justify-between">
        <span className="text-gray-500">{icon}</span>
      </div>
      <p className={`text-2xl font-bold mt-2 ${className || "text-gray-900 dark:text-white"}`}>
        {value}
      </p>
      <p className="text-sm text-gray-500">{label}</p>
    </div>
  );
}

function SeverityBadge({ severity }: { severity: string }) {
  const styles: Record<string, string> = {
    critical: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
    high: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
    medium: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
    low: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  };

  return (
    <span className={`px-2 py-1 text-xs font-medium rounded-full ${styles[severity] || styles.low}`}>
      {severity}
    </span>
  );
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const mins = Math.floor(ms / 60000);
  const secs = Math.round((ms % 60000) / 1000);
  return `${mins}m ${secs}s`;
}

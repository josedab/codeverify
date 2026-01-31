import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Clock, GitPullRequest } from "lucide-react";

// Mock data - in production, fetch from API
const stats = {
  totalAnalyses: 1247,
  passRate: 89,
  issuesFound: 342,
  criticalIssues: 12,
  avgDuration: "2.4 min",
  prsCovered: 98,
};

const recentAnalyses = [
  {
    id: "1",
    repo: "acme/api-service",
    pr: 423,
    status: "passed",
    findings: 2,
    time: "5 min ago",
  },
  {
    id: "2",
    repo: "acme/web-app",
    pr: 891,
    status: "failed",
    findings: 5,
    time: "12 min ago",
  },
  {
    id: "3",
    repo: "acme/data-pipeline",
    pr: 156,
    status: "passed",
    findings: 0,
    time: "25 min ago",
  },
  {
    id: "4",
    repo: "acme/mobile-app",
    pr: 78,
    status: "running",
    findings: null,
    time: "Just now",
  },
];

export default function DashboardPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">
        Dashboard
      </h1>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <StatCard
          title="Total Analyses"
          value={stats.totalAnalyses.toLocaleString()}
          change="+15%"
          trending="up"
          icon={<GitPullRequest className="h-6 w-6" />}
        />
        <StatCard
          title="Pass Rate"
          value={`${stats.passRate}%`}
          change="+3%"
          trending="up"
          icon={<CheckCircle className="h-6 w-6" />}
        />
        <StatCard
          title="Issues Found"
          value={stats.issuesFound.toLocaleString()}
          change="-8%"
          trending="down"
          icon={<AlertTriangle className="h-6 w-6" />}
        />
        <StatCard
          title="Critical Issues"
          value={stats.criticalIssues.toString()}
          change="-40%"
          trending="down"
          icon={<AlertTriangle className="h-6 w-6 text-red-500" />}
        />
        <StatCard
          title="Avg. Duration"
          value={stats.avgDuration}
          change="-12%"
          trending="down"
          icon={<Clock className="h-6 w-6" />}
        />
        <StatCard
          title="PRs Covered"
          value={`${stats.prsCovered}%`}
          change="+5%"
          trending="up"
          icon={<GitPullRequest className="h-6 w-6" />}
        />
      </div>

      {/* Recent Analyses */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Recent Analyses
          </h2>
        </div>
        <div className="divide-y divide-gray-200 dark:divide-gray-700">
          {recentAnalyses.map((analysis) => (
            <div
              key={analysis.id}
              className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700 transition cursor-pointer"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <StatusBadge status={analysis.status} />
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">
                      {analysis.repo}
                    </p>
                    <p className="text-sm text-gray-500">PR #{analysis.pr}</p>
                  </div>
                </div>
                <div className="text-right">
                  {analysis.findings !== null && (
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      {analysis.findings} {analysis.findings === 1 ? "finding" : "findings"}
                    </p>
                  )}
                  <p className="text-xs text-gray-400">{analysis.time}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  change,
  trending,
  icon,
}: {
  title: string;
  value: string;
  change: string;
  trending: "up" | "down";
  icon: React.ReactNode;
}) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <span className="text-gray-500 dark:text-gray-400">{icon}</span>
        <span
          className={`flex items-center text-sm ${
            trending === "up" ? "text-green-500" : "text-red-500"
          }`}
        >
          {trending === "up" ? (
            <TrendingUp className="h-4 w-4 mr-1" />
          ) : (
            <TrendingDown className="h-4 w-4 mr-1" />
          )}
          {change}
        </span>
      </div>
      <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
      <p className="text-sm text-gray-500 dark:text-gray-400">{title}</p>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles = {
    passed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
    failed: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
    running: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  };

  return (
    <span
      className={`px-2 py-1 text-xs font-medium rounded-full ${
        styles[status as keyof typeof styles] || styles.running
      }`}
    >
      {status}
    </span>
  );
}

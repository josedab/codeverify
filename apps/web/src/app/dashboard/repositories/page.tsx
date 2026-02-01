import Link from "next/link";
import { GitBranch, Settings, CheckCircle, XCircle, Clock } from "lucide-react";

// Mock data
const repositories = [
  {
    id: "r1",
    name: "api-service",
    full_name: "acme/api-service",
    default_branch: "main",
    enabled: true,
    last_analysis: {
      conclusion: "passed",
      pr_number: 423,
      created_at: "2026-01-28T10:30:00Z",
    },
    stats: { total: 45, passed: 42, failed: 3 },
  },
  {
    id: "r2",
    name: "web-app",
    full_name: "acme/web-app",
    default_branch: "main",
    enabled: true,
    last_analysis: {
      conclusion: "failed",
      pr_number: 891,
      created_at: "2026-01-28T10:15:00Z",
    },
    stats: { total: 123, passed: 98, failed: 25 },
  },
  {
    id: "r3",
    name: "data-pipeline",
    full_name: "acme/data-pipeline",
    default_branch: "develop",
    enabled: false,
    last_analysis: null,
    stats: { total: 0, passed: 0, failed: 0 },
  },
];

export default function RepositoriesPage() {
  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Repositories
        </h1>
        <button className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700">
          Add Repository
        </button>
      </div>

      <div className="grid gap-4">
        {repositories.map((repo) => (
          <div
            key={repo.id}
            className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-4">
                <div className={`p-3 rounded-lg ${
                  repo.enabled 
                    ? "bg-primary-100 dark:bg-primary-900" 
                    : "bg-gray-100 dark:bg-gray-700"
                }`}>
                  <GitBranch className={`h-6 w-6 ${
                    repo.enabled 
                      ? "text-primary-600" 
                      : "text-gray-400"
                  }`} />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {repo.full_name}
                  </h3>
                  <p className="text-sm text-gray-500">
                    Default branch: {repo.default_branch}
                  </p>
                  <div className="flex items-center space-x-4 mt-3">
                    {repo.enabled ? (
                      <span className="flex items-center text-sm text-green-600">
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Enabled
                      </span>
                    ) : (
                      <span className="flex items-center text-sm text-gray-400">
                        <XCircle className="h-4 w-4 mr-1" />
                        Disabled
                      </span>
                    )}
                    {repo.last_analysis && (
                      <span className="flex items-center text-sm text-gray-500">
                        <Clock className="h-4 w-4 mr-1" />
                        Last: PR #{repo.last_analysis.pr_number}
                        <span className={`ml-2 px-2 py-0.5 text-xs rounded ${
                          repo.last_analysis.conclusion === "passed"
                            ? "bg-green-100 text-green-800"
                            : "bg-red-100 text-red-800"
                        }`}>
                          {repo.last_analysis.conclusion}
                        </span>
                      </span>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <div className="text-right">
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {repo.stats.total}
                  </p>
                  <p className="text-sm text-gray-500">analyses</p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-green-600">
                    {repo.stats.total > 0 
                      ? Math.round((repo.stats.passed / repo.stats.total) * 100) 
                      : 0}%
                  </p>
                  <p className="text-sm text-gray-500">pass rate</p>
                </div>
                <Link
                  href={`/dashboard/repositories/${repo.id}/settings`}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
                >
                  <Settings className="h-5 w-5" />
                </Link>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Empty state */}
      {repositories.length === 0 && (
        <div className="text-center py-12">
          <GitBranch className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No repositories yet
          </h3>
          <p className="text-gray-500 mb-4">
            Add a repository to start analyzing pull requests
          </p>
          <button className="px-4 py-2 bg-primary-600 text-white rounded-lg">
            Add Repository
          </button>
        </div>
      )}
    </div>
  );
}

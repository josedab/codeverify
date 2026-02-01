import Link from "next/link";
import { Filter, Search, ChevronRight } from "lucide-react";

// Mock data
const analyses = [
  {
    id: "a1b2c3d4",
    repo: "acme/api-service",
    prNumber: 423,
    prTitle: "Add user authentication endpoints",
    status: "completed",
    conclusion: "passed",
    findings: { critical: 0, high: 0, medium: 2, low: 1 },
    duration: "2m 34s",
    createdAt: "2026-01-28T10:30:00Z",
  },
  {
    id: "e5f6g7h8",
    repo: "acme/web-app",
    prNumber: 891,
    prTitle: "Implement payment processing",
    status: "completed",
    conclusion: "failed",
    findings: { critical: 1, high: 2, medium: 1, low: 0 },
    duration: "3m 12s",
    createdAt: "2026-01-28T10:15:00Z",
  },
];

export default function AnalysesPage() {
  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Analyses
        </h1>
        <div className="flex space-x-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search analyses..."
              className="pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
            />
          </div>
          <button className="flex items-center space-x-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg">
            <Filter className="h-4 w-4" />
            <span>Filters</span>
          </button>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-50 dark:bg-gray-700">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Repository / PR</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Findings</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Duration</th>
              <th className="px-6 py-3"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {analyses.map((analysis) => (
              <tr key={analysis.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                <td className="px-6 py-4">
                  <p className="font-medium text-gray-900 dark:text-white">{analysis.repo}</p>
                  <p className="text-sm text-gray-500">#{analysis.prNumber}: {analysis.prTitle}</p>
                </td>
                <td className="px-6 py-4">
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    analysis.conclusion === "passed" 
                      ? "bg-green-100 text-green-800" 
                      : "bg-red-100 text-red-800"
                  }`}>
                    {analysis.conclusion}
                  </span>
                </td>
                <td className="px-6 py-4 text-sm text-gray-500">
                  {Object.values(analysis.findings).reduce((a, b) => a + b, 0)} issues
                </td>
                <td className="px-6 py-4 text-sm text-gray-500">{analysis.duration}</td>
                <td className="px-6 py-4">
                  <Link href={`/analysis/${analysis.id}`}>
                    <ChevronRight className="h-5 w-5 text-primary-600" />
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

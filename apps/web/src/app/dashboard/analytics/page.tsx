"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { format, parseISO } from "date-fns";
import {
  Users,
  TrendingUp,
  Activity,
  Trophy,
  Calendar,
  Target,
} from "lucide-react";
import api, { TeamStats, TrendsResponse, LeaderboardEntry } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent, Button, Select, Badge } from "@/components/ui";

// Mock organization ID - in real app, get from context/auth
const MOCK_ORG_ID = "00000000-0000-0000-0000-000000000001";

export default function AnalyticsPage() {
  const [period, setPeriod] = useState<"7d" | "30d" | "90d">("30d");
  const [leaderboardMetric, setLeaderboardMetric] = useState<
    "analyses" | "findings_fixed" | "activity"
  >("analyses");

  // Fetch team stats
  const { data: teamStats, isLoading: teamLoading } = useQuery({
    queryKey: ["teamStats", MOCK_ORG_ID],
    queryFn: () => api.getTeamStats(MOCK_ORG_ID),
  });

  // Fetch trends
  const { data: trends, isLoading: trendsLoading } = useQuery({
    queryKey: ["trends", MOCK_ORG_ID, period],
    queryFn: () => api.getTrends({ organization_id: MOCK_ORG_ID, period }),
  });

  // Fetch leaderboard
  const { data: leaderboard } = useQuery({
    queryKey: ["leaderboard", MOCK_ORG_ID, leaderboardMetric, period],
    queryFn: () =>
      api.getLeaderboard({
        organization_id: MOCK_ORG_ID,
        metric: leaderboardMetric,
        period,
      }),
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Team Analytics
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Track team performance and verification trends
          </p>
        </div>
        <Select
          value={period}
          onChange={(e) => setPeriod(e.target.value as "7d" | "30d" | "90d")}
        >
          <option value="7d">Last 7 days</option>
          <option value="30d">Last 30 days</option>
          <option value="90d">Last 90 days</option>
        </Select>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard
          title="Team Members"
          value={teamStats?.total_members?.toString() || "0"}
          subtitle={`${teamStats?.active_members_7d || 0} active this week`}
          icon={<Users className="h-5 w-5" />}
          color="blue"
        />
        <StatsCard
          title="Pass Rate"
          value={`${((trends?.summary.pass_rate || 0) * 100).toFixed(1)}%`}
          subtitle={`${trends?.summary.total_passed || 0} passed`}
          icon={<Target className="h-5 w-5" />}
          color="green"
        />
        <StatsCard
          title="Total Analyses"
          value={trends?.summary.total_analyses?.toString() || "0"}
          subtitle={`${trends?.summary.total_findings || 0} findings`}
          icon={<Activity className="h-5 w-5" />}
          color="purple"
        />
        <StatsCard
          title="Avg Findings/PR"
          value={trends?.summary.avg_findings_per_analysis?.toFixed(1) || "0"}
          subtitle="per analysis"
          icon={<TrendingUp className="h-5 w-5" />}
          color="orange"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trend Chart */}
        <Card padding="none">
          <CardHeader>
            <CardTitle>Analysis Trends</CardTitle>
          </CardHeader>
          <CardContent>
            {trendsLoading ? (
              <div className="h-64 flex items-center justify-center">
                <div className="animate-spin h-8 w-8 border-2 border-primary-500 border-t-transparent rounded-full" />
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={trends?.data || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) => format(parseISO(value), "MMM d")}
                    stroke="#9CA3AF"
                    fontSize={12}
                  />
                  <YAxis stroke="#9CA3AF" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "none",
                      borderRadius: "8px",
                    }}
                    labelFormatter={(value) =>
                      format(parseISO(value as string), "MMMM d, yyyy")
                    }
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="passed"
                    name="Passed"
                    stackId="1"
                    stroke="#10B981"
                    fill="#10B981"
                    fillOpacity={0.6}
                  />
                  <Area
                    type="monotone"
                    dataKey="failed"
                    name="Failed"
                    stackId="1"
                    stroke="#EF4444"
                    fill="#EF4444"
                    fillOpacity={0.6}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Findings Chart */}
        <Card padding="none">
          <CardHeader>
            <CardTitle>Findings Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            {trendsLoading ? (
              <div className="h-64 flex items-center justify-center">
                <div className="animate-spin h-8 w-8 border-2 border-primary-500 border-t-transparent rounded-full" />
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={trends?.data || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) => format(parseISO(value), "MMM d")}
                    stroke="#9CA3AF"
                    fontSize={12}
                  />
                  <YAxis stroke="#9CA3AF" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "none",
                      borderRadius: "8px",
                    }}
                    labelFormatter={(value) =>
                      format(parseISO(value as string), "MMMM d, yyyy")
                    }
                  />
                  <Line
                    type="monotone"
                    dataKey="findings"
                    name="Findings"
                    stroke="#8B5CF6"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Team Activity & Leaderboard */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Activity Heatmap */}
        <Card padding="none">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calendar className="h-5 w-5" />
              Daily Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            {teamLoading ? (
              <div className="h-32 flex items-center justify-center">
                <div className="animate-spin h-8 w-8 border-2 border-primary-500 border-t-transparent rounded-full" />
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={teamStats?.activity_by_day || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) => format(parseISO(value), "d")}
                    stroke="#9CA3AF"
                    fontSize={10}
                  />
                  <YAxis stroke="#9CA3AF" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "none",
                      borderRadius: "8px",
                    }}
                    labelFormatter={(value) =>
                      format(parseISO(value as string), "MMMM d")
                    }
                  />
                  <Bar
                    dataKey="count"
                    name="Analyses"
                    fill="#3B82F6"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Leaderboard */}
        <Card padding="none">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Trophy className="h-5 w-5 text-yellow-500" />
              Leaderboard
            </CardTitle>
            <Select
              value={leaderboardMetric}
              onChange={(e) =>
                setLeaderboardMetric(
                  e.target.value as "analyses" | "findings_fixed" | "activity"
                )
              }
            >
              <option value="analyses">Analyses</option>
              <option value="findings_fixed">Findings Fixed</option>
              <option value="activity">Activity Score</option>
            </Select>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {leaderboard?.entries.slice(0, 5).map((entry, index) => (
                <LeaderboardRow key={entry.user_id} entry={entry} index={index} />
              ))}
              {(!leaderboard || leaderboard.entries.length === 0) && (
                <p className="text-center text-gray-500 py-4">
                  No activity yet
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Team Members Table */}
      <Card padding="none">
        <CardHeader>
          <CardTitle>Team Members</CardTitle>
        </CardHeader>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900 dark:text-white">
                  Member
                </th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900 dark:text-white">
                  Analyses
                </th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900 dark:text-white">
                  Findings Dismissed
                </th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900 dark:text-white">
                  Last Active
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {teamStats?.members.map((member) => (
                <tr
                  key={member.user_id}
                  className="hover:bg-gray-50 dark:hover:bg-gray-800"
                >
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      {member.avatar_url ? (
                        <img
                          src={member.avatar_url}
                          alt={member.username}
                          className="h-8 w-8 rounded-full"
                        />
                      ) : (
                        <div className="h-8 w-8 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                          <Users className="h-4 w-4 text-gray-500" />
                        </div>
                      )}
                      <span className="font-medium text-gray-900 dark:text-white">
                        {member.username}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-gray-600 dark:text-gray-300">
                    {member.analyses_triggered}
                  </td>
                  <td className="px-6 py-4 text-gray-600 dark:text-gray-300">
                    {member.findings_dismissed}
                  </td>
                  <td className="px-6 py-4 text-gray-500 dark:text-gray-400 text-sm">
                    {member.last_active
                      ? format(parseISO(member.last_active), "MMM d, yyyy")
                      : "Never"}
                  </td>
                </tr>
              ))}
              {(!teamStats || teamStats.members.length === 0) && (
                <tr>
                  <td
                    colSpan={4}
                    className="px-6 py-8 text-center text-gray-500"
                  >
                    No team members found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function StatsCard({
  title,
  value,
  subtitle,
  icon,
  color,
}: {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
  color: "blue" | "green" | "purple" | "orange";
}) {
  const colorClasses = {
    blue: "bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-400",
    green: "bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-400",
    purple:
      "bg-purple-100 text-purple-600 dark:bg-purple-900 dark:text-purple-400",
    orange:
      "bg-orange-100 text-orange-600 dark:bg-orange-900 dark:text-orange-400",
  };

  return (
    <Card padding="md">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">{title}</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
            {value}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            {subtitle}
          </p>
        </div>
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>{icon}</div>
      </div>
    </Card>
  );
}

function LeaderboardRow({
  entry,
  index,
}: {
  entry: LeaderboardEntry;
  index: number;
}) {
  const medals = ["🥇", "🥈", "🥉"];
  const medal = medals[index];

  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex items-center gap-3">
        <span className="w-6 text-center text-lg">{medal || entry.rank}</span>
        {entry.avatar_url ? (
          <img
            src={entry.avatar_url}
            alt={entry.username}
            className="h-8 w-8 rounded-full"
          />
        ) : (
          <div className="h-8 w-8 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
            <Users className="h-4 w-4 text-gray-500" />
          </div>
        )}
        <span className="font-medium text-gray-900 dark:text-white">
          {entry.username}
        </span>
      </div>
      <Badge variant={index < 3 ? "success" : "default"}>{entry.score}</Badge>
    </div>
  );
}

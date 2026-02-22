"use client";

import { useState } from "react";
import {
  Trophy,
  Target,
  Flame,
  Award,
  BookOpen,
  Code,
  ChevronRight,
  CheckCircle,
  XCircle,
  Star,
} from "lucide-react";

// Types
interface WeaknessArea {
  category: string;
  finding_count: number;
  improvement_trend: string;
}

interface DeveloperProgress {
  developer_id: string;
  developer_name: string;
  lessons_completed: number;
  challenges_completed: number;
  total_points: number;
  current_streak: number;
  longest_streak: number;
  badges: string[];
  skill_level: string;
}

interface Challenge {
  id: string;
  category: string;
  title: string;
  description: string;
  buggy_code: string;
  hint: string;
  difficulty: string;
  points: number;
}

interface LeaderboardEntry {
  name: string;
  points: number;
  challenges: number;
  streak: number;
  badges: number;
}

// Mock data — in production, fetch from /api/v1/training/*
const mockProgress: DeveloperProgress = {
  developer_id: "dev-1",
  developer_name: "You",
  lessons_completed: 8,
  challenges_completed: 12,
  total_points: 145,
  current_streak: 5,
  longest_streak: 12,
  badges: ["first_fix", "streak_7"],
  skill_level: "intermediate",
};

const mockWeaknesses: WeaknessArea[] = [
  { category: "null_safety", finding_count: 8, improvement_trend: "improving" },
  { category: "injection", finding_count: 5, improvement_trend: "stable" },
  { category: "credential_exposure", finding_count: 3, improvement_trend: "declining" },
];

const mockChallenges: Challenge[] = [
  {
    id: "c1", category: "null_safety", title: "Fix the Null Dereference",
    description: "This function crashes when user is None.",
    buggy_code: "def greet(user):\n    return f'Hello, {user.name}'",
    hint: "Check if user is None before accessing .name", difficulty: "beginner", points: 10,
  },
  {
    id: "c2", category: "injection", title: "Fix the SQL Injection",
    description: "This query is vulnerable to SQL injection.",
    buggy_code: "def get_user(name):\n    query = f\"SELECT * FROM users WHERE name = '{name}'\"",
    hint: "Never use f-strings in SQL queries", difficulty: "intermediate", points: 20,
  },
  {
    id: "c3", category: "division_by_zero", title: "Fix the Division by Zero",
    description: "This function crashes when count is 0.",
    buggy_code: "def average(total, count):\n    return total / count",
    hint: "What happens when count is 0?", difficulty: "beginner", points: 10,
  },
];

const mockLeaderboard: LeaderboardEntry[] = [
  { name: "Alice", points: 280, challenges: 24, streak: 15, badges: 5 },
  { name: "You", points: 145, challenges: 12, streak: 5, badges: 2 },
  { name: "Bob", points: 120, challenges: 10, streak: 3, badges: 2 },
  { name: "Carol", points: 95, challenges: 8, streak: 0, badges: 1 },
  { name: "Dave", points: 60, challenges: 5, streak: 2, badges: 1 },
];

function StatCard({ icon: Icon, label, value, subtext }: {
  icon: typeof Trophy; label: string; value: string | number; subtext?: string;
}) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-primary-50 dark:bg-primary-900/20">
          <Icon className="h-5 w-5 text-primary-600" />
        </div>
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
          {subtext && <p className="text-xs text-gray-400">{subtext}</p>}
        </div>
      </div>
    </div>
  );
}

function SkillBadge({ badge }: { badge: string }) {
  const cfg: Record<string, { label: string; color: string }> = {
    first_fix: { label: "First Fix", color: "bg-green-100 text-green-800" },
    streak_7: { label: "7-Day Streak", color: "bg-orange-100 text-orange-800" },
    streak_30: { label: "30-Day Streak", color: "bg-red-100 text-red-800" },
    zero_critical: { label: "Zero Critical", color: "bg-blue-100 text-blue-800" },
  };
  const c = cfg[badge] || { label: badge, color: "bg-gray-100 text-gray-800" };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${c.color}`}>
      <Star className="h-3 w-3" /> {c.label}
    </span>
  );
}

export default function TrainingPage() {
  const [activeTab, setActiveTab] = useState<"overview" | "challenges" | "leaderboard">("overview");

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <Trophy className="h-7 w-7 text-primary-600" /> Security Training
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Learn from your real findings — fix vulnerabilities, earn badges, climb the leaderboard
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard icon={Target} label="Points" value={mockProgress.total_points} subtext={mockProgress.skill_level} />
        <StatCard icon={Code} label="Challenges" value={mockProgress.challenges_completed} />
        <StatCard icon={Flame} label="Streak" value={`${mockProgress.current_streak} days`} subtext={`Best: ${mockProgress.longest_streak}`} />
        <StatCard icon={BookOpen} label="Lessons" value={mockProgress.lessons_completed} />
      </div>

      {/* Badges */}
      <div className="mb-6 flex gap-2 flex-wrap">
        {mockProgress.badges.map((b) => <SkillBadge key={b} badge={b} />)}
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
        <nav className="flex gap-6">
          {(["overview", "challenges", "leaderboard"] as const).map((tab) => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`pb-3 text-sm font-medium border-b-2 transition ${
                activeTab === tab ? "border-primary-600 text-primary-600" : "border-transparent text-gray-500 hover:text-gray-700"
              }`}>{tab.charAt(0).toUpperCase() + tab.slice(1)}</button>
          ))}
        </nav>
      </div>

      {activeTab === "overview" && (
        <div className="space-y-3">
          <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Your Weakness Areas</h2>
          {mockWeaknesses.map((a) => (
            <div key={a.category} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 flex justify-between items-center">
              <div>
                <p className="font-medium text-gray-900 dark:text-white">{a.category.replace(/_/g, " ")}</p>
                <p className="text-sm text-gray-500">{a.finding_count} findings</p>
              </div>
              <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                a.improvement_trend === "improving" ? "bg-green-100 text-green-800" :
                a.improvement_trend === "declining" ? "bg-red-100 text-red-800" : "bg-gray-100 text-gray-800"
              }`}>{a.improvement_trend}</span>
            </div>
          ))}
        </div>
      )}

      {activeTab === "challenges" && (
        <div className="grid md:grid-cols-2 gap-4">
          {mockChallenges.map((ch) => (
            <div key={ch.id} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
              <div className="flex justify-between items-start mb-2">
                <h3 className="font-semibold text-gray-900 dark:text-white">{ch.title}</h3>
                <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                  ch.difficulty === "beginner" ? "text-green-600 bg-green-50" : "text-yellow-600 bg-yellow-50"
                }`}>{ch.difficulty}</span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">{ch.description}</p>
              <pre className="bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs font-mono mb-3 overflow-x-auto">{ch.buggy_code}</pre>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500"><Target className="h-4 w-4 inline mr-1" />{ch.points} pts</span>
                <button className="bg-primary-600 text-white px-3 py-1.5 rounded-lg text-sm hover:bg-primary-700 flex items-center gap-1">
                  <Code className="h-4 w-4" /> Attempt <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === "leaderboard" && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <table className="w-full">
            <thead><tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left p-4 text-sm font-medium text-gray-500">#</th>
              <th className="text-left p-4 text-sm font-medium text-gray-500">Developer</th>
              <th className="text-right p-4 text-sm font-medium text-gray-500">Points</th>
              <th className="text-right p-4 text-sm font-medium text-gray-500">Challenges</th>
              <th className="text-right p-4 text-sm font-medium text-gray-500">Streak</th>
              <th className="text-right p-4 text-sm font-medium text-gray-500">Badges</th>
            </tr></thead>
            <tbody>
              {mockLeaderboard.map((e, i) => (
                <tr key={e.name} className={`border-b border-gray-100 dark:border-gray-700/50 ${e.name === "You" ? "bg-primary-50 dark:bg-primary-900/10" : ""}`}>
                  <td className="p-4 text-sm">{i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : i + 1}</td>
                  <td className="p-4 text-sm font-medium text-gray-900 dark:text-white">{e.name}</td>
                  <td className="p-4 text-sm text-right font-mono">{e.points}</td>
                  <td className="p-4 text-sm text-right">{e.challenges}</td>
                  <td className="p-4 text-sm text-right">{e.streak > 0 ? `🔥 ${e.streak}` : "—"}</td>
                  <td className="p-4 text-sm text-right">{e.badges}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

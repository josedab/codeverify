const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export interface Organization {
  id: string;
  github_id: number;
  name: string;
  slug: string;
  avatar_url?: string;
  plan: string;
  created_at: string;
}

export interface Repository {
  id: string;
  github_id: number;
  name: string;
  full_name: string;
  default_branch: string;
  enabled: boolean;
  config?: Record<string, unknown>;
  organization_id: string;
  created_at: string;
}

export interface Analysis {
  id: string;
  repository_id: string;
  pr_number?: number;
  pr_title?: string;
  head_sha: string;
  base_sha?: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  conclusion?: "passed" | "failed" | "error";
  started_at?: string;
  completed_at?: string;
  summary?: AnalysisSummary;
  created_at: string;
}

export interface AnalysisSummary {
  total_issues: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
  pass: boolean;
}

export interface Finding {
  id: string;
  analysis_id: string;
  category: string;
  severity: "critical" | "high" | "medium" | "low";
  title: string;
  description: string;
  file_path: string;
  line_start: number;
  line_end: number;
  confidence: number;
  verification_type: "ai" | "formal" | "pattern";
  fix_suggestion?: string;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface User {
  id: string;
  github_id: number;
  username: string;
  email?: string;
  avatar_url?: string;
  organizations: Organization[];
}

// Audit Log Types
export interface AuditLog {
  id: string;
  org_id: string | null;
  user_id: string | null;
  username: string | null;
  action: string;
  resource_type: string | null;
  resource_id: string | null;
  details: Record<string, unknown>;
  ip_address: string | null;
  user_agent: string | null;
  created_at: string;
}

export interface AuditLogListResponse {
  items: AuditLog[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface AuditLogStats {
  total_events: number;
  events_today: number;
  events_this_week: number;
  by_action: Record<string, number>;
  by_resource_type: Record<string, number>;
  top_users: Array<{ username: string; event_count: number }>;
}

// Team Analytics Types
export interface TeamMemberStats {
  user_id: string;
  username: string;
  avatar_url: string | null;
  analyses_triggered: number;
  findings_created: number;
  findings_dismissed: number;
  last_active: string | null;
}

export interface TeamStats {
  organization_id: string;
  total_members: number;
  active_members_7d: number;
  active_members_30d: number;
  members: TeamMemberStats[];
  activity_by_day: Array<{ date: string; count: number }>;
}

export interface TrendDataPoint {
  date: string;
  analyses: number;
  passed: number;
  failed: number;
  findings: number;
}

export interface TrendsResponse {
  period: string;
  data: TrendDataPoint[];
  summary: {
    total_analyses: number;
    total_passed: number;
    total_failed: number;
    total_findings: number;
    pass_rate: number;
    avg_findings_per_analysis: number;
  };
}

export interface LeaderboardEntry {
  rank: number;
  user_id: string;
  username: string;
  avatar_url: string | null;
  score: number;
  metric: string;
}

export interface LeaderboardResponse {
  period: string;
  entries: LeaderboardEntry[];
}

// Cross-Repo Types
export interface CrossRepoGraph {
  repositories: Record<string, unknown>;
  dependencies: Array<{
    source: string;
    target: string;
    dependency_type: string;
    contracts: string[];
  }>;
  contracts: Record<string, unknown>;
  stats: {
    repository_count: number;
    dependency_count: number;
    contract_count: number;
  };
}

// Debugger Types
export interface VerificationStep {
  id: number;
  step_type: string;
  description: string;
  formula: string | null;
  result: string | null;
  status: string;
  duration_ms: number;
}

export interface VerificationTrace {
  id: string;
  name: string;
  description: string | null;
  steps: VerificationStep[];
  constraints: string[];
  variables: Record<string, string>;
  result: string | null;
  counterexample: Record<string, unknown> | null;
  total_duration_ms: number;
  explanation: Record<string, unknown> | null;
  visualization: Record<string, unknown> | null;
}

export interface DebuggerSessionState {
  variables: Record<string, string>;
  constraints: Array<{ formula: string; description: string }>;
  stack_depth: number;
  history: Array<{ action: string; timestamp: string }>;
}

class ApiClient {
  private token: string | null = null;

  setToken(token: string) {
    this.token = token;
  }

  clearToken() {
    this.token = null;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options.headers as Record<string, string>),
    };

    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }

    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `API error: ${response.status}`);
    }

    return response.json();
  }

  // Auth
  async getCurrentUser(): Promise<User> {
    return this.request<User>("/auth/me");
  }

  // Organizations
  async getOrganizations(): Promise<Organization[]> {
    return this.request<Organization[]>("/organizations");
  }

  async getOrganization(orgId: string): Promise<Organization> {
    return this.request<Organization>(`/organizations/${orgId}`);
  }

  // Repositories
  async getRepositories(orgId?: string): Promise<Repository[]> {
    const query = orgId ? `?organization_id=${orgId}` : "";
    return this.request<Repository[]>(`/repositories${query}`);
  }

  async getRepository(repoId: string): Promise<Repository> {
    return this.request<Repository>(`/repositories/${repoId}`);
  }

  async updateRepository(
    repoId: string,
    data: Partial<Repository>
  ): Promise<Repository> {
    return this.request<Repository>(`/repositories/${repoId}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  // Analyses
  async getAnalyses(params?: {
    repository_id?: string;
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<Analysis[]> {
    const query = new URLSearchParams();
    if (params?.repository_id) query.set("repository_id", params.repository_id);
    if (params?.status) query.set("status", params.status);
    if (params?.limit) query.set("limit", params.limit.toString());
    if (params?.offset) query.set("offset", params.offset.toString());
    const queryStr = query.toString() ? `?${query.toString()}` : "";
    return this.request<Analysis[]>(`/analyses${queryStr}`);
  }

  async getAnalysis(analysisId: string): Promise<Analysis> {
    return this.request<Analysis>(`/analyses/${analysisId}`);
  }

  async getAnalysisFindings(analysisId: string): Promise<Finding[]> {
    return this.request<Finding[]>(`/analyses/${analysisId}/findings`);
  }

  async cancelAnalysis(analysisId: string): Promise<void> {
    await this.request(`/analyses/${analysisId}/cancel`, {
      method: "POST",
    });
  }

  async retryAnalysis(analysisId: string): Promise<Analysis> {
    return this.request<Analysis>(`/analyses/${analysisId}/retry`, {
      method: "POST",
    });
  }

  // Stats
  async getDashboardStats(orgId?: string): Promise<{
    total_analyses: number;
    passed: number;
    failed: number;
    total_findings: number;
    by_severity: Record<string, number>;
    recent_activity: Analysis[];
  }> {
    const query = orgId ? `?organization_id=${orgId}` : "";
    return this.request(`/stats/dashboard${query}`);
  }

  // Audit Logs
  async getAuditLogs(params?: {
    organization_id?: string;
    user_id?: string;
    action?: string;
    resource_type?: string;
    start_date?: string;
    end_date?: string;
    search?: string;
    page?: number;
    page_size?: number;
  }): Promise<AuditLogListResponse> {
    const query = new URLSearchParams();
    if (params?.organization_id) query.set("organization_id", params.organization_id);
    if (params?.user_id) query.set("user_id", params.user_id);
    if (params?.action) query.set("action", params.action);
    if (params?.resource_type) query.set("resource_type", params.resource_type);
    if (params?.start_date) query.set("start_date", params.start_date);
    if (params?.end_date) query.set("end_date", params.end_date);
    if (params?.search) query.set("search", params.search);
    if (params?.page) query.set("page", params.page.toString());
    if (params?.page_size) query.set("page_size", params.page_size.toString());
    const queryStr = query.toString() ? `?${query.toString()}` : "";
    return this.request<AuditLogListResponse>(`/audit-logs${queryStr}`);
  }

  async getAuditLogStats(orgId?: string): Promise<AuditLogStats> {
    const query = orgId ? `?organization_id=${orgId}` : "";
    return this.request<AuditLogStats>(`/audit-logs/stats${query}`);
  }

  async getAuditLogActions(): Promise<string[]> {
    return this.request<string[]>("/audit-logs/actions");
  }

  async getAuditLogResourceTypes(): Promise<string[]> {
    return this.request<string[]>("/audit-logs/resource-types");
  }

  async exportAuditLogs(params?: {
    organization_id?: string;
    start_date?: string;
    end_date?: string;
    format?: "csv" | "json";
  }): Promise<Blob> {
    const query = new URLSearchParams();
    if (params?.organization_id) query.set("organization_id", params.organization_id);
    if (params?.start_date) query.set("start_date", params.start_date);
    if (params?.end_date) query.set("end_date", params.end_date);
    if (params?.format) query.set("format", params.format);
    const queryStr = query.toString() ? `?${query.toString()}` : "";
    
    const response = await fetch(`${API_BASE}/audit-logs/export${queryStr}`, {
      headers: this.token ? { Authorization: `Bearer ${this.token}` } : {},
    });
    return response.blob();
  }

  // Team Analytics
  async getTeamStats(organizationId: string): Promise<TeamStats> {
    return this.request<TeamStats>(`/stats/team?organization_id=${organizationId}`);
  }

  async getTrends(params?: {
    organization_id?: string;
    period?: "7d" | "30d" | "90d";
  }): Promise<TrendsResponse> {
    const query = new URLSearchParams();
    if (params?.organization_id) query.set("organization_id", params.organization_id);
    if (params?.period) query.set("period", params.period);
    const queryStr = query.toString() ? `?${query.toString()}` : "";
    return this.request<TrendsResponse>(`/stats/trends${queryStr}`);
  }

  async getLeaderboard(params: {
    organization_id: string;
    metric?: "analyses" | "findings_fixed" | "activity";
    period?: "7d" | "30d" | "90d" | "all";
  }): Promise<LeaderboardResponse> {
    const query = new URLSearchParams();
    query.set("organization_id", params.organization_id);
    if (params.metric) query.set("metric", params.metric);
    if (params.period) query.set("period", params.period);
    return this.request<LeaderboardResponse>(`/stats/leaderboard?${query.toString()}`);
  }

  // Cross-Repo
  async getCrossRepoGraph(): Promise<CrossRepoGraph> {
    return this.request<CrossRepoGraph>("/cross-repo/graph");
  }

  async getCrossRepoDependents(owner: string, name: string, recursive?: boolean): Promise<{
    repository: string;
    dependents: string[];
    count: number;
  }> {
    const query = recursive ? "?recursive=true" : "";
    return this.request(`/cross-repo/dependencies/${owner}/${name}/dependents${query}`);
  }

  // Debugger
  async createDebuggerSession(timeoutMs?: number): Promise<{ session_id: string }> {
    const query = timeoutMs ? `?timeout_ms=${timeoutMs}` : "";
    return this.request(`/debugger/session/create${query}`, { method: "POST" });
  }

  async executeDebuggerAction(
    sessionId: string,
    action: string,
    data?: Record<string, unknown>
  ): Promise<{
    success: boolean;
    result: Record<string, unknown> | null;
    state: DebuggerSessionState;
    error: string | null;
  }> {
    return this.request(`/debugger/session/${sessionId}/execute`, {
      method: "POST",
      body: JSON.stringify({ action, data: data || {} }),
    });
  }

  async getDebuggerSessionState(sessionId: string): Promise<{
    session_id: string;
    state: DebuggerSessionState;
    history: Array<{ action: string; timestamp: string }>;
  }> {
    return this.request(`/debugger/session/${sessionId}`);
  }

  async deleteDebuggerSession(sessionId: string): Promise<{ deleted: boolean }> {
    return this.request(`/debugger/session/${sessionId}`, { method: "DELETE" });
  }

  async verifyWithTrace(params: {
    formula: string;
    name?: string;
    description?: string;
    timeout_ms?: number;
  }): Promise<VerificationTrace> {
    return this.request("/debugger/trace", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }
}

export const api = new ApiClient();
export default api;

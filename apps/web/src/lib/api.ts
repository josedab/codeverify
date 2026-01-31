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
}

export const api = new ApiClient();
export default api;

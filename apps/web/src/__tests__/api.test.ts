import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiClient } from "../lib/api";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
const fetchMock = vi.fn<typeof fetch>();

function jsonResponse(body: unknown, init: ResponseInit = {}) {
  return new Response(JSON.stringify(body), {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("ApiClient", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("adds and clears the authorization token", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ id: "user-1" }))
      .mockResolvedValueOnce(jsonResponse([]));

    const client = new ApiClient();
    client.setToken("test-token");
    await client.getCurrentUser();

    expect(fetchMock).toHaveBeenNthCalledWith(1, `${API_BASE}/auth/me`, {
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer test-token",
      },
    });

    client.clearToken();
    await client.getOrganizations();

    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      `${API_BASE}/organizations`,
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  });

  it("serializes query parameters", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse([]));

    const client = new ApiClient();
    await client.getAnalyses({
      repository_id: "org/repo & fork",
      status: "completed",
      limit: 25,
      offset: 50,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE}/analyses?repository_id=org%2Frepo+%26+fork&status=completed&limit=25&offset=50`,
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  });

  it("returns successful JSON responses", async () => {
    const repository = {
      id: "repo-1",
      github_id: 123,
      name: "codeverify",
      full_name: "acme/codeverify",
      default_branch: "main",
      enabled: true,
      organization_id: "org-1",
      created_at: "2026-07-30T12:00:00Z",
    };
    fetchMock.mockResolvedValueOnce(jsonResponse(repository));

    const client = new ApiClient();

    await expect(client.getRepository("repo-1")).resolves.toEqual(repository);
  });

  it("uses the API error detail when a request fails", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        { detail: "Repository not found" },
        { status: 404, statusText: "Not Found" }
      )
    );

    const client = new ApiClient();

    await expect(client.getRepository("missing")).rejects.toThrow(
      "Repository not found"
    );
  });

  it("exports audit logs as a Blob with serialized filters and auth", async () => {
    const csv = "action,username\nanalysis.created,octocat\n";
    fetchMock.mockResolvedValueOnce(
      new Response(csv, {
        headers: { "Content-Type": "text/csv" },
      })
    );

    const client = new ApiClient();
    client.setToken("export-token");

    const result = await client.exportAuditLogs({
      organization_id: "org/alpha",
      start_date: "2026-07-01",
      end_date: "2026-07-30",
      format: "csv",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE}/audit-logs/export?organization_id=org%2Falpha&start_date=2026-07-01&end_date=2026-07-30&format=csv`,
      {
        headers: {
          Authorization: "Bearer export-token",
        },
      }
    );
    expect(result).toBeInstanceOf(Blob);
    expect(result.type).toBe("text/csv");
    await expect(result.text()).resolves.toBe(csv);
  });
});

import { Readable } from "node:stream";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  createGithubApp,
  type AppLogger,
  type RedisClient,
} from "../src/app.js";

const webhookSecret = "test-webhook-secret";

const pullRequestPayload = {
  action: "opened",
  installation: { id: 987 },
  repository: {
    id: 123,
    full_name: "octo/codeverify",
  },
  pull_request: {
    number: 42,
    title: "Verify webhook handling",
    head: { sha: "0123456789abcdef0123456789abcdef01234567" },
    base: { sha: "fedcba9876543210fedcba9876543210fedcba98" },
  },
};

function createRedisMock(): RedisClient {
  return {
    set: vi.fn(async () => "OK"),
    lpush: vi.fn(async () => 1),
    hset: vi.fn(async () => 1),
  };
}

function createLoggerMock(): AppLogger {
  return {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  };
}

function createTestApp() {
  const redis = createRedisMock();
  const logger = createLoggerMock();
  const githubApp = createGithubApp({
    apiUrl: "http://api.invalid",
    logger,
    redis,
    webhookSecret,
    fetch: vi.fn(),
  });

  return { ...githubApp, logger, redis };
}

async function invokeWebhook(
  middleware: ReturnType<typeof createGithubApp>["webhookMiddleware"],
  payload: string,
  headers: Record<string, string>
) {
  const request = Readable.from([Buffer.from(payload)]) as Readable & {
    headers: Record<string, string>;
    method: string;
    url: string;
  };
  request.headers = headers;
  request.method = "POST";
  request.url = "/";

  const result: {
    body?: string;
    headers?: Record<string, string>;
    status?: number;
  } = {};
  const response = {
    writeHead(status: number, responseHeaders: Record<string, string>) {
      result.status = status;
      result.headers = { ...responseHeaders };
      return response;
    },
    end(body?: string) {
      result.body = body;
      return response;
    },
  };

  await middleware(request, response);

  return result;
}

function webhookHeaders(signature?: string) {
  return {
    "content-type": "application/json",
    "x-github-delivery": "delivery-123",
    "x-github-event": "pull_request",
    ...(signature ? { "x-hub-signature-256": signature } : {}),
  };
}

describe("GitHub webhook handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("rejects requests without the signature header", async () => {
    const { redis, webhookMiddleware } = createTestApp();
    const payload = JSON.stringify(pullRequestPayload);

    const response = await invokeWebhook(
      webhookMiddleware,
      payload,
      webhookHeaders()
    );

    expect(response.status).toBe(400);
    expect(response.body).toContain(
      "Required headers missing: x-hub-signature-256"
    );
    expect(redis.set).not.toHaveBeenCalled();
    expect(redis.lpush).not.toHaveBeenCalled();
  });

  it("rejects payloads with an invalid signature", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const { redis, webhookMiddleware } = createTestApp();
    const payload = JSON.stringify(pullRequestPayload);

    const response = await invokeWebhook(
      webhookMiddleware,
      payload,
      webhookHeaders(`sha256=${"0".repeat(64)}`)
    );

    expect(response.status).toBe(400);
    expect(response.body).toContain(
      "signature does not match event payload and secret"
    );
    expect(redis.set).not.toHaveBeenCalled();
    expect(redis.lpush).not.toHaveBeenCalled();
  });

  it("acknowledges but ignores unsupported pull request actions", async () => {
    const { redis, webhookMiddleware, webhooks } = createTestApp();
    const payload = JSON.stringify({
      ...pullRequestPayload,
      action: "closed",
    });
    const signature = await webhooks.sign(payload);

    const response = await invokeWebhook(
      webhookMiddleware,
      payload,
      webhookHeaders(signature)
    );

    expect(response.status).toBe(200);
    expect(response.body).toBe("ok\n");
    expect(redis.set).not.toHaveBeenCalled();
    expect(redis.lpush).not.toHaveBeenCalled();
  });

  it("queues the expected analysis job for an opened pull request", async () => {
    const { redis, webhookMiddleware, webhooks } = createTestApp();
    const payload = JSON.stringify(pullRequestPayload);
    const signature = await webhooks.sign(payload);

    const response = await invokeWebhook(
      webhookMiddleware,
      payload,
      webhookHeaders(signature)
    );

    const jobId = "octo/codeverify#42@01234567";
    expect(response.status).toBe(200);
    expect(response.body).toBe("ok\n");
    expect(redis.set).toHaveBeenCalledOnce();
    expect(redis.lpush).toHaveBeenCalledWith(
      "codeverify:analysis:queue",
      jobId
    );

    const [jobKey, serializedJob, expirationMode, ttl] = vi.mocked(redis.set)
      .mock.calls[0];
    expect(jobKey).toBe(`job:${jobId}`);
    expect(expirationMode).toBe("EX");
    expect(ttl).toBe(86400);

    const { queued_at: queuedAt, ...jobData } = JSON.parse(serializedJob);
    expect(jobData).toEqual({
      job_id: jobId,
      repo_full_name: "octo/codeverify",
      repo_id: 123,
      pr_number: 42,
      pr_title: "Verify webhook handling",
      head_sha: "0123456789abcdef0123456789abcdef01234567",
      base_sha: "fedcba9876543210fedcba9876543210fedcba98",
      installation_id: 987,
    });
    expect(new Date(queuedAt).toISOString()).toBe(queuedAt);
    expect(
      vi.mocked(redis.set).mock.invocationCallOrder[0]
    ).toBeLessThan(vi.mocked(redis.lpush).mock.invocationCallOrder[0]);
  });
});

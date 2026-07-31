import express from "express";
import { createNodeMiddleware, Webhooks } from "@octokit/webhooks";

export interface RedisClient {
  set(
    key: string,
    value: string,
    expirationMode: "EX",
    time: number
  ): Promise<unknown>;
  lpush(key: string, value: string): Promise<unknown>;
  hset(key: string, values: Record<string, string>): Promise<unknown>;
}

export interface AppLogger {
  info(message: string): void;
  info(bindings: Record<string, unknown>, message: string): void;
  warn(bindings: Record<string, unknown>, message: string): void;
  error(bindings: Record<string, unknown>, message: string): void;
}

export interface GithubAppOptions {
  apiUrl: string;
  logger: AppLogger;
  redis: RedisClient;
  webhookSecret: string;
  fetch?: typeof fetch;
  webhooks?: Webhooks;
}

export function createGithubApp({
  apiUrl,
  logger,
  redis,
  webhookSecret,
  fetch: fetchImpl = fetch,
  webhooks = new Webhooks({ secret: webhookSecret }),
}: GithubAppOptions) {
  const app = express();

  app.get("/health", (req, res) => {
    res.json({ status: "healthy", service: "codeverify-github-app" });
  });

  webhooks.on("pull_request.opened", async ({ payload }) => {
    logger.info(
      {
        action: "pr_opened",
        repo: payload.repository.full_name,
        pr: payload.pull_request.number,
        sha: payload.pull_request.head.sha,
      },
      "Processing PR opened event"
    );

    await queueAnalysis(payload);
  });

  webhooks.on("pull_request.synchronize", async ({ payload }) => {
    logger.info(
      {
        action: "pr_synchronize",
        repo: payload.repository.full_name,
        pr: payload.pull_request.number,
        sha: payload.pull_request.head.sha,
      },
      "Processing PR synchronize event"
    );

    await queueAnalysis(payload);
  });

  webhooks.on("pull_request.reopened", async ({ payload }) => {
    logger.info(
      {
        action: "pr_reopened",
        repo: payload.repository.full_name,
        pr: payload.pull_request.number,
      },
      "Processing PR reopened event"
    );

    await queueAnalysis(payload);
  });

  webhooks.on("installation.created", async ({ payload }) => {
    const account = payload.installation.account;
    const accountName = account
      ? "login" in account
        ? account.login
        : account.name
      : "unknown";

    logger.info(
      {
        action: "installation_created",
        installationId: payload.installation.id,
        account: accountName,
      },
      "New installation created"
    );

    await storeInstallation(payload.installation.id, accountName, "created");
  });

  webhooks.on("installation.deleted", async ({ payload }) => {
    const account = payload.installation.account;
    const accountName = account
      ? "login" in account
        ? account.login
        : account.name
      : "unknown";

    logger.info(
      {
        action: "installation_deleted",
        installationId: payload.installation.id,
        account: accountName,
      },
      "Installation deleted"
    );

    await storeInstallation(payload.installation.id, accountName, "deleted");
  });

  async function storeInstallation(
    installationId: number,
    accountName: string,
    status: "created" | "deleted"
  ): Promise<void> {
    const key = `installation:${installationId}`;

    if (status === "created") {
      await redis.hset(key, {
        account: accountName,
        status: "active",
        installed_at: new Date().toISOString(),
      });
      logger.info({ installationId, accountName }, "Installation stored");
    } else {
      await redis.hset(key, {
        status: "deleted",
        deleted_at: new Date().toISOString(),
      });
      logger.info({ installationId }, "Installation marked as deleted");
    }

    try {
      await fetchImpl(`${apiUrl}/api/v1/webhooks/installation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ installationId, accountName, status }),
      });
    } catch (error) {
      logger.warn(
        { error, installationId },
        "Failed to notify API of installation change"
      );
    }
  }

  async function queueAnalysis(payload: any): Promise<string> {
    const jobId = `${payload.repository.full_name}#${payload.pull_request.number}@${payload.pull_request.head.sha.substring(0, 8)}`;

    const jobData = {
      job_id: jobId,
      repo_full_name: payload.repository.full_name,
      repo_id: payload.repository.id,
      pr_number: payload.pull_request.number,
      pr_title: payload.pull_request.title,
      head_sha: payload.pull_request.head.sha,
      base_sha: payload.pull_request.base.sha,
      installation_id: payload.installation?.id,
      queued_at: new Date().toISOString(),
    };

    await redis.set(`job:${jobId}`, JSON.stringify(jobData), "EX", 86400);
    await redis.lpush("codeverify:analysis:queue", jobId);

    logger.info(
      { jobId, repo: payload.repository.full_name },
      "Analysis job queued"
    );

    return jobId;
  }

  const webhookMiddleware = createNodeMiddleware(webhooks, { path: "/" });
  app.use("/webhooks/github", webhookMiddleware);

  webhooks.onError((error) => {
    logger.error({ error: error.message }, "Webhook error");
  });

  return { app, webhookMiddleware, webhooks };
}

/**
 * CodeVerify GitHub App - Main Entry Point
 *
 * This service handles GitHub webhooks and coordinates with the
 * analysis workers to process pull requests.
 */

import express from "express";
import { createNodeMiddleware, Webhooks } from "@octokit/webhooks";
import { App } from "@octokit/app";
import Redis from "ioredis";
import pino from "pino";

// Configuration
const config = {
  port: parseInt(process.env.PORT || "3001", 10),
  githubAppId: process.env.GITHUB_APP_ID || "",
  githubPrivateKey: process.env.GITHUB_APP_PRIVATE_KEY || "",
  githubWebhookSecret: process.env.GITHUB_WEBHOOK_SECRET || "development",
  redisUrl: process.env.REDIS_URL || "redis://localhost:6379/0",
  apiUrl: process.env.API_URL || "http://localhost:8000",
  environment: process.env.NODE_ENV || "development",
};

// Logger
const logger = pino({
  level: process.env.LOG_LEVEL || "info",
  transport:
    config.environment === "development"
      ? { target: "pino-pretty", options: { colorize: true } }
      : undefined,
});

// Redis client for job queue
const redis = new Redis(config.redisUrl);

// GitHub App setup
const webhooks = new Webhooks({
  secret: config.githubWebhookSecret,
});

// Express app
const app = express();

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "healthy", service: "codeverify-github-app" });
});

// Handle pull request events
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

// Handle installation events
webhooks.on("installation.created", async ({ payload }) => {
  const account = payload.installation.account;
  const accountName = account ? ('login' in account ? account.login : account.name) : 'unknown';

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
  const accountName = account ? ('login' in account ? account.login : account.name) : 'unknown';

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

// Store installation status in Redis and notify API
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

  // Notify API service of installation change
  try {
    await fetch(`${config.apiUrl}/api/v1/webhooks/installation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ installationId, accountName, status }),
    });
  } catch (error) {
    logger.warn({ error, installationId }, "Failed to notify API of installation change");
  }
}

// Queue analysis job
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

  // Store job data and push to queue
  await redis.set(`job:${jobId}`, JSON.stringify(jobData), "EX", 86400);
  await redis.lpush("codeverify:analysis:queue", jobId);

  logger.info({ jobId, repo: payload.repository.full_name }, "Analysis job queued");

  return jobId;
}

// Mount webhook middleware
app.use(
  "/webhooks/github",
  createNodeMiddleware(webhooks, { path: "/" })
);

// Error handling
webhooks.onError((error) => {
  logger.error({ error: error.message }, "Webhook error");
});

// Start server
app.listen(config.port, () => {
  logger.info(
    { port: config.port, environment: config.environment },
    "CodeVerify GitHub App started"
  );
});

// Graceful shutdown
process.on("SIGTERM", async () => {
  logger.info("Shutting down...");
  await redis.quit();
  process.exit(0);
});

/**
 * CodeVerify GitHub App - Main Entry Point
 *
 * This service handles GitHub webhooks and coordinates with the
 * analysis workers to process pull requests.
 */

import Redis from "ioredis";
import pino from "pino";
import { createGithubApp } from "./app.js";

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

const { app } = createGithubApp({
  apiUrl: config.apiUrl,
  logger,
  redis,
  webhookSecret: config.githubWebhookSecret,
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

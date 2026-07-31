/**
 * Pure types and logic backing the Copilot suggestion interceptor.
 *
 * `copilotInterceptorProvider.ts` creates VS Code decoration types at module
 * load time, so it can only be imported inside the VS Code extension host.
 * The types and verification logic below have no `vscode` dependency, so
 * they live in this separate module and can be unit tested directly with
 * Node's built-in test runner. `copilotInterceptorProvider.ts` imports and
 * re-exports everything here for backward compatibility.
 */

/**
 * Verification status for a Copilot suggestion
 */
export enum VerificationStatus {
  Pending = "pending",
  Verified = "verified",
  Warning = "warning",
  Error = "error",
  Timeout = "timeout",
}

/**
 * Issue found during verification
 */
export interface VerificationIssue {
  line: number;
  column: number;
  message: string;
  severity: "error" | "warning" | "info";
  category: string;
  fix?: string;
}

/**
 * Result of verifying a suggestion
 */
export interface SuggestionVerification {
  suggestionId: string;
  status: VerificationStatus;
  issues: VerificationIssue[];
  score: number; // 0-100 trust score
  verificationTimeMs: number;
  metadata: Record<string, unknown>;
}

/**
 * Configuration for the interceptor
 */
export interface InterceptorConfig {
  enabled: boolean;
  autoVerify: boolean;
  showInlineStatus: boolean;
  blockOnError: boolean;
  verificationTimeout: number;
  minTrustScore: number;
  checks: string[];
}

/**
 * Context for a Copilot suggestion. `cursorPosition` is typed as
 * `vscode.Position` for parity with the real interceptor, but is only ever
 * imported as a type here (erased at compile time), so constructing this
 * object in a test does not require the `vscode` module to exist at runtime.
 */
export interface SuggestionContext {
  filePath: string;
  language: string;
  surroundingCode: string;
  cursorPosition: import("vscode").Position;
  documentVersion: number;
}

/**
 * CodeVerify client interface used by the interceptor to verify suggestions.
 */
export interface SuggestionVerificationClient {
  verifySuggestion(
    code: string,
    context: SuggestionContext
  ): Promise<SuggestionVerification>;
  getConfig(): Promise<InterceptorConfig>;
}

/**
 * Default interceptor configuration, used until the real config is loaded
 * from the server (see `CopilotInterceptorProvider.initialize`).
 */
export function getDefaultInterceptorConfig(): InterceptorConfig {
  return {
    enabled: true,
    autoVerify: true,
    showInlineStatus: true,
    blockOnError: false,
    verificationTimeout: 5000,
    minTrustScore: 60,
    checks: ["null_safety", "overflow", "bounds", "security"],
  };
}

/**
 * Mock client for testing without server connection
 */
export class MockCodeVerifyClient implements SuggestionVerificationClient {
  async verifySuggestion(
    code: string,
    context: SuggestionContext
  ): Promise<SuggestionVerification> {
    // Simulate verification delay
    await new Promise((resolve) => setTimeout(resolve, 500));

    const issues: VerificationIssue[] = [];

    // Check for common issues
    if (code.includes("eval(")) {
      issues.push({
        line: 0,
        column: code.indexOf("eval("),
        message: "Potentially unsafe eval() usage detected",
        severity: "error",
        category: "security",
        fix: "// Consider using JSON.parse() or a safer alternative",
      });
    }

    if (code.includes("password") && code.includes("=")) {
      issues.push({
        line: 0,
        column: 0,
        message: "Potential hardcoded password detected",
        severity: "error",
        category: "security",
      });
    }

    if (/\[\s*\w+\s*\]/.test(code) && !code.includes("length")) {
      issues.push({
        line: 0,
        column: 0,
        message: "Array access without bounds checking",
        severity: "warning",
        category: "bounds",
      });
    }

    const status =
      issues.filter((i) => i.severity === "error").length > 0
        ? VerificationStatus.Error
        : issues.length > 0
          ? VerificationStatus.Warning
          : VerificationStatus.Verified;

    const score = Math.max(0, 100 - issues.length * 20);

    return {
      suggestionId: `mock-${Date.now()}`,
      status,
      issues,
      score,
      verificationTimeMs: 500,
      metadata: {},
    };
  }

  async getConfig(): Promise<InterceptorConfig> {
    return getDefaultInterceptorConfig();
  }
}

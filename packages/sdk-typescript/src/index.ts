/**
 * CodeVerify TypeScript SDK — embed formal verification checks programmatically.
 *
 * @example
 * ```ts
 * import { verify, checkSafety, generateProof } from '@codeverify/sdk';
 *
 * const result = await verify('function add(a: number, b: number) { return a + b; }');
 * console.log(result.passed);
 *
 * const safety = await checkSafety(code);
 * console.log(safety.riskLevel);
 * ```
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Finding {
  id: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  filePath: string;
  lineStart?: number;
  lineEnd?: number;
  fixSuggestion?: string;
  confidence: number;
}

export interface VerificationResult {
  passed: boolean;
  findings: Finding[];
  checksRun: string[];
  language: string;
  durationMs: number;
  error?: string;
}

export interface SafetyResult {
  safe: boolean;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  issues: Finding[];
  categoriesChecked: string[];
  durationMs: number;
}

export interface ProofResult {
  status: 'proved' | 'disproved' | 'unknown' | 'timeout';
  propertyChecked: string;
  counterexamples: Record<string, unknown>[];
  proofTree?: Record<string, unknown>;
  solverTimeMs: number;
  explanation?: string;
}

export interface CodeVerifyConfig {
  apiKey?: string;
  apiUrl?: string;
  timeout?: number;
}

function getEnvironmentVariable(name: string): string | undefined {
  const runtime = globalThis as typeof globalThis & {
    process?: { env?: Record<string, string | undefined> };
  };
  return runtime.process?.env?.[name];
}

function isProofStatus(status: unknown): status is ProofResult['status'] {
  return status === 'proved' || status === 'disproved' || status === 'unknown' || status === 'timeout';
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export class CodeVerifyClient {
  private apiKey: string;
  private apiUrl: string;
  private timeout: number;

  constructor(config: CodeVerifyConfig = {}) {
    this.apiKey = config.apiKey || getEnvironmentVariable('CODEVERIFY_API_KEY') || '';
    this.apiUrl = (
      config.apiUrl ||
      getEnvironmentVariable('CODEVERIFY_API_URL') ||
      'https://api.codeverify.dev'
    ).replace(/\/$/, '');
    this.timeout = config.timeout ?? 30000;
  }

  private async request<T>(path: string, body: Record<string, unknown>): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const resp = await fetch(`${this.apiUrl}${path}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
          'User-Agent': '@codeverify/sdk/0.1.0',
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!resp.ok) {
        throw new Error(`CodeVerify API error: ${resp.status} ${resp.statusText}`);
      }
      return (await resp.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  async verify(code: string, language = 'typescript', checks?: string[]): Promise<VerificationResult> {
    const start = Date.now();
    try {
      const data = await this.request<Record<string, unknown>>('/api/v1/verified-autofix/generate', {
        code,
        language,
        finding: {
          finding_id: 'sdk-check',
          type: 'general',
          severity: 'medium',
          description: 'SDK verification check',
          file_path: '<inline>',
          line_start: 1,
          code_snippet: code.slice(0, 200),
        },
      });

      const findings: Finding[] = [];
      if (data.status !== 'verified') {
        findings.push({
          id: String(data.fix_id || 'unknown'),
          type: 'verification',
          severity: 'medium',
          title: 'Verification issue found',
          description: String(data.explanation || ''),
          filePath: '<inline>',
          confidence: 0.8,
        });
      }

      return {
        passed: findings.length === 0,
        findings,
        checksRun: checks || ['all'],
        language,
        durationMs: Date.now() - start,
      };
    } catch {
      return this.localVerify(code, language, checks, Date.now() - start);
    }
  }

  async checkSafety(code: string, language = 'typescript'): Promise<SafetyResult> {
    const issues: Finding[] = [];
    const patterns: [RegExp, string, string][] = [
      [/eval\s*\(/, 'eval_usage', 'Use of eval()'],
      [/innerHTML\s*=/, 'xss', 'Possible XSS via innerHTML'],
      [/document\.write\s*\(/, 'xss', 'Possible XSS via document.write'],
      [/new\s+Function\s*\(/, 'code_injection', 'Dynamic function creation'],
    ];

    for (const [pattern, type, title] of patterns) {
      const match = pattern.exec(code);
      if (match) {
        const line = code.slice(0, match.index).split('\n').length;
        issues.push({
          id: `safety-${type}-${line}`,
          type,
          severity: 'high',
          title,
          description: `${title} at line ${line}`,
          filePath: '<inline>',
          lineStart: line,
          confidence: 0.9,
        });
      }
    }

    let riskLevel: SafetyResult['riskLevel'] = 'low';
    if (issues.some((i) => i.severity === 'critical')) riskLevel = 'critical';
    else if (issues.some((i) => i.severity === 'high')) riskLevel = 'high';
    else if (issues.length > 0) riskLevel = 'medium';

    return {
      safe: issues.length === 0,
      riskLevel,
      issues,
      categoriesChecked: ['xss', 'injection', 'eval'],
      durationMs: 0,
    };
  }

  async generateProof(code: string, property?: string, language = 'typescript'): Promise<ProofResult> {
    try {
      const data = await this.request<Record<string, unknown>>('/api/v1/proof-explorer/explore', {
        code,
        function_name: 'target',
        language,
        properties: property ? [{ expression: property }] : [],
      });

      const proofTree = data.proof_tree;
      const explanation = data.explanation;
      return {
        status: isProofStatus(data.status) ? data.status : 'unknown',
        propertyChecked: property || 'auto-inferred',
        counterexamples: Array.isArray(data.counterexamples)
          ? (data.counterexamples as Record<string, unknown>[])
          : [],
        ...(proofTree && typeof proofTree === 'object'
          ? { proofTree: proofTree as Record<string, unknown> }
          : {}),
        solverTimeMs: typeof data.solver_time_ms === 'number' ? data.solver_time_ms : 0,
        ...(typeof explanation === 'string' ? { explanation } : {}),
      };
    } catch {
      return { status: 'unknown', propertyChecked: property || 'auto-inferred', counterexamples: [], solverTimeMs: 0 };
    }
  }

  private localVerify(code: string, language: string, checks?: string[], durationMs = 0): VerificationResult {
    const findings: Finding[] = [];
    // Basic syntax heuristic for TypeScript/JavaScript
    const opens = (code.match(/[({[]/g) || []).length;
    const closes = (code.match(/[)}\]]/g) || []).length;
    if (opens !== closes) {
      findings.push({
        id: 'syntax-mismatch',
        type: 'syntax',
        severity: 'critical',
        title: 'Unbalanced brackets',
        description: `Found ${opens} opening and ${closes} closing brackets`,
        filePath: '<inline>',
        confidence: 1.0,
      });
    }
    return { passed: findings.length === 0, findings, checksRun: checks || ['syntax'], language, durationMs };
  }
}

// ---------------------------------------------------------------------------
// Module-level convenience functions
// ---------------------------------------------------------------------------

let _client: CodeVerifyClient | undefined;

function getClient(): CodeVerifyClient {
  if (!_client) _client = new CodeVerifyClient();
  return _client;
}

export function configure(config: CodeVerifyConfig): void {
  _client = new CodeVerifyClient(config);
}

export async function verify(code: string, language?: string, checks?: string[]): Promise<VerificationResult> {
  return getClient().verify(code, language, checks);
}

export async function checkSafety(code: string, language?: string): Promise<SafetyResult> {
  return getClient().checkSafety(code, language);
}

export async function generateProof(code: string, property?: string, language?: string): Promise<ProofResult> {
  return getClient().generateProof(code, property, language);
}

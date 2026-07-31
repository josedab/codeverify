import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CodeVerifyClient,
  configure,
  verify,
} from '../src/index';

const fetchMock = vi.fn<typeof fetch>();

function jsonResponse(
  body: unknown,
  status = 200,
  statusText = 'OK',
): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response;
}

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal('fetch', fetchMock);
  vi.stubEnv('CODEVERIFY_API_KEY', '');
  vi.stubEnv('CODEVERIFY_API_URL', '');
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('CodeVerifyClient requests', () => {
  it('serializes verification requests and sends configured authentication headers', async () => {
    const code = `export const value = "${'x'.repeat(220)}";`;
    fetchMock.mockResolvedValue(jsonResponse({ status: 'verified' }));
    const client = new CodeVerifyClient({
      apiKey: 'test-api-key',
      apiUrl: 'https://codeverify.example/',
    });

    const result = await client.verify(code, 'typescript', ['syntax', 'safety']);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('https://codeverify.example/api/v1/verified-autofix/generate');
    expect(init).toMatchObject({
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'Bearer test-api-key',
        'User-Agent': '@codeverify/sdk/0.1.0',
      },
    });
    expect(init?.signal).toBeInstanceOf(AbortSignal);
    expect(JSON.parse(String(init?.body))).toEqual({
      code,
      language: 'typescript',
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
    expect(result).toMatchObject({
      passed: true,
      findings: [],
      checksRun: ['syntax', 'safety'],
      language: 'typescript',
    });
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it('serializes proof requests without an authorization header and maps successful responses', async () => {
    const proofTree = { id: 'root', status: 'disproved' };
    const counterexamples = [{ id: 'ce-1', variables: [{ name: 'x', value: -1 }] }];
    fetchMock.mockResolvedValue(jsonResponse({
      status: 'disproved',
      counterexamples,
      proof_tree: proofTree,
      solver_time_ms: 12.5,
      explanation: 'A negative input violates the property.',
    }));
    const client = new CodeVerifyClient({ apiUrl: 'https://codeverify.example/' });

    const result = await client.generateProof(
      'function target(x: number) { return x; }',
      'result >= 0',
      'typescript',
    );

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('https://codeverify.example/api/v1/proof-explorer/explore');
    expect(init?.headers).toEqual({
      'Content-Type': 'application/json',
      'User-Agent': '@codeverify/sdk/0.1.0',
    });
    expect(JSON.parse(String(init?.body))).toEqual({
      code: 'function target(x: number) { return x; }',
      function_name: 'target',
      language: 'typescript',
      properties: [{ expression: 'result >= 0' }],
    });
    expect(result).toEqual({
      status: 'disproved',
      propertyChecked: 'result >= 0',
      counterexamples,
      proofTree,
      solverTimeMs: 12.5,
      explanation: 'A negative input violates the property.',
    });
  });

  it('uses environment configuration when explicit options are absent', async () => {
    vi.stubEnv('CODEVERIFY_API_KEY', 'environment-key');
    vi.stubEnv('CODEVERIFY_API_URL', 'https://environment.example/');
    fetchMock.mockResolvedValue(jsonResponse({ status: 'verified' }));

    await new CodeVerifyClient().verify('const valid = true;');

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('https://environment.example/api/v1/verified-autofix/generate');
    expect(init?.headers).toMatchObject({
      Authorization: 'Bearer environment-key',
    });
  });
});

describe('response and error mapping', () => {
  it('maps an unverified API response to a verification finding', async () => {
    fetchMock.mockResolvedValue(jsonResponse({
      status: 'failed',
      fix_id: 42,
      explanation: 'The proposed change could not be verified.',
    }));

    const result = await new CodeVerifyClient({ apiUrl: 'https://codeverify.example' })
      .verify('const value = risky();');

    expect(result).toMatchObject({
      passed: false,
      checksRun: ['all'],
      language: 'typescript',
      findings: [{
        id: '42',
        type: 'verification',
        severity: 'medium',
        title: 'Verification issue found',
        description: 'The proposed change could not be verified.',
        filePath: '<inline>',
        confidence: 0.8,
      }],
    });
  });

  it('falls back to local verification after an HTTP error', async () => {
    fetchMock.mockResolvedValue(jsonResponse({}, 503, 'Service Unavailable'));

    const result = await new CodeVerifyClient({ apiUrl: 'https://codeverify.example' })
      .verify('function broken() {', 'typescript');

    expect(result).toMatchObject({
      passed: false,
      checksRun: ['syntax'],
      language: 'typescript',
      findings: [{
        id: 'syntax-mismatch',
        type: 'syntax',
        severity: 'critical',
        title: 'Unbalanced brackets',
        description: 'Found 2 opening and 1 closing brackets',
        filePath: '<inline>',
        confidence: 1,
      }],
    });
  });

  it('normalizes malformed proof responses and transport errors', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        status: 'unexpected',
        counterexamples: null,
        proof_tree: null,
        solver_time_ms: 'slow',
      }))
      .mockRejectedValueOnce(new TypeError('offline'));
    const client = new CodeVerifyClient({ apiUrl: 'https://codeverify.example' });

    await expect(client.generateProof('const value = 1;')).resolves.toEqual({
      status: 'unknown',
      propertyChecked: 'auto-inferred',
      counterexamples: [],
      solverTimeMs: 0,
    });
    await expect(client.generateProof('const value = 1;', 'value > 0')).resolves.toEqual({
      status: 'unknown',
      propertyChecked: 'value > 0',
      counterexamples: [],
      solverTimeMs: 0,
    });
  });

  it('honors a zero timeout and maps an aborted proof request to unknown', async () => {
    vi.useFakeTimers();
    let signal: AbortSignal | undefined;
    fetchMock.mockImplementation((_url, init) => {
      signal = init?.signal as AbortSignal;
      return new Promise<Response>((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
      });
    });
    const client = new CodeVerifyClient({
      apiUrl: 'https://codeverify.example',
      timeout: 0,
    });

    const resultPromise = client.generateProof('const value = 1;');
    await vi.advanceTimersByTimeAsync(0);

    expect(signal?.aborted).toBe(true);
    await expect(resultPromise).resolves.toEqual({
      status: 'unknown',
      propertyChecked: 'auto-inferred',
      counterexamples: [],
      solverTimeMs: 0,
    });
  });
});

describe('local safety checks and convenience API', () => {
  it('returns structured safety findings with line numbers and risk level', async () => {
    const result = await new CodeVerifyClient().checkSafety([
      'const input = getInput();',
      'eval(input);',
      'element.innerHTML = input;',
    ].join('\n'));

    expect(result).toEqual({
      safe: false,
      riskLevel: 'high',
      issues: [
        {
          id: 'safety-eval_usage-2',
          type: 'eval_usage',
          severity: 'high',
          title: 'Use of eval()',
          description: 'Use of eval() at line 2',
          filePath: '<inline>',
          lineStart: 2,
          confidence: 0.9,
        },
        {
          id: 'safety-xss-3',
          type: 'xss',
          severity: 'high',
          title: 'Possible XSS via innerHTML',
          description: 'Possible XSS via innerHTML at line 3',
          filePath: '<inline>',
          lineStart: 3,
          confidence: 0.9,
        },
      ],
      categoriesChecked: ['xss', 'injection', 'eval'],
      durationMs: 0,
    });
  });

  it('returns a low-risk result for code with no matched safety patterns', async () => {
    await expect(new CodeVerifyClient().checkSafety(
      'const total = values.reduce((sum, value) => sum + value, 0);',
    )).resolves.toEqual({
      safe: true,
      riskLevel: 'low',
      issues: [],
      categoriesChecked: ['xss', 'injection', 'eval'],
      durationMs: 0,
    });
  });

  it('applies configure options to module-level verification', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ status: 'verified' }));
    configure({
      apiKey: 'module-key',
      apiUrl: 'https://module.example/',
    });

    const result = await verify('const configured = true;');

    expect(result.passed).toBe(true);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('https://module.example/api/v1/verified-autofix/generate');
    expect(init?.headers).toMatchObject({
      Authorization: 'Bearer module-key',
    });
  });
});

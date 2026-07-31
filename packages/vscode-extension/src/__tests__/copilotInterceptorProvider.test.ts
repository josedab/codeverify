/**
 * Tests for Copilot Interceptor Provider
 *
 * The real provider module (../providers/copilotInterceptorProvider) creates
 * VS Code decoration types at module-load time, so it can only be imported
 * inside the VS Code extension host. However, the types and verification
 * logic it depends on live in ../copilotVerification, a pure module with no
 * `vscode` runtime dependency (see that file's header comment), so these
 * tests import and exercise the REAL `VerificationStatus`,
 * `getDefaultInterceptorConfig`, and `MockCodeVerifyClient` from there
 * directly instead of hand-copied simulations of them.
 * `copilotInterceptorProvider.ts` re-exports all of these for backward
 * compatibility, so testing them here is equivalent to testing what the
 * real provider uses.
 */

import * as assert from 'assert';
import { describe, it } from 'node:test';
import {
    VerificationStatus,
    getDefaultInterceptorConfig,
    MockCodeVerifyClient,
    type SuggestionContext,
} from '../copilotVerification';

/**
 * Builds a minimal SuggestionContext for tests. `cursorPosition` is typed as
 * `vscode.Position` in production but is unused by `verifySuggestion`'s
 * logic, so a structurally-empty stand-in is sufficient here without
 * depending on the real `vscode` module.
 */
function makeContext(overrides: Partial<SuggestionContext> = {}): SuggestionContext {
    return {
        filePath: '/workspace/example.ts',
        language: 'typescript',
        surroundingCode: '',
        cursorPosition: {} as SuggestionContext['cursorPosition'],
        documentVersion: 1,
        ...overrides,
    };
}

describe('CopilotInterceptorProvider', () => {
    describe('VerificationStatus', () => {
        it('should have the expected status values', () => {
            assert.strictEqual(VerificationStatus.Pending, 'pending');
            assert.strictEqual(VerificationStatus.Verified, 'verified');
            assert.strictEqual(VerificationStatus.Warning, 'warning');
            assert.strictEqual(VerificationStatus.Error, 'error');
            assert.strictEqual(VerificationStatus.Timeout, 'timeout');
        });
    });

    describe('InterceptorConfig defaults', () => {
        it('should have sensible default values', () => {
            const config = getDefaultInterceptorConfig();
            assert.strictEqual(config.enabled, true);
            assert.strictEqual(config.autoVerify, true);
            assert.strictEqual(config.showInlineStatus, true);
            assert.strictEqual(config.blockOnError, false);
            assert.strictEqual(config.verificationTimeout, 5000);
            assert.strictEqual(config.minTrustScore, 60);
            assert.deepStrictEqual(config.checks, ['null_safety', 'overflow', 'bounds', 'security']);
        });
    });

    describe('MockCodeVerifyClient.verifySuggestion (provider/client path)', () => {
        it('should flag eval() usage as an error-level security issue', async () => {
            const client = new MockCodeVerifyClient();
            const result = await client.verifySuggestion('const data = eval(userInput);', makeContext());
            assert.strictEqual(result.status, VerificationStatus.Error);
            assert.ok(result.issues.some((i) => i.message.includes('eval()')));
        });

        it('should flag hardcoded passwords as an error-level security issue', async () => {
            const client = new MockCodeVerifyClient();
            const result = await client.verifySuggestion('const password = "hunter2";', makeContext());
            assert.strictEqual(result.status, VerificationStatus.Error);
            assert.ok(result.issues.some((i) => i.message.includes('hardcoded password')));
        });

        it('should flag unchecked array access as a warning', async () => {
            const client = new MockCodeVerifyClient();
            const result = await client.verifySuggestion('return items[index];', makeContext());
            assert.strictEqual(result.status, VerificationStatus.Warning);
            assert.ok(result.issues.some((i) => i.category === 'bounds'));
        });

        it('should not flag array access that is already length-checked', async () => {
            const client = new MockCodeVerifyClient();
            const result = await client.verifySuggestion(
                'if (index < items.length) { return items[index]; }',
                makeContext()
            );
            assert.strictEqual(result.issues.some((i) => i.category === 'bounds'), false);
        });

        it('should report Verified status with a perfect score for clean code', async () => {
            const client = new MockCodeVerifyClient();
            const result = await client.verifySuggestion('return a + b;', makeContext());
            assert.strictEqual(result.status, VerificationStatus.Verified);
            assert.strictEqual(result.score, 100);
            assert.strictEqual(result.issues.length, 0);
        });

        it('should deduct 20 points per detected issue', async () => {
            const client = new MockCodeVerifyClient();
            const result = await client.verifySuggestion('const password = eval(x);', makeContext());
            assert.strictEqual(result.issues.length, 2);
            assert.strictEqual(result.score, 60);
        });
    });

    describe('SuggestionVerification shape', () => {
        it('should support constructing a verification result', () => {
            const verification = {
                suggestionId: 'test-1',
                status: VerificationStatus.Warning,
                issues: [
                    {
                        line: 1,
                        column: 0,
                        message: 'Potential null reference',
                        severity: 'warning' as const,
                        category: 'null_safety',
                    },
                ],
                score: 70,
                verificationTimeMs: 120,
                metadata: {},
            };

            assert.strictEqual(verification.status, VerificationStatus.Warning);
            assert.strictEqual(verification.issues.length, 1);
            assert.strictEqual(verification.score, 70);
        });
    });
});

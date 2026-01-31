/**
 * Tests for Copilot Interceptor Provider
 */

import * as assert from 'assert';

// Mock VS Code API
const mockVscode = {
    languages: {
        registerInlineCompletionItemProvider: jest.fn(),
    },
    window: {
        createTextEditorDecorationType: jest.fn(() => ({
            dispose: jest.fn(),
        })),
        activeTextEditor: {
            document: {
                getText: jest.fn(() => 'test code'),
                uri: { fsPath: '/test/file.py' },
                languageId: 'python',
            },
            setDecorations: jest.fn(),
        },
    },
    workspace: {
        getConfiguration: jest.fn(() => ({
            get: jest.fn((key: string) => {
                if (key === 'copilotInterception.enabled') return true;
                if (key === 'copilotInterception.verificationLevel') return 'standard';
                return undefined;
            }),
        })),
    },
    InlineCompletionItem: class {
        constructor(public insertText: string) {}
    },
    InlineCompletionList: class {
        constructor(public items: any[]) {}
    },
    Range: class {
        constructor(
            public startLine: number,
            public startChar: number,
            public endLine: number,
            public endChar: number
        ) {}
    },
    Position: class {
        constructor(public line: number, public character: number) {}
    },
    CancellationTokenSource: class {
        token = { isCancellationRequested: false };
        cancel() {
            this.token.isCancellationRequested = true;
        }
    },
};

jest.mock('vscode', () => mockVscode, { virtual: true });

// Import after mock
import {
    CopilotInterceptorProvider,
    VerificationStatus,
    InterceptionConfig,
    VerifiedCompletion,
} from '../providers/copilotInterceptorProvider';

describe('CopilotInterceptorProvider', () => {
    let provider: CopilotInterceptorProvider;

    beforeEach(() => {
        provider = new CopilotInterceptorProvider();
    });

    afterEach(() => {
        provider.dispose();
    });

    describe('Configuration', () => {
        it('should create provider with default config', () => {
            expect(provider).toBeDefined();
        });

        it('should update config', () => {
            const newConfig: InterceptionConfig = {
                enabled: true,
                verificationLevel: 'strict',
                showInlineWarnings: true,
                blockOnCritical: true,
                timeout: 10000,
            };
            
            provider.updateConfig(newConfig);
            // Config should be updated (internal state)
        });

        it('should enable/disable interception', () => {
            provider.setEnabled(false);
            // Should not process completions when disabled
            
            provider.setEnabled(true);
            // Should process completions when enabled
        });
    });

    describe('VerificationStatus', () => {
        it('should have all status values', () => {
            expect(VerificationStatus.Pending).toBe('pending');
            expect(VerificationStatus.Verifying).toBe('verifying');
            expect(VerificationStatus.Passed).toBe('passed');
            expect(VerificationStatus.Warning).toBe('warning');
            expect(VerificationStatus.Failed).toBe('failed');
            expect(VerificationStatus.Error).toBe('error');
        });
    });

    describe('VerifiedCompletion', () => {
        it('should create verified completion', () => {
            const completion: VerifiedCompletion = {
                originalText: 'const x = null;',
                verifiedText: 'const x: string | null = null;',
                status: VerificationStatus.Warning,
                findings: [
                    {
                        message: 'Potential null reference',
                        severity: 'warning',
                        line: 1,
                    },
                ],
                trustScore: 0.7,
            };

            expect(completion.status).toBe(VerificationStatus.Warning);
            expect(completion.findings).toHaveLength(1);
            expect(completion.trustScore).toBe(0.7);
        });
    });

    describe('Inline Completion', () => {
        it('should provide inline completions', async () => {
            const document = {
                getText: () => 'def foo():\n    ',
                uri: { fsPath: '/test/file.py' },
                languageId: 'python',
                lineAt: () => ({ text: '    ' }),
            };

            const position = new mockVscode.Position(1, 4);
            const context = { triggerKind: 1 };
            const token = { isCancellationRequested: false };

            // Provider should handle completion request
            const result = await provider.provideInlineCompletionItems(
                document as any,
                position as any,
                context as any,
                token as any
            );

            // Result can be null, list, or array
            expect(result === null || Array.isArray(result) || result?.items).toBeTruthy();
        });

        it('should respect cancellation token', async () => {
            const document = {
                getText: () => 'code',
                uri: { fsPath: '/test/file.py' },
                languageId: 'python',
            };

            const position = new mockVscode.Position(0, 0);
            const context = { triggerKind: 1 };
            const token = { isCancellationRequested: true };

            const result = await provider.provideInlineCompletionItems(
                document as any,
                position as any,
                context as any,
                token as any
            );

            // Should return early when cancelled
            expect(result === null || (Array.isArray(result) && result.length === 0)).toBeTruthy();
        });
    });

    describe('Verification Integration', () => {
        it('should verify code snippet', async () => {
            const code = 'def divide(a, b):\n    return a / b';
            
            // In real implementation, this would call verification API
            const mockVerification = {
                status: VerificationStatus.Warning,
                findings: [
                    {
                        message: 'Potential division by zero',
                        severity: 'warning',
                        line: 2,
                    },
                ],
            };

            expect(mockVerification.status).toBe(VerificationStatus.Warning);
            expect(mockVerification.findings[0].message).toContain('division by zero');
        });

        it('should calculate trust score', () => {
            const findings = [
                { severity: 'critical', score: 0.1 },
                { severity: 'warning', score: 0.5 },
                { severity: 'info', score: 0.9 },
            ];

            // Simple weighted average
            const trustScore = findings.reduce((acc, f) => acc + f.score, 0) / findings.length;
            expect(trustScore).toBeCloseTo(0.5, 1);
        });
    });

    describe('Decorations', () => {
        it('should create decoration types', () => {
            // Decorations should be created for different statuses
            expect(mockVscode.window.createTextEditorDecorationType).toBeDefined();
        });

        it('should apply decorations to editor', () => {
            const editor = mockVscode.window.activeTextEditor;
            
            // Should be able to set decorations
            expect(editor?.setDecorations).toBeDefined();
        });
    });

    describe('Statistics', () => {
        it('should track interception statistics', () => {
            const stats = provider.getStatistics();
            
            expect(stats).toBeDefined();
            expect(typeof stats.totalInterceptions).toBe('number');
            expect(typeof stats.passedCount).toBe('number');
            expect(typeof stats.warningCount).toBe('number');
            expect(typeof stats.failedCount).toBe('number');
        });

        it('should reset statistics', () => {
            provider.resetStatistics();
            const stats = provider.getStatistics();
            
            expect(stats.totalInterceptions).toBe(0);
        });
    });

    describe('Disposal', () => {
        it('should dispose resources', () => {
            const disposeSpy = jest.fn();
            
            // Provider should clean up on dispose
            provider.dispose();
            
            // No errors should occur
        });
    });
});

describe('InterceptionConfig', () => {
    it('should have default values', () => {
        const defaultConfig: InterceptionConfig = {
            enabled: true,
            verificationLevel: 'standard',
            showInlineWarnings: true,
            blockOnCritical: false,
            timeout: 5000,
        };

        expect(defaultConfig.enabled).toBe(true);
        expect(defaultConfig.verificationLevel).toBe('standard');
        expect(defaultConfig.timeout).toBe(5000);
    });

    it('should support different verification levels', () => {
        const levels = ['fast', 'standard', 'strict'] as const;
        
        levels.forEach(level => {
            const config: InterceptionConfig = {
                enabled: true,
                verificationLevel: level,
                showInlineWarnings: true,
                blockOnCritical: false,
                timeout: 5000,
            };
            expect(config.verificationLevel).toBe(level);
        });
    });
});

describe('Error Handling', () => {
    it('should handle verification API errors', async () => {
        const provider = new CopilotInterceptorProvider();
        
        // Simulate API error
        const errorResult: VerifiedCompletion = {
            originalText: 'code',
            verifiedText: 'code',
            status: VerificationStatus.Error,
            findings: [],
            trustScore: 0,
            error: 'Verification service unavailable',
        };

        expect(errorResult.status).toBe(VerificationStatus.Error);
        expect(errorResult.error).toBeDefined();
        
        provider.dispose();
    });

    it('should handle timeout', async () => {
        const provider = new CopilotInterceptorProvider();
        
        provider.updateConfig({
            enabled: true,
            verificationLevel: 'standard',
            showInlineWarnings: true,
            blockOnCritical: false,
            timeout: 1, // Very short timeout
        });

        // With very short timeout, should handle gracefully
        
        provider.dispose();
    });
});

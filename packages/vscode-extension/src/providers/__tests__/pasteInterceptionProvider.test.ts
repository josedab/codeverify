/**
 * Tests for PasteInterceptionProvider
 *
 * `../pasteInterceptionProvider`'s constructor eagerly creates VS Code
 * status bar items and event emitters, so the class itself can only be
 * exercised inside the VS Code extension host. Its actual decision logic
 * (code detection, AI/security pattern matching, trust scoring, risk-level
 * derivation, statistics) lives in the pure, `vscode`-free
 * `../../pasteAnalysis` and `../../localAnalysis` modules (moved there
 * verbatim so it can be unit tested directly), and the provider delegates
 * to them. This file imports and exercises those real functions instead of
 * hand-copied simulations of them. Numeric expectations below were verified
 * directly against the compiled production functions before being written.
 */

import * as assert from 'assert';
import { describe, it } from 'node:test';
import type { Finding, TrustScore } from '../../client';
import { localTrustScore, localQuickAnalysis } from '../../localAnalysis';
import {
    looksLikeCode,
    determineRiskLevel,
    detectModel,
    generateRecommendations,
    getDefaultTrustScore,
    initializeStatistics,
    updatePasteStatistics,
    updateDecisionStatistics,
} from '../../pasteAnalysis';
import type { PasteAnalysisResult } from '../pasteInterceptionProvider';

function makeTrustScore(overrides: Partial<TrustScore> = {}): TrustScore {
    return {
        score: 70,
        ai_probability: 0,
        risk_level: '',
        complexity_score: 0,
        pattern_score: 0,
        quality_score: 0,
        verification_score: 0,
        factors: {},
        ...overrides,
    };
}

function makeAnalysisResult(overrides: Partial<PasteAnalysisResult> = {}): PasteAnalysisResult {
    return {
        id: 'test-1',
        code: '',
        isAiGenerated: false,
        aiConfidence: 0,
        trustScore: 70,
        riskLevel: 'medium',
        findings: [],
        detectedModel: 'unknown',
        recommendations: [],
        analysisTimeMs: 10,
        ...overrides,
    };
}

describe('PasteInterceptionProvider', () => {
    describe('looksLikeCode', () => {
        it('should detect Python code', () => {
            const pythonCode = `
def calculate_sum(a: int, b: int) -> int:
    # Add two numbers
    return a + b
`;
            assert.strictEqual(looksLikeCode(pythonCode, 'python'), true);
        });

        it('should detect TypeScript code', () => {
            const tsCode = `
function greet(name: string): void {
    console.log(\`Hello, \${name}!\`);
}
`;
            assert.strictEqual(looksLikeCode(tsCode, 'typescript'), true);
        });

        it('should detect JavaScript arrow functions', () => {
            const jsCode = `
const add = (a, b) => {
    return a + b;
};
`;
            assert.strictEqual(looksLikeCode(jsCode, 'javascript'), true);
        });

        it('should not detect plain text', () => {
            const plainText = `
This is just some plain text
that doesn't contain any code.
`;
            assert.strictEqual(looksLikeCode(plainText, 'plaintext'), false);
        });

        it('should not detect single-line content', () => {
            const singleLine = 'def foo(): pass';
            assert.strictEqual(looksLikeCode(singleLine, 'python'), false);
        });
    });

    describe('AI detection (localTrustScore.ai_probability)', () => {
        it('should detect placeholder patterns', () => {
            const code = `
def foo():
    pass  # placeholder
`;
            assert.ok(localTrustScore(code, 'python').ai_probability >= 25);
        });

        it('should detect generic TODO patterns', () => {
            const code = `
def bar():
    # TODO: implement this function
    pass
`;
            assert.ok(localTrustScore(code, 'python').ai_probability >= 25);
        });

        it('should detect verbose AI-style comments', () => {
            const code = `
def calculate(x, y):
    # This function does the calculation of two numbers
    return x + y
`;
            assert.ok(localTrustScore(code, 'python').ai_probability >= 25);
        });

        it('should detect NotImplementedError stubs', () => {
            const code = `
def process_data(data):
    raise NotImplementedError
`;
            assert.ok(localTrustScore(code, 'python').ai_probability >= 25);
        });

        it('should not detect patterns in clean human code', () => {
            const code = `
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b
`;
            assert.strictEqual(localTrustScore(code, 'python').ai_probability, 0);
        });
    });

    describe('Security pattern detection (localQuickAnalysis)', () => {
        it('should detect eval usage', () => {
            const code = `
result = eval(user_input)
`;
            const findings = localQuickAnalysis(code, 'python');
            assert.ok(findings.some(f => f.title === 'Unsafe eval() usage detected' && f.severity === 'critical'));
        });

        it('should detect hardcoded passwords', () => {
            const code = `
password = "supersecret123"
`;
            const findings = localQuickAnalysis(code, 'python');
            assert.ok(findings.some(f => f.title === 'Hardcoded password detected' && f.severity === 'critical'));
        });

        it('should detect shell injection risk', () => {
            const code = `
subprocess.run(cmd, shell=True)
`;
            const findings = localQuickAnalysis(code, 'python');
            assert.ok(findings.some(f => f.title === 'Shell injection risk' && f.severity === 'high'));
        });

        it('should pass clean code', () => {
            const code = `
def safe_function(data):
    return data.strip()
`;
            const findings = localQuickAnalysis(code, 'python');
            assert.strictEqual(findings.length, 0);
        });
    });

    describe('Trust score calculation (localTrustScore)', () => {
        const humanBaseline = 'def foo():\n    return 1\n';
        const qualityCode =
            'def test_add():\n    assert add(1, 2) == 3\n    try:\n        pass\n    except ValueError:\n        pass\n';
        const evalCode = 'result = eval(user_input)\n';
        const aiCode =
            'def foo():\n    pass  # placeholder\n    # TODO: implement this function\n' +
            '    # This function does the calculation\n    raise NotImplementedError\n';
        const lowConfidenceAiCode = 'def foo():\n    pass  # placeholder\n    return 1\n';

        it('should return a high score for clean, well-tested code', () => {
            const score = localTrustScore(qualityCode, 'python').score;
            assert.ok(score >= 80, `expected >= 80, got ${score}`);
        });

        it('should penalize security issues', () => {
            const scoreClean = localTrustScore(qualityCode, 'python').score;
            const scoreWithIssues = localTrustScore(evalCode, 'python').score;
            assert.ok(scoreWithIssues < scoreClean);
        });

        it('should penalize heavily AI-flagged code', () => {
            const scoreHuman = localTrustScore(humanBaseline, 'python').score;
            const scoreAi = localTrustScore(aiCode, 'python').score;
            assert.ok(scoreAi < scoreHuman);
        });

        it('should not penalize low-confidence AI detection', () => {
            const scoreHuman = localTrustScore(humanBaseline, 'python').score;
            const scoreLowConfAi = localTrustScore(lowConfidenceAiCode, 'python').score;
            assert.strictEqual(scoreHuman, scoreLowConfAi);
        });

        it('should reward quality signals', () => {
            const scoreNoQuality = localTrustScore(humanBaseline, 'python').score;
            const scoreWithQuality = localTrustScore(qualityCode, 'python').score;
            assert.ok(scoreWithQuality > scoreNoQuality);
        });
    });

    describe('Risk level determination (determineRiskLevel)', () => {
        it('should return low for high trust scores', () => {
            assert.strictEqual(determineRiskLevel(makeTrustScore({ score: 85 })), 'low');
        });

        it('should return medium for moderate trust scores', () => {
            assert.strictEqual(determineRiskLevel(makeTrustScore({ score: 70 })), 'medium');
        });

        it('should return high for low trust scores', () => {
            assert.strictEqual(determineRiskLevel(makeTrustScore({ score: 50 })), 'high');
        });

        it('should return critical for very low trust scores', () => {
            assert.strictEqual(determineRiskLevel(makeTrustScore({ score: 20 })), 'critical');
        });

        it("should prefer the trust score's own risk_level over the derived score", () => {
            // A low score would normally derive to 'critical', but an
            // explicit risk_level (e.g. set by the server) always wins.
            const result = determineRiskLevel(makeTrustScore({ score: 30, risk_level: 'low' }));
            assert.strictEqual(result, 'low');
        });
    });

    describe('detectModel', () => {
        it('should detect GitHub Copilot from telltale comments', () => {
            assert.strictEqual(detectModel('# Copilot suggestion below', makeTrustScore()), 'GitHub Copilot');
        });

        it('should return unknown when no model pattern matches', () => {
            assert.strictEqual(detectModel('def add(a, b):\n    return a + b\n', makeTrustScore()), 'unknown');
        });
    });

    describe('generateRecommendations', () => {
        it('should recommend manual review for high AI probability', () => {
            const recs = generateRecommendations(makeTrustScore({ score: 80, ai_probability: 80 }), []);
            assert.ok(recs.includes('Manual review recommended for AI-generated code'));
        });

        it('should recommend fixing critical findings', () => {
            const findings: Finding[] = [
                {
                    category: 'logic_error',
                    severity: 'critical',
                    title: 'Null pointer',
                    description: '',
                    file_path: 'a.ts',
                    line_start: 1,
                    confidence: 1,
                    verification_type: 'pattern',
                },
            ];
            const recs = generateRecommendations(makeTrustScore({ score: 80 }), findings);
            assert.ok(recs.includes('Critical issues detected - fix before committing'));
        });

        it('should fall back to a generic recommendation when nothing else applies', () => {
            const recs = generateRecommendations(makeTrustScore({ score: 90 }), []);
            assert.deepStrictEqual(recs, ['Code looks good, but manual review is still recommended']);
        });
    });

    describe('getDefaultTrustScore', () => {
        it('should return a conservative medium-risk default', () => {
            const trustScore = getDefaultTrustScore();
            assert.strictEqual(trustScore.score, 50);
            assert.strictEqual(trustScore.risk_level, 'medium');
        });
    });

    describe('Statistics tracking (initializeStatistics / updatePasteStatistics / updateDecisionStatistics)', () => {
        it('should track total interceptions and per-decision counts', () => {
            let stats = initializeStatistics();
            stats = updatePasteStatistics(stats, makeAnalysisResult({ trustScore: 80 }));
            stats = updateDecisionStatistics(stats, 'accept');
            stats = updatePasteStatistics(stats, makeAnalysisResult({ isAiGenerated: true, trustScore: 60 }));
            stats = updateDecisionStatistics(stats, 'reject');
            stats = updatePasteStatistics(stats, makeAnalysisResult({ isAiGenerated: true, trustScore: 50 }));
            stats = updateDecisionStatistics(stats, 'modify');

            assert.strictEqual(stats.totalInterceptions, 3);
            assert.strictEqual(stats.acceptedCount, 1);
            assert.strictEqual(stats.rejectedCount, 1);
            assert.strictEqual(stats.modifiedCount, 1);
        });

        it('should track AI detection rate', () => {
            let stats = initializeStatistics();
            stats = updatePasteStatistics(stats, makeAnalysisResult({ isAiGenerated: true, trustScore: 60 }));
            stats = updatePasteStatistics(stats, makeAnalysisResult({ isAiGenerated: true, trustScore: 50 }));
            stats = updatePasteStatistics(stats, makeAnalysisResult({ isAiGenerated: false, trustScore: 80 }));

            assert.strictEqual(stats.aiDetectedCount, 2);
            const aiRate = stats.aiDetectedCount / stats.totalInterceptions;
            assert.ok(Math.abs(aiRate - 0.667) < 0.01);
        });

        it('should calculate rolling average trust score', () => {
            let stats = initializeStatistics();
            stats = updatePasteStatistics(stats, makeAnalysisResult({ trustScore: 80 }));
            stats = updatePasteStatistics(stats, makeAnalysisResult({ trustScore: 60 }));
            stats = updatePasteStatistics(stats, makeAnalysisResult({ trustScore: 70 }));

            // Average of 80, 60, 70 = 70
            assert.strictEqual(stats.averageTrustScore, 70);
        });
    });
});

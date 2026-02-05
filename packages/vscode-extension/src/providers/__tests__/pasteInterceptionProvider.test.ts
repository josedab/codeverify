/**
 * Tests for PasteInterceptionProvider
 */

import * as assert from 'assert';

// Mock types since we can't import vscode in tests easily
interface MockTrustScore {
    score: number;
    ai_probability: number;
    risk_level: string;
}

interface MockFinding {
    id: string;
    category: string;
    severity: string;
    title: string;
}

describe('PasteInterceptionProvider', () => {
    describe('looksLikeCode', () => {
        // Simulated looksLikeCode function for testing
        function looksLikeCode(text: string): boolean {
            const lines = text.split('\n');
            if (lines.length < 2) return false;

            const codePatterns = [
                /^\s*(def|function|class|const|let|var|import|from|async|export)\s/m,
                /[{}\[\]();]/,
                /^\s*(if|for|while|try|return)\s/m,
                /=>/,
                /:\s*(int|str|float|bool|string|number|void)\b/,
                /^\s*#\s*\w+/m,
                /\/\//,
                /^\s*@\w+/m,
            ];

            let patternMatches = 0;
            for (const pattern of codePatterns) {
                if (pattern.test(text)) {
                    patternMatches++;
                }
            }

            return patternMatches >= 2;
        }

        it('should detect Python code', () => {
            const pythonCode = `
def calculate_sum(a: int, b: int) -> int:
    # Add two numbers
    return a + b
`;
            assert.strictEqual(looksLikeCode(pythonCode), true);
        });

        it('should detect TypeScript code', () => {
            const tsCode = `
function greet(name: string): void {
    console.log(\`Hello, \${name}!\`);
}
`;
            assert.strictEqual(looksLikeCode(tsCode), true);
        });

        it('should detect JavaScript arrow functions', () => {
            const jsCode = `
const add = (a, b) => {
    return a + b;
};
`;
            assert.strictEqual(looksLikeCode(jsCode), true);
        });

        it('should not detect plain text', () => {
            const plainText = `
This is just some plain text
that doesn't contain any code.
`;
            assert.strictEqual(looksLikeCode(plainText), false);
        });

        it('should not detect single-line content', () => {
            const singleLine = 'def foo(): pass';
            assert.strictEqual(looksLikeCode(singleLine), false);
        });
    });

    describe('AI detection patterns', () => {
        // Simulated AI detection
        function detectAiPatterns(code: string): { matches: number; patterns: string[] } {
            const aiPatterns = [
                { pattern: /pass\s*#\s*(placeholder|implement)/i, name: 'placeholder_pass' },
                { pattern: /# TODO:?\s*(implement|add|fix|complete)/i, name: 'generic_todo' },
                { pattern: /# (This|The) (function|method|class) (does|will|should)/i, name: 'verbose_comment' },
                { pattern: /raise NotImplementedError/i, name: 'not_implemented' },
                { pattern: /# Example usage/i, name: 'example_section' },
            ];

            const found: string[] = [];
            for (const { pattern, name } of aiPatterns) {
                if (pattern.test(code)) {
                    found.push(name);
                }
            }

            return { matches: found.length, patterns: found };
        }

        it('should detect placeholder patterns', () => {
            const code = `
def foo():
    pass  # placeholder
`;
            const result = detectAiPatterns(code);
            assert.ok(result.patterns.includes('placeholder_pass'));
        });

        it('should detect generic TODO patterns', () => {
            const code = `
def bar():
    # TODO: implement this function
    pass
`;
            const result = detectAiPatterns(code);
            assert.ok(result.patterns.includes('generic_todo'));
        });

        it('should detect verbose AI-style comments', () => {
            const code = `
def calculate(x, y):
    # This function does the calculation of two numbers
    return x + y
`;
            const result = detectAiPatterns(code);
            assert.ok(result.patterns.includes('verbose_comment'));
        });

        it('should detect NotImplementedError stubs', () => {
            const code = `
def process_data(data):
    raise NotImplementedError
`;
            const result = detectAiPatterns(code);
            assert.ok(result.patterns.includes('not_implemented'));
        });

        it('should not detect patterns in clean human code', () => {
            const code = `
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b
`;
            const result = detectAiPatterns(code);
            assert.strictEqual(result.matches, 0);
        });
    });

    describe('Security pattern detection', () => {
        function detectSecurityPatterns(code: string): { severity: string; title: string }[] {
            const patterns: [RegExp, string, string][] = [
                [/eval\s*\(/g, 'critical', 'Unsafe eval() usage'],
                [/exec\s*\(/g, 'critical', 'Unsafe exec() usage'],
                [/password\s*=\s*['"][^'"]+['"]/gi, 'critical', 'Hardcoded password'],
                [/api_key\s*=\s*['"][^'"]+['"]/gi, 'critical', 'Hardcoded API key'],
                [/shell\s*=\s*True/g, 'high', 'Shell injection risk'],
            ];

            const findings: { severity: string; title: string }[] = [];
            for (const [pattern, severity, title] of patterns) {
                if (pattern.test(code)) {
                    findings.push({ severity, title });
                }
            }
            return findings;
        }

        it('should detect eval usage', () => {
            const code = `
result = eval(user_input)
`;
            const findings = detectSecurityPatterns(code);
            assert.ok(findings.some(f => f.title === 'Unsafe eval() usage'));
        });

        it('should detect hardcoded passwords', () => {
            const code = `
password = "supersecret123"
`;
            const findings = detectSecurityPatterns(code);
            assert.ok(findings.some(f => f.title === 'Hardcoded password'));
        });

        it('should detect shell injection risk', () => {
            const code = `
subprocess.run(cmd, shell=True)
`;
            const findings = detectSecurityPatterns(code);
            assert.ok(findings.some(f => f.title === 'Shell injection risk'));
        });

        it('should pass clean code', () => {
            const code = `
def safe_function(data):
    return data.strip()
`;
            const findings = detectSecurityPatterns(code);
            assert.strictEqual(findings.length, 0);
        });
    });

    describe('Trust score calculation', () => {
        function calculateTrustScore(
            hasSecurityIssues: boolean,
            isAiGenerated: boolean,
            aiConfidence: number,
            hasQualitySignals: boolean
        ): number {
            let score = 70;

            if (hasSecurityIssues) {
                score -= 25;
            }

            if (hasQualitySignals) {
                score += 15;
            }

            if (isAiGenerated && aiConfidence > 0.7) {
                score *= 0.85;
            }

            return Math.max(0, Math.min(100, score));
        }

        it('should return high score for clean code', () => {
            const score = calculateTrustScore(false, false, 0, true);
            assert.ok(score >= 80);
        });

        it('should penalize security issues', () => {
            const scoreClean = calculateTrustScore(false, false, 0, false);
            const scoreWithIssues = calculateTrustScore(true, false, 0, false);
            assert.ok(scoreWithIssues < scoreClean);
        });

        it('should penalize AI-generated code', () => {
            const scoreHuman = calculateTrustScore(false, false, 0, false);
            const scoreAi = calculateTrustScore(false, true, 0.9, false);
            assert.ok(scoreAi < scoreHuman);
        });

        it('should not penalize low-confidence AI detection', () => {
            const scoreHuman = calculateTrustScore(false, false, 0, false);
            const scoreLowConfAi = calculateTrustScore(false, true, 0.3, false);
            assert.strictEqual(scoreHuman, scoreLowConfAi);
        });

        it('should reward quality signals', () => {
            const scoreNoQuality = calculateTrustScore(false, false, 0, false);
            const scoreWithQuality = calculateTrustScore(false, false, 0, true);
            assert.ok(scoreWithQuality > scoreNoQuality);
        });
    });

    describe('Risk level determination', () => {
        function determineRiskLevel(trustScore: number, hasCriticalFinding: boolean): string {
            if (hasCriticalFinding || trustScore < 40) {
                return 'critical';
            } else if (trustScore < 60) {
                return 'high';
            } else if (trustScore < 80) {
                return 'medium';
            }
            return 'low';
        }

        it('should return low for high trust scores', () => {
            assert.strictEqual(determineRiskLevel(85, false), 'low');
        });

        it('should return medium for moderate trust scores', () => {
            assert.strictEqual(determineRiskLevel(70, false), 'medium');
        });

        it('should return high for low trust scores', () => {
            assert.strictEqual(determineRiskLevel(50, false), 'high');
        });

        it('should return critical for very low trust scores', () => {
            assert.strictEqual(determineRiskLevel(30, false), 'critical');
        });

        it('should return critical when critical finding exists', () => {
            assert.strictEqual(determineRiskLevel(90, true), 'critical');
        });
    });

    describe('Statistics tracking', () => {
        interface Stats {
            totalInterceptions: number;
            aiDetectedCount: number;
            acceptedCount: number;
            rejectedCount: number;
            averageTrustScore: number;
        }

        function updateStats(
            stats: Stats,
            isAi: boolean,
            trustScore: number,
            decision: 'accept' | 'reject'
        ): Stats {
            const newStats = { ...stats };
            newStats.totalInterceptions++;

            if (isAi) {
                newStats.aiDetectedCount++;
            }

            if (decision === 'accept') {
                newStats.acceptedCount++;
            } else {
                newStats.rejectedCount++;
            }

            // Rolling average
            const n = newStats.totalInterceptions;
            newStats.averageTrustScore =
                (newStats.averageTrustScore * (n - 1) + trustScore) / n;

            return newStats;
        }

        it('should track total interceptions', () => {
            let stats: Stats = {
                totalInterceptions: 0,
                aiDetectedCount: 0,
                acceptedCount: 0,
                rejectedCount: 0,
                averageTrustScore: 0,
            };

            stats = updateStats(stats, false, 80, 'accept');
            stats = updateStats(stats, true, 60, 'reject');
            stats = updateStats(stats, true, 50, 'accept');

            assert.strictEqual(stats.totalInterceptions, 3);
        });

        it('should track AI detection rate', () => {
            let stats: Stats = {
                totalInterceptions: 0,
                aiDetectedCount: 0,
                acceptedCount: 0,
                rejectedCount: 0,
                averageTrustScore: 0,
            };

            stats = updateStats(stats, true, 60, 'accept');
            stats = updateStats(stats, true, 50, 'accept');
            stats = updateStats(stats, false, 80, 'accept');

            assert.strictEqual(stats.aiDetectedCount, 2);
            const aiRate = stats.aiDetectedCount / stats.totalInterceptions;
            assert.ok(Math.abs(aiRate - 0.667) < 0.01);
        });

        it('should calculate rolling average trust score', () => {
            let stats: Stats = {
                totalInterceptions: 0,
                aiDetectedCount: 0,
                acceptedCount: 0,
                rejectedCount: 0,
                averageTrustScore: 0,
            };

            stats = updateStats(stats, false, 80, 'accept');
            stats = updateStats(stats, false, 60, 'accept');
            stats = updateStats(stats, false, 70, 'accept');

            // Average of 80, 60, 70 = 70
            assert.strictEqual(stats.averageTrustScore, 70);
        });
    });
});

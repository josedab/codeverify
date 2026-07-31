/**
 * Local (offline) analysis fallbacks used by CodeVerifyClient.
 *
 * These are pure, dependency-free functions (no `vscode` import) so they can
 * be unit tested directly with Node's built-in test runner, without needing
 * to run inside the VS Code extension host. `client.ts` imports and delegates
 * to these functions when no API endpoint is configured or the API call
 * fails; keeping them here (rather than as private methods on
 * `CodeVerifyClient`) lets tests exercise the exact production algorithm
 * instead of a hand-copied simulation of it.
 */

import type { Finding, NLToZ3Result, SpecTemplate, TrustScore } from './client';

/**
 * Local quick analysis using pattern matching
 */
export function localQuickAnalysis(code: string, language: string): Finding[] {
    const findings: Finding[] = [];

    // Quick pattern-based checks
    const patterns: [RegExp, string, string, string][] = [
        [/eval\s*\(/g, 'security', 'critical', 'Unsafe eval() usage detected'],
        [/exec\s*\(/g, 'security', 'critical', 'Unsafe exec() usage detected'],
        [/password\s*=\s*['"][^'"]+['"]/gi, 'security', 'critical', 'Hardcoded password detected'],
        [/api_key\s*=\s*['"][^'"]+['"]/gi, 'security', 'critical', 'Hardcoded API key detected'],
        [/except\s*:\s*pass/g, 'logic_error', 'high', 'Silent exception swallowing'],
        [/raise\s+NotImplementedError/g, 'logic_error', 'medium', 'Unimplemented function stub'],
        [/TODO:?\s*(implement|add|fix)/gi, 'logic_error', 'low', 'TODO comment found'],
        [/shell\s*=\s*True/g, 'security', 'high', 'Shell injection risk'],
        [/verify\s*=\s*False/g, 'security', 'high', 'SSL verification disabled'],
    ];

    for (const [pattern, category, severity, title] of patterns) {
        const matches = code.match(pattern);
        if (matches) {
            for (let i = 0; i < Math.min(matches.length, 3); i++) {
                findings.push({
                    id: `quick-${category}-${findings.length}`,
                    category,
                    severity,
                    title,
                    description: `Pattern detected: ${matches[i]}`,
                    file_path: 'clipboard',
                    line_start: 1,
                    confidence: 0.8,
                    verification_type: 'pattern',
                });
            }
        }
    }

    return findings;
}

/**
 * Local NL-to-Z3 conversion using simple pattern matching
 */
export function localNLToZ3(specification: string): NLToZ3Result {
    const normalized = specification.toLowerCase().trim();
    let z3_expr: string | undefined;
    let python_assert: string | undefined;
    let explanation = '';
    let confidence = 0;
    const variables: Record<string, string> = {};

    // Pattern matching for common specifications
    const patterns: Array<{
        pattern: RegExp;
        template: (matches: RegExpMatchArray) => { z3: string; py: string; vars: Record<string, string> };
        name: string;
    }> = [
        {
            pattern: /(\w+)\s+(?:must be |is |should be )?positive/,
            template: (m) => ({
                z3: `${m[1]} > 0`,
                py: `assert ${m[1]} > 0`,
                vars: { [m[1]]: 'Int' },
            }),
            name: 'Positive constraint',
        },
        {
            pattern: /(\w+)\s+(?:must be |is |should be )?non-negative/,
            template: (m) => ({
                z3: `${m[1]} >= 0`,
                py: `assert ${m[1]} >= 0`,
                vars: { [m[1]]: 'Int' },
            }),
            name: 'Non-negative constraint',
        },
        {
            pattern: /(\w+)\s+(?:must be |is |should be )?(?:between|in range)\s+(\d+)\s+(?:and|to)\s+(\d+)/,
            template: (m) => ({
                z3: `And(${m[1]} >= ${m[2]}, ${m[1]} <= ${m[3]})`,
                py: `assert ${m[2]} <= ${m[1]} <= ${m[3]}`,
                vars: { [m[1]]: 'Int' },
            }),
            name: 'Range constraint',
        },
        {
            pattern: /(\w+)\s+(?:must be |is |should be )?less than\s+(\w+)/,
            template: (m) => ({
                z3: `${m[1]} < ${m[2]}`,
                py: `assert ${m[1]} < ${m[2]}`,
                vars: { [m[1]]: 'Int', [m[2]]: 'Int' },
            }),
            name: 'Less than constraint',
        },
        {
            pattern: /(\w+)\s+(?:must |should )?not (?:be )?(?:null|none)/,
            template: (m) => ({
                z3: `${m[1]} != None`,
                py: `assert ${m[1]} is not None`,
                vars: { [m[1]]: 'Any' },
            }),
            name: 'Not null constraint',
        },
        {
            pattern: /(\w+)\s+(?:must |should )?not (?:be )?empty/,
            template: (m) => ({
                z3: `Length(${m[1]}) > 0`,
                py: `assert len(${m[1]}) > 0`,
                vars: { [m[1]]: 'Seq' },
            }),
            name: 'Not empty constraint',
        },
    ];

    for (const { pattern, template, name } of patterns) {
        const match = normalized.match(pattern);
        if (match) {
            const result = template(match);
            z3_expr = result.z3;
            python_assert = result.py;
            Object.assign(variables, result.vars);
            explanation = `Matched template: ${name}`;
            confidence = 0.85;
            break;
        }
    }

    return {
        success: !!z3_expr,
        z3_expr,
        python_assert,
        explanation: explanation || 'Could not match specification pattern',
        confidence,
        variables,
        ambiguities: !z3_expr ? ['Could not parse specification'] : [],
        clarification_questions: !z3_expr ? ['Which variable should this constraint apply to?'] : [],
        processing_time_ms: 0,
    };
}

/**
 * Local spec suggestions based on function signature
 */
export function localSuggestSpecs(signature: string): { suggestions: string[]; count: number } {
    const suggestions: string[] = [];

    // Extract parameter names and types
    const paramPattern = /(\w+)\s*:\s*(\w+)/g;
    let match;

    while ((match = paramPattern.exec(signature)) !== null) {
        const [, paramName, paramType] = match;
        const typeLower = paramType.toLowerCase();

        if (typeLower === 'int' || typeLower === 'integer') {
            suggestions.push(`${paramName} must be positive`);
            suggestions.push(`${paramName} must be non-negative`);
        } else if (typeLower === 'str' || typeLower === 'string') {
            suggestions.push(`${paramName} must not be empty`);
        } else if (typeLower.includes('list') || typeLower.includes('array')) {
            suggestions.push(`${paramName} must not be empty`);
        }
    }

    // Check for return type
    if (signature.includes('-> int') || signature.includes('-> Int')) {
        suggestions.push('the function returns a positive value');
    }

    return { suggestions, count: suggestions.length };
}

/**
 * Get local template library
 */
export function getLocalTemplates(): SpecTemplate[] {
    return [
        {
            id: 'positive',
            name: 'Positive Number',
            domain: 'numeric',
            complexity: 'simple',
            nl_pattern: '{var} must be positive',
            z3_template: '{var} > 0',
            smtlib_template: '(assert (> {var} 0))',
            python_template: 'assert {var} > 0',
            examples: [{ nl: 'x must be positive', z3: 'x > 0' }],
        },
        {
            id: 'non_negative',
            name: 'Non-negative Number',
            domain: 'numeric',
            complexity: 'simple',
            nl_pattern: '{var} must be non-negative',
            z3_template: '{var} >= 0',
            smtlib_template: '(assert (>= {var} 0))',
            python_template: 'assert {var} >= 0',
            examples: [{ nl: 'index must be non-negative', z3: 'index >= 0' }],
        },
        {
            id: 'range',
            name: 'Value in Range',
            domain: 'numeric',
            complexity: 'simple',
            nl_pattern: '{var} must be between {min} and {max}',
            z3_template: 'And({var} >= {min}, {var} <= {max})',
            smtlib_template: '(assert (and (>= {var} {min}) (<= {var} {max})))',
            python_template: 'assert {min} <= {var} <= {max}',
            examples: [{ nl: 'age must be between 0 and 150', z3: 'And(age >= 0, age <= 150)' }],
        },
        {
            id: 'not_null',
            name: 'Not Null',
            domain: 'general',
            complexity: 'simple',
            nl_pattern: '{var} must not be null',
            z3_template: '{var} != None',
            smtlib_template: '(assert (not (= {var} nil)))',
            python_template: 'assert {var} is not None',
            examples: [{ nl: 'user must not be null', z3: 'user != None' }],
        },
        {
            id: 'not_empty',
            name: 'Not Empty',
            domain: 'collection',
            complexity: 'simple',
            nl_pattern: '{var} must not be empty',
            z3_template: 'Length({var}) > 0',
            smtlib_template: '(assert (> (seq.len {var}) 0))',
            python_template: 'assert len({var}) > 0',
            examples: [{ nl: 'items must not be empty', z3: 'Length(items) > 0' }],
        },
    ];
}

/**
 * Local trust score calculation
 */
export function localTrustScore(code: string, language: string): TrustScore {
    let score = 70; // Base score
    let aiProbability = 0;

    // AI detection patterns
    const aiPatterns = [
        /pass\s*#\s*(placeholder|implement)/i,
        /# TODO:?\s*(implement|add|fix|complete)/i,
        /# (This|The) (function|method|class) (does|will|should)/i,
        /raise NotImplementedError/,
        /# Example usage/i,
    ];

    let aiMatches = 0;
    for (const pattern of aiPatterns) {
        if (pattern.test(code)) {
            aiMatches++;
        }
    }
    aiProbability = Math.min(aiMatches * 25, 95);

    // Quality patterns (positive)
    const qualityPatterns = [
        /def test_/,
        /assert\s+/,
        /try:\s*\n.*\n\s*except\s+\w+/,
        /:\s*(int|str|float|bool|list|dict|Optional|Union)/,
        /"""[\s\S]*?Args:/,
    ];

    let qualityScore = 0;
    for (const pattern of qualityPatterns) {
        if (pattern.test(code)) {
            qualityScore += 5;
        }
    }
    score += qualityScore;

    // Risk patterns (negative)
    const riskPatterns = [
        [/eval\s*\(/, 20],
        [/exec\s*\(/, 20],
        [/password\s*=\s*['"]/, 25],
        [/shell\s*=\s*True/, 15],
    ];

    for (const [pattern, penalty] of riskPatterns) {
        if ((pattern as RegExp).test(code)) {
            score -= penalty as number;
        }
    }

    // AI penalty
    if (aiProbability > 70) {
        score *= 0.85;
    }

    score = Math.max(0, Math.min(100, score));

    const riskLevel = score >= 80 ? 'low' :
                     score >= 60 ? 'medium' :
                     score >= 40 ? 'high' : 'critical';

    return {
        score: Math.round(score),
        ai_probability: aiProbability,
        risk_level: riskLevel,
        complexity_score: 0,
        pattern_score: 0,
        quality_score: qualityScore,
        verification_score: 0,
        factors: {},
    };
}

/**
 * Extract the ordered, de-duplicated list of `{varName}` placeholders from a
 * spec template's natural-language pattern (e.g. `{var} must be positive`
 * -> `['var']`). Mirrors the placeholder-scanning logic in
 * `FormalSpecAssistantProvider.useTemplate`.
 */
export function extractTemplateVariables(nlPattern: string): string[] {
    const varPattern = /\{(\w+)\}/g;
    const variables: string[] = [];
    let match: RegExpExecArray | null;
    while ((match = varPattern.exec(nlPattern)) !== null) {
        if (!variables.includes(match[1])) {
            variables.push(match[1]);
        }
    }
    return variables;
}

/**
 * Replace every `{varName}` placeholder in `nlPattern` with its corresponding
 * value from `values`. Variables without a provided value are left
 * untouched. Mirrors the template-filling logic in
 * `FormalSpecAssistantProvider.useTemplate`.
 */
export function fillTemplate(nlPattern: string, values: Record<string, string>): string {
    let spec = nlPattern;
    for (const [varName, value] of Object.entries(values)) {
        spec = spec.replace(new RegExp(`\\{${varName}\\}`, 'g'), value);
    }
    return spec;
}

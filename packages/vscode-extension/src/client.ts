/**
 * CodeVerify API Client
 *
 * Handles communication with CodeVerify API and CLI.
 */

import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';
import axios, { AxiosInstance } from 'axios';
import { logger } from './logger';

const execAsync = promisify(exec);

export interface Finding {
    id?: string;
    category: string;
    severity: string;
    title: string;
    description: string;
    file_path: string;
    line_start: number;
    line_end?: number;
    confidence: number;
    verification_type: string;
    fix_suggestion?: string;
}

export interface TrustScore {
    score: number;
    ai_probability: number;
    risk_level: string;
    complexity_score: number;
    pattern_score: number;
    quality_score: number;
    verification_score: number;
    factors: Record<string, number>;
}

export interface NLToZ3Result {
    success: boolean;
    z3_expr?: string;
    smtlib?: string;
    python_assert?: string;
    explanation: string;
    confidence: number;
    variables: Record<string, string>;
    ambiguities: string[];
    clarification_questions: string[];
    processing_time_ms: number;
}

export interface SpecTemplate {
    id: string;
    name: string;
    domain: string;
    complexity: string;
    nl_pattern: string;
    z3_template: string;
    smtlib_template: string;
    python_template: string;
    examples: Array<{ nl: string; z3: string }>;
}

export interface SpecValidationResult {
    is_satisfiable: boolean;
    message?: string;
    model?: Record<string, any>;
}

export interface ClientConfig {
    apiEndpoint: string;
    apiKey: string;
    cliPath: string;
    localAnalysisEnabled: boolean;
}

export class CodeVerifyClient {
    private config: ClientConfig;
    private httpClient: AxiosInstance | null = null;

    constructor(config: ClientConfig) {
        this.config = config;
        
        if (config.apiKey) {
            this.httpClient = axios.create({
                baseURL: config.apiEndpoint,
                headers: {
                    'Authorization': `Bearer ${config.apiKey}`,
                    'Content-Type': 'application/json',
                },
                timeout: 60000,
            });
        }
    }

    /**
     * Analyze a single file
     */
    async analyzeFile(filePath: string): Promise<Finding[]> {
        // Try local CLI first if enabled
        if (this.config.localAnalysisEnabled) {
            try {
                return await this.analyzeWithCli(filePath);
            } catch (error) {
                logger.debug('CLI analysis failed, falling back to API', error);
            }
        }

        // Fall back to API
        if (this.httpClient) {
            return await this.analyzeWithApi(filePath);
        }

        return [];
    }

    /**
     * Analyze code snippet
     */
    async analyzeCode(code: string, language: string): Promise<Finding[]> {
        if (this.httpClient) {
            try {
                const response = await this.httpClient.post('/api/v1/analyses/inline', {
                    content: code,
                    language: language,
                });
                return response.data.findings || [];
            } catch (error) {
                logger.warn('API analysis failed', error);
            }
        }

        // Fall back to CLI with temp file
        if (this.config.localAnalysisEnabled) {
            const fs = require('fs');
            const path = require('path');
            const os = require('os');
            
            const ext = this.getExtensionForLanguage(language);
            const tempFile = path.join(os.tmpdir(), `codeverify_${Date.now()}${ext}`);
            
            try {
                fs.writeFileSync(tempFile, code);
                const findings = await this.analyzeWithCli(tempFile);
                return findings;
            } finally {
                try { fs.unlinkSync(tempFile); } catch {}
            }
        }

        return [];
    }

    /**
     * Analyze a directory
     */
    async analyzeDirectory(dirPath: string): Promise<Finding[]> {
        if (this.config.localAnalysisEnabled) {
            try {
                return await this.analyzeWithCli(dirPath);
            } catch (error) {
                logger.debug('CLI directory analysis failed', error);
            }
        }

        return [];
    }

    /**
     * Get trust score for a file
     */
    async getTrustScore(filePath: string): Promise<TrustScore> {
        // Try API first
        if (this.httpClient) {
            try {
                const fs = require('fs');
                const content = fs.readFileSync(filePath, 'utf8');
                
                const response = await this.httpClient.post('/api/v1/trust-score/analyze', {
                    code: content,
                    file_path: filePath,
                });
                return response.data;
            } catch (error) {
                logger.debug('API trust score failed', error);
            }
        }

        // Try CLI
        if (this.config.localAnalysisEnabled) {
            try {
                const { stdout } = await execAsync(
                    `${this.config.cliPath} trust-score "${filePath}" -f json`,
                    { maxBuffer: 1024 * 1024 }
                );
                return JSON.parse(stdout);
            } catch (error) {
                logger.debug('CLI trust score failed', error);
            }
        }

        // Return default score
        return {
            score: 0,
            ai_probability: 0,
            risk_level: 'unknown',
            complexity_score: 0,
            pattern_score: 0,
            quality_score: 0,
            verification_score: 0,
            factors: {},
        };
    }

    /**
     * Get explanation for a finding
     */
    async explainFinding(diagnostic: vscode.Diagnostic): Promise<any> {
        if (this.httpClient) {
            try {
                const response = await this.httpClient.post('/api/v1/analyses/explain', {
                    message: diagnostic.message,
                    severity: this.mapSeverity(diagnostic.severity),
                    code: diagnostic.code,
                });
                return response.data;
            } catch (error) {
                logger.debug('API explain failed', error);
            }
        }

        // Return basic explanation
        return {
            explanation: diagnostic.message,
            impact: 'This finding may affect code quality or security.',
            fix_suggestion: null,
            references: [],
        };
    }

    /**
     * Debug verification for a file
     */
    async debugVerification(filePath: string): Promise<any> {
        // Try API first
        if (this.httpClient) {
            try {
                const fs = require('fs');
                const content = fs.readFileSync(filePath, 'utf8');
                
                const response = await this.httpClient.post('/api/v1/debugger/trace', {
                    code: content,
                    file_path: filePath,
                });
                return response.data;
            } catch (error) {
                logger.debug('API debug failed', error);
            }
        }

        // Try CLI
        if (this.config.localAnalysisEnabled) {
            try {
                const { stdout } = await execAsync(
                    `${this.config.cliPath} debug "${filePath}" -f json`,
                    { maxBuffer: 10 * 1024 * 1024 }
                );
                return JSON.parse(stdout);
            } catch (error) {
                logger.debug('CLI debug failed', error);
            }
        }

        // Return empty debug info
        return {
            result: 'unknown',
            steps: [],
        };
    }

    /**
     * Analyze using local CLI
     */
    private async analyzeWithCli(path: string): Promise<Finding[]> {
        try {
            const { stdout } = await execAsync(
                `${this.config.cliPath} analyze "${path}" -f json`,
                { maxBuffer: 10 * 1024 * 1024 } // 10MB buffer
            );

            const result = JSON.parse(stdout);
            return result.findings || [];
        } catch (error: any) {
            // Check if CLI is not installed
            if (error.code === 'ENOENT' || error.message?.includes('not found')) {
                throw new Error('CodeVerify CLI not found. Install with: pip install codeverify-cli');
            }
            
            // Try to parse error output as JSON (CLI might return findings in stderr on failure)
            if (error.stdout) {
                try {
                    const result = JSON.parse(error.stdout);
                    return result.findings || [];
                } catch {
                    // Ignore parse errors
                }
            }
            
            throw error;
        }
    }

    /**
     * Analyze using API
     */
    private async analyzeWithApi(filePath: string): Promise<Finding[]> {
        if (!this.httpClient) {
            throw new Error('API client not configured');
        }

        // Read file content
        const fs = require('fs');
        const content = fs.readFileSync(filePath, 'utf8');

        const response = await this.httpClient.post('/api/v1/analyses/inline', {
            file_path: filePath,
            content: content,
        });

        return response.data.findings || [];
    }

    /**
     * Dismiss a finding
     */
    async dismissFinding(findingId: string, reason: string): Promise<void> {
        if (this.httpClient && findingId) {
            await this.httpClient.post(`/api/v1/feedback/dismiss/${findingId}`, {
                reason: reason,
                learn_pattern: true,
            });
        }
    }

    /**
     * Submit feedback on a finding
     */
    async submitFeedback(findingId: string, feedbackType: string, comment?: string): Promise<void> {
        if (this.httpClient && findingId) {
            await this.httpClient.post('/api/v1/feedback', {
                finding_id: findingId,
                feedback_type: feedbackType,
                comment: comment,
            });
        }
    }

    /**
     * AI Pair Review for real-time code analysis
     */
    async pairReview(code: string, context: {
        language: string;
        filePath?: string;
        lineStart?: number;
        surroundingContext?: string;
        unitType?: string;
        unitName?: string;
    }): Promise<any[]> {
        if (this.httpClient) {
            try {
                const response = await this.httpClient.post('/api/v1/pair-review', {
                    code,
                    language: context.language,
                    file_path: context.filePath,
                    line_start: context.lineStart,
                    surrounding_context: context.surroundingContext,
                    unit_type: context.unitType,
                    unit_name: context.unitName,
                });
                return response.data.findings || [];
            } catch (error) {
                logger.debug('API pair review failed', error);
            }
        }

        // Fall back to CLI for local analysis
        if (this.config.localAnalysisEnabled) {
            const fs = require('fs');
            const path = require('path');
            const os = require('os');
            
            const ext = this.getExtensionForLanguage(context.language);
            const tempFile = path.join(os.tmpdir(), `codeverify_pair_${Date.now()}${ext}`);
            
            try {
                fs.writeFileSync(tempFile, code);
                const { stdout } = await execAsync(
                    `${this.config.cliPath} analyze "${tempFile}" -f json --quick`,
                    { maxBuffer: 1024 * 1024 }
                );
                const result = JSON.parse(stdout);
                return result.findings || [];
            } catch (error) {
                logger.debug('CLI pair review failed', error);
            } finally {
                try { fs.unlinkSync(tempFile); } catch {}
            }
        }

        return [];
    }

    /**
     * Quick analysis for paste interception (optimized for speed)
     */
    async analyzeCodeQuick(code: string, language: string): Promise<Finding[]> {
        if (this.httpClient) {
            try {
                const response = await this.httpClient.post('/api/v1/analyses/quick', {
                    content: code,
                    language: language,
                    mode: 'quick',
                }, {
                    timeout: 5000, // 5 second timeout for quick analysis
                });
                return response.data.findings || [];
            } catch (error) {
                logger.debug('Quick API analysis failed', error);
            }
        }

        // Fall back to local pattern-based analysis for speed
        return this.localQuickAnalysis(code, language);
    }

    /**
     * Get trust score for code snippet (for paste interception)
     */
    async getTrustScoreForCode(code: string, language: string): Promise<TrustScore> {
        if (this.httpClient) {
            try {
                const response = await this.httpClient.post('/api/v1/trust-score/quick', {
                    code: code,
                    language: language,
                }, {
                    timeout: 3000, // 3 second timeout
                });
                return response.data;
            } catch (error) {
                logger.debug('Quick trust score API failed', error);
            }
        }

        // Fall back to local trust score calculation
        return this.localTrustScore(code, language);
    }

    /**
     * Local quick analysis using pattern matching
     */
    private localQuickAnalysis(code: string, language: string): Finding[] {
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

    // =========================================================================
    // Formal Spec Assistant Methods
    // =========================================================================

    /**
     * Convert natural language specification to Z3
     */
    async convertNLToZ3(
        specification: string,
        context?: Record<string, any>
    ): Promise<NLToZ3Result> {
        if (this.httpClient) {
            try {
                const response = await this.httpClient.post('/api/v1/specs/nl-to-z3', {
                    specification,
                    context,
                });
                return response.data;
            } catch (error) {
                logger.debug('NL-to-Z3 conversion failed', error);
            }
        }

        // Fallback to local template matching
        return this.localNLToZ3(specification);
    }

    /**
     * Batch convert multiple natural language specs to Z3
     */
    async convertNLToZ3Batch(
        specifications: string[],
        context?: Record<string, any>
    ): Promise<{ results: NLToZ3Result[]; total: number; successful: number }> {
        if (this.httpClient) {
            try {
                const response = await this.httpClient.post('/api/v1/specs/nl-to-z3/batch', {
                    specifications,
                    context,
                });
                return response.data;
            } catch (error) {
                logger.debug('Batch NL-to-Z3 conversion failed', error);
            }
        }

        // Fallback to local
        const results = specifications.map(s => this.localNLToZ3(s));
        return {
            results,
            total: results.length,
            successful: results.filter(r => r.success).length,
        };
    }

    /**
     * Validate a Z3 specification
     */
    async validateZ3Spec(
        z3_expr: string,
        variables: Record<string, string>
    ): Promise<SpecValidationResult> {
        if (this.httpClient) {
            try {
                const response = await this.httpClient.post('/api/v1/specs/nl-to-z3/validate', {
                    z3_expr,
                    variables,
                });
                return response.data;
            } catch (error) {
                logger.debug('Z3 validation failed', error);
            }
        }

        return {
            is_satisfiable: false,
            message: 'Z3 validation requires API connection',
        };
    }

    /**
     * Refine a specification with feedback
     */
    async refineSpec(
        original_spec: string,
        current_z3: string,
        feedback: string
    ): Promise<NLToZ3Result> {
        if (this.httpClient) {
            try {
                const response = await this.httpClient.post('/api/v1/specs/nl-to-z3/refine', {
                    original_spec,
                    current_z3,
                    feedback,
                });
                return response.data;
            } catch (error) {
                logger.debug('Spec refinement failed', error);
            }
        }

        // Return unchanged if no API
        return this.localNLToZ3(original_spec);
    }

    /**
     * Suggest specifications for a function signature
     */
    async suggestSpecs(
        function_signature: string,
        docstring?: string
    ): Promise<{ suggestions: string[]; count: number }> {
        if (this.httpClient) {
            try {
                const response = await this.httpClient.post('/api/v1/specs/nl-to-z3/suggest', {
                    function_signature,
                    docstring,
                });
                return response.data;
            } catch (error) {
                logger.debug('Spec suggestion failed', error);
            }
        }

        // Local suggestion fallback
        return this.localSuggestSpecs(function_signature);
    }

    /**
     * Get the NL-to-Z3 template library
     */
    async getSpecTemplates(): Promise<{ templates: SpecTemplate[]; count: number }> {
        if (this.httpClient) {
            try {
                const response = await this.httpClient.get('/api/v1/specs/nl-to-z3/templates');
                return response.data;
            } catch (error) {
                logger.debug('Get templates failed', error);
            }
        }

        return { templates: this.getLocalTemplates(), count: this.getLocalTemplates().length };
    }

    /**
     * Local NL-to-Z3 conversion using simple pattern matching
     */
    private localNLToZ3(specification: string): NLToZ3Result {
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
    private localSuggestSpecs(signature: string): { suggestions: string[]; count: number } {
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
    private getLocalTemplates(): SpecTemplate[] {
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
    private localTrustScore(code: string, language: string): TrustScore {
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

    private getExtensionForLanguage(language: string): string {
        const extensions: Record<string, string> = {
            'python': '.py',
            'typescript': '.ts',
            'javascript': '.js',
            'typescriptreact': '.tsx',
            'javascriptreact': '.jsx',
            'go': '.go',
            'java': '.java',
            'rust': '.rs',
        };
        return extensions[language] || '.txt';
    }

    private mapSeverity(severity: vscode.DiagnosticSeverity | undefined): string {
        switch (severity) {
            case vscode.DiagnosticSeverity.Error:
                return 'critical';
            case vscode.DiagnosticSeverity.Warning:
                return 'high';
            case vscode.DiagnosticSeverity.Information:
                return 'medium';
            default:
                return 'low';
        }
    }
}

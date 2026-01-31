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

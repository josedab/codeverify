/**
 * Analysis Provider
 * 
 * Coordinates analysis across different sources.
 */

import * as vscode from 'vscode';
import { CodeVerifyClient, Finding } from '../client';

export class AnalysisProvider {
    private client: CodeVerifyClient;
    private analysisCache: Map<string, { findings: Finding[]; timestamp: number }> = new Map();
    private readonly cacheTimeout = 60000; // 1 minute

    constructor(client: CodeVerifyClient) {
        this.client = client;
    }

    /**
     * Analyze a file, using cache if available
     */
    async analyzeFile(filePath: string, forceRefresh: boolean = false): Promise<Finding[]> {
        // Check cache
        if (!forceRefresh) {
            const cached = this.analysisCache.get(filePath);
            if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.findings;
            }
        }

        // Perform analysis
        const findings = await this.client.analyzeFile(filePath);

        // Update cache
        this.analysisCache.set(filePath, {
            findings,
            timestamp: Date.now(),
        });

        return findings;
    }

    /**
     * Clear cache for a file
     */
    clearCache(filePath?: string): void {
        if (filePath) {
            this.analysisCache.delete(filePath);
        } else {
            this.analysisCache.clear();
        }
    }

    /**
     * Get cached findings for a file
     */
    getCachedFindings(filePath: string): Finding[] | undefined {
        return this.analysisCache.get(filePath)?.findings;
    }
}

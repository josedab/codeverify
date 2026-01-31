/**
 * CodeVerify Extension Logger
 *
 * Provides structured logging via VS Code's output channel.
 */

import * as vscode from 'vscode';

let outputChannel: vscode.OutputChannel | undefined;

export function initializeLogger(): vscode.OutputChannel {
    if (!outputChannel) {
        outputChannel = vscode.window.createOutputChannel('CodeVerify');
    }
    return outputChannel;
}

export function getLogger(): vscode.OutputChannel {
    if (!outputChannel) {
        return initializeLogger();
    }
    return outputChannel;
}

function formatMessage(level: string, message: string, data?: unknown): string {
    const timestamp = new Date().toISOString();
    let formatted = `[${timestamp}] [${level}] ${message}`;
    if (data !== undefined) {
        try {
            const dataStr = typeof data === 'object' ? JSON.stringify(data) : String(data);
            formatted += ` ${dataStr}`;
        } catch {
            formatted += ' [unserializable data]';
        }
    }
    return formatted;
}

export const logger = {
    info(message: string, data?: unknown): void {
        getLogger().appendLine(formatMessage('INFO', message, data));
    },

    warn(message: string, data?: unknown): void {
        getLogger().appendLine(formatMessage('WARN', message, data));
    },

    error(message: string, data?: unknown): void {
        getLogger().appendLine(formatMessage('ERROR', message, data));
    },

    debug(message: string, data?: unknown): void {
        const config = vscode.workspace.getConfiguration('codeverify');
        if (config.get('debugLogging', false)) {
            getLogger().appendLine(formatMessage('DEBUG', message, data));
        }
    },

    show(): void {
        getLogger().show();
    },

    dispose(): void {
        outputChannel?.dispose();
        outputChannel = undefined;
    },
};

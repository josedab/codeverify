/**
 * Small generic collection helpers with no `vscode` dependency, extracted
 * so they can be unit tested directly and reused by vscode-dependent
 * providers without duplicating the algorithm.
 */

/**
 * Insert `entry` at the front of `list` (most-recent-first), evicting the
 * oldest entry once `maxSize` is exceeded. Mutates `list` in place, mirroring
 * how bounded history lists are maintained in providers like
 * FormalSpecAssistantProvider.
 */
export function pushBounded<T>(list: T[], entry: T, maxSize: number): void {
    list.unshift(entry);
    if (list.length > maxSize) {
        list.pop();
    }
}

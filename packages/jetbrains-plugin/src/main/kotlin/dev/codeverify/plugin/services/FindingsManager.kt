package dev.codeverify.plugin.services

import com.intellij.codeInsight.daemon.DaemonCodeAnalyzer
import com.intellij.openapi.components.Service
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import java.util.concurrent.ConcurrentHashMap

/**
 * Manages verification findings across the project.
 * Provides centralized access to all findings for UI components.
 */
@Service(Service.Level.PROJECT)
class FindingsManager(private val project: Project) {
    
    private val fileFindings = ConcurrentHashMap<String, List<Finding>>()
    private val listeners = mutableListOf<FindingsListener>()
    
    /**
     * Update findings for a file.
     */
    fun updateFindings(filePath: String, findings: List<Finding>) {
        val oldFindings = fileFindings[filePath] ?: emptyList()
        fileFindings[filePath] = findings
        
        // Notify listeners
        listeners.forEach { listener ->
            listener.onFindingsUpdated(filePath, oldFindings, findings)
        }
        
        // Refresh editor annotations
        DaemonCodeAnalyzer.getInstance(project).restart()
    }
    
    /**
     * Get findings for a specific file.
     */
    fun getFindingsForFile(filePath: String): List<Finding> {
        return fileFindings[filePath] ?: emptyList()
    }
    
    /**
     * Get findings for a specific line in a file.
     */
    fun getFindingsForLine(filePath: String, lineNumber: Int): List<Finding> {
        return getFindingsForFile(filePath).filter { finding ->
            finding.lineStart?.let { start ->
                finding.lineEnd?.let { end ->
                    lineNumber in start..end
                } ?: (lineNumber == start)
            } ?: false
        }
    }
    
    /**
     * Get all findings across the project.
     */
    fun getAllFindings(): Map<String, List<Finding>> {
        return fileFindings.toMap()
    }
    
    /**
     * Get findings by severity.
     */
    fun getFindingsBySeverity(severity: String): List<Pair<String, Finding>> {
        return fileFindings.flatMap { (path, findings) ->
            findings.filter { it.severity.equals(severity, ignoreCase = true) }
                .map { path to it }
        }
    }
    
    /**
     * Get total findings count.
     */
    fun getTotalFindingsCount(): Int {
        return fileFindings.values.sumOf { it.size }
    }
    
    /**
     * Get findings count by severity.
     */
    fun getFindingsCountBySeverity(): Map<String, Int> {
        val counts = mutableMapOf<String, Int>()
        fileFindings.values.flatten().forEach { finding ->
            counts[finding.severity] = (counts[finding.severity] ?: 0) + 1
        }
        return counts
    }
    
    /**
     * Clear findings for a file.
     */
    fun clearFindings(filePath: String) {
        fileFindings.remove(filePath)
        DaemonCodeAnalyzer.getInstance(project).restart()
    }
    
    /**
     * Clear all findings.
     */
    fun clearAllFindings() {
        fileFindings.clear()
        DaemonCodeAnalyzer.getInstance(project).restart()
    }
    
    /**
     * Add a listener for findings updates.
     */
    fun addListener(listener: FindingsListener) {
        listeners.add(listener)
    }
    
    /**
     * Remove a listener.
     */
    fun removeListener(listener: FindingsListener) {
        listeners.remove(listener)
    }
    
    companion object {
        fun getInstance(project: Project): FindingsManager {
            return project.getService(FindingsManager::class.java)
        }
    }
}

/**
 * Listener for findings changes.
 */
interface FindingsListener {
    fun onFindingsUpdated(filePath: String, oldFindings: List<Finding>, newFindings: List<Finding>)
}

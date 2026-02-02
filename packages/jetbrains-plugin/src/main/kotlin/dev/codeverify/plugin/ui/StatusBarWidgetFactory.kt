package dev.codeverify.plugin.ui

import com.intellij.openapi.project.Project
import com.intellij.openapi.util.Disposer
import com.intellij.openapi.wm.StatusBar
import com.intellij.openapi.wm.StatusBarWidget
import com.intellij.openapi.wm.StatusBarWidgetFactory
import com.intellij.util.Consumer
import dev.codeverify.plugin.services.Finding
import dev.codeverify.plugin.services.FindingsListener
import dev.codeverify.plugin.services.FindingsManager
import java.awt.event.MouseEvent
import javax.swing.Icon

/**
 * Factory for CodeVerify status bar widget.
 */
class StatusBarWidgetFactory : StatusBarWidgetFactory {
    
    override fun getId(): String = "CodeVerifyStatus"
    
    override fun getDisplayName(): String = "CodeVerify Status"
    
    override fun isAvailable(project: Project): Boolean = true
    
    override fun createWidget(project: Project): StatusBarWidget {
        return CodeVerifyStatusWidget(project)
    }
    
    override fun disposeWidget(widget: StatusBarWidget) {
        Disposer.dispose(widget)
    }
    
    override fun canBeEnabledOn(statusBar: StatusBar): Boolean = true
}

/**
 * Status bar widget showing CodeVerify status.
 */
class CodeVerifyStatusWidget(private val project: Project) : 
    StatusBarWidget, 
    StatusBarWidget.TextPresentation,
    FindingsListener {
    
    private var statusBar: StatusBar? = null
    private var currentText = "CodeVerify: Ready"
    
    init {
        FindingsManager.getInstance(project).addListener(this)
    }
    
    override fun ID(): String = "CodeVerifyStatus"
    
    override fun getPresentation(): StatusBarWidget.WidgetPresentation = this
    
    override fun install(statusBar: StatusBar) {
        this.statusBar = statusBar
        updateWidget()
    }
    
    override fun dispose() {
        FindingsManager.getInstance(project).removeListener(this)
    }
    
    // TextPresentation implementation
    
    override fun getText(): String = currentText
    
    override fun getAlignment(): Float = 0f
    
    override fun getTooltipText(): String {
        val manager = FindingsManager.getInstance(project)
        val total = manager.getTotalFindingsCount()
        val counts = manager.getFindingsCountBySeverity()
        
        return buildString {
            append("CodeVerify Findings: $total\n")
            counts.entries.sortedByDescending { 
                when (it.key) {
                    "critical" -> 4
                    "high" -> 3
                    "medium" -> 2
                    "low" -> 1
                    else -> 0
                }
            }.forEach { (severity, count) ->
                append("$severity: $count\n")
            }
        }
    }
    
    override fun getClickConsumer(): Consumer<MouseEvent>? {
        return Consumer {
            // Open tool window on click
            com.intellij.openapi.wm.ToolWindowManager.getInstance(project)
                .getToolWindow("CodeVerify")
                ?.show()
        }
    }
    
    // FindingsListener implementation
    
    override fun onFindingsUpdated(filePath: String, oldFindings: List<Finding>, newFindings: List<Finding>) {
        updateWidget()
    }
    
    private fun updateWidget() {
        val manager = FindingsManager.getInstance(project)
        val total = manager.getTotalFindingsCount()
        val counts = manager.getFindingsCountBySeverity()
        
        currentText = when {
            total == 0 -> "✓ CodeVerify"
            counts.containsKey("critical") || counts.containsKey("high") -> 
                "⚠ CodeVerify: $total issues"
            else -> "CodeVerify: $total"
        }
        
        statusBar?.updateWidget(ID())
    }
}

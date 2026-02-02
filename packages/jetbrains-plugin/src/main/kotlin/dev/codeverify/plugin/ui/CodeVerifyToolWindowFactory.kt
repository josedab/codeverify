package dev.codeverify.plugin.ui

import com.intellij.openapi.project.DumbAware
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.ui.components.JBLabel
import com.intellij.ui.components.JBScrollPane
import com.intellij.ui.content.ContentFactory
import com.intellij.ui.table.JBTable
import dev.codeverify.plugin.services.Finding
import dev.codeverify.plugin.services.FindingsListener
import dev.codeverify.plugin.services.FindingsManager
import java.awt.BorderLayout
import java.awt.Component
import javax.swing.*
import javax.swing.table.AbstractTableModel
import javax.swing.table.DefaultTableCellRenderer

/**
 * Tool window factory for CodeVerify.
 */
class CodeVerifyToolWindowFactory : ToolWindowFactory, DumbAware {
    
    override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) {
        val contentFactory = ContentFactory.getInstance()
        
        // Findings panel
        val findingsPanel = FindingsPanel(project)
        val findingsContent = contentFactory.createContent(findingsPanel, "Findings", false)
        toolWindow.contentManager.addContent(findingsContent)
        
        // Summary panel
        val summaryPanel = SummaryPanel(project)
        val summaryContent = contentFactory.createContent(summaryPanel, "Summary", false)
        toolWindow.contentManager.addContent(summaryContent)
    }
}

/**
 * Panel showing all findings.
 */
class FindingsPanel(private val project: Project) : JPanel(BorderLayout()), FindingsListener {
    
    private val tableModel = FindingsTableModel()
    private val table = JBTable(tableModel)
    
    init {
        // Configure table
        table.fillsViewportHeight = true
        table.setShowGrid(false)
        table.rowHeight = 24
        
        // Custom renderer for severity column
        table.columnModel.getColumn(0).cellRenderer = SeverityRenderer()
        
        // Set column widths
        table.columnModel.getColumn(0).preferredWidth = 80
        table.columnModel.getColumn(1).preferredWidth = 200
        table.columnModel.getColumn(2).preferredWidth = 100
        table.columnModel.getColumn(3).preferredWidth = 150
        
        // Double-click to navigate
        table.addMouseListener(object : java.awt.event.MouseAdapter() {
            override fun mouseClicked(e: java.awt.event.MouseEvent) {
                if (e.clickCount == 2) {
                    val row = table.selectedRow
                    if (row >= 0) {
                        navigateToFinding(row)
                    }
                }
            }
        })
        
        add(JBScrollPane(table), BorderLayout.CENTER)
        
        // Toolbar
        val toolbar = JPanel()
        toolbar.add(JButton("Refresh").apply {
            addActionListener { refreshFindings() }
        })
        toolbar.add(JButton("Clear All").apply {
            addActionListener { clearFindings() }
        })
        add(toolbar, BorderLayout.NORTH)
        
        // Register listener
        FindingsManager.getInstance(project).addListener(this)
        
        // Initial load
        refreshFindings()
    }
    
    override fun onFindingsUpdated(filePath: String, oldFindings: List<Finding>, newFindings: List<Finding>) {
        SwingUtilities.invokeLater {
            refreshFindings()
        }
    }
    
    private fun refreshFindings() {
        val findingsManager = FindingsManager.getInstance(project)
        val allFindings = findingsManager.getAllFindings()
            .flatMap { (path, findings) -> findings.map { path to it } }
            .sortedByDescending { (_, finding) -> finding.severityLevel }
        
        tableModel.setFindings(allFindings)
    }
    
    private fun clearFindings() {
        FindingsManager.getInstance(project).clearAllFindings()
        refreshFindings()
    }
    
    private fun navigateToFinding(row: Int) {
        val (filePath, finding) = tableModel.getFindingAt(row) ?: return
        
        val file = com.intellij.openapi.vfs.LocalFileSystem.getInstance()
            .findFileByPath(filePath) ?: return
        
        com.intellij.openapi.fileEditor.OpenFileDescriptor(
            project,
            file,
            (finding.lineStart ?: 1) - 1,
            0
        ).navigate(true)
    }
}

/**
 * Table model for findings.
 */
class FindingsTableModel : AbstractTableModel() {
    
    private val columns = arrayOf("Severity", "Title", "Category", "File", "Line")
    private var findings = listOf<Pair<String, Finding>>()
    
    fun setFindings(newFindings: List<Pair<String, Finding>>) {
        findings = newFindings
        fireTableDataChanged()
    }
    
    fun getFindingAt(row: Int): Pair<String, Finding>? {
        return findings.getOrNull(row)
    }
    
    override fun getRowCount(): Int = findings.size
    override fun getColumnCount(): Int = columns.size
    override fun getColumnName(column: Int): String = columns[column]
    
    override fun getValueAt(rowIndex: Int, columnIndex: Int): Any {
        val (path, finding) = findings[rowIndex]
        return when (columnIndex) {
            0 -> finding.severity
            1 -> finding.title
            2 -> finding.category
            3 -> path.substringAfterLast("/")
            4 -> finding.lineStart?.toString() ?: "-"
            else -> ""
        }
    }
}

/**
 * Custom renderer for severity column.
 */
class SeverityRenderer : DefaultTableCellRenderer() {
    
    override fun getTableCellRendererComponent(
        table: JTable?,
        value: Any?,
        isSelected: Boolean,
        hasFocus: Boolean,
        row: Int,
        column: Int
    ): Component {
        val component = super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column)
        
        if (component is JLabel) {
            val severity = value?.toString()?.lowercase() ?: ""
            component.foreground = when (severity) {
                "critical" -> java.awt.Color(220, 38, 38) // Red
                "high" -> java.awt.Color(234, 88, 12) // Orange
                "medium" -> java.awt.Color(202, 138, 4) // Yellow
                "low" -> java.awt.Color(34, 197, 94) // Green
                else -> table?.foreground ?: java.awt.Color.BLACK
            }
        }
        
        return component
    }
}

/**
 * Summary panel showing statistics.
 */
class SummaryPanel(private val project: Project) : JPanel(BorderLayout()), FindingsListener {
    
    private val statsLabel = JBLabel()
    
    init {
        add(JBScrollPane(statsLabel), BorderLayout.CENTER)
        
        FindingsManager.getInstance(project).addListener(this)
        refreshStats()
    }
    
    override fun onFindingsUpdated(filePath: String, oldFindings: List<Finding>, newFindings: List<Finding>) {
        SwingUtilities.invokeLater {
            refreshStats()
        }
    }
    
    private fun refreshStats() {
        val findingsManager = FindingsManager.getInstance(project)
        val counts = findingsManager.getFindingsCountBySeverity()
        val total = findingsManager.getTotalFindingsCount()
        
        val html = buildString {
            append("<html><body style='padding: 10px;'>")
            append("<h2>Verification Summary</h2>")
            append("<p>Total Findings: <b>$total</b></p>")
            append("<hr/>")
            append("<h3>By Severity:</h3>")
            append("<ul>")
            
            listOf("critical", "high", "medium", "low").forEach { severity ->
                val count = counts[severity] ?: 0
                val color = when (severity) {
                    "critical" -> "#dc2626"
                    "high" -> "#ea580c"
                    "medium" -> "#ca8a04"
                    "low" -> "#22c55e"
                    else -> "#000000"
                }
                append("<li><span style='color: $color;'>${severity.capitalize()}: <b>$count</b></span></li>")
            }
            
            append("</ul>")
            append("</body></html>")
        }
        
        statsLabel.text = html
    }
}

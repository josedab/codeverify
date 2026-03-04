package dev.codeverify.plugin.ui

import com.intellij.codeInsight.daemon.GutterIconNavigationHandler
import com.intellij.codeInsight.daemon.LineMarkerInfo
import com.intellij.codeInsight.daemon.LineMarkerProvider
import com.intellij.openapi.components.service
import com.intellij.openapi.editor.markup.GutterIconRenderer
import com.intellij.openapi.ui.popup.JBPopupFactory
import com.intellij.psi.PsiElement
import com.intellij.psi.PsiFile
import com.intellij.ui.awt.RelativePoint
import dev.codeverify.plugin.services.FindingsManager
import java.awt.event.MouseEvent
import javax.swing.Icon
import javax.swing.ImageIcon

/**
 * Provides gutter icons (verification annotations) next to lines that have findings.
 *
 * - ✅ Green shield: line verified, no issues
 * - ⚠️ Yellow warning: low/medium severity finding
 * - ❌ Red error: high/critical severity finding
 * - 🔧 Wrench: auto-fix available
 */
class CodeVerifyGutterIconProvider : LineMarkerProvider {

    override fun getLineMarkerInfo(element: PsiElement): LineMarkerInfo<*>? {
        // Only process the first token of each line to avoid duplicates
        if (element.parent !is PsiFile && element.prevSibling != null) return null

        val project = element.project
        val findingsManager = project.service<FindingsManager>()
        val file = element.containingFile?.virtualFile ?: return null
        val document = element.containingFile?.viewProvider?.document ?: return null

        val lineNumber = document.getLineNumber(element.textOffset) + 1
        val findings = findingsManager.getFindingsForLine(file.path, lineNumber)

        if (findings.isEmpty()) return null

        val maxSeverity = findings.maxByOrNull { severityOrder(it.severity) }?.severity ?: "info"
        val hasAutofix = findings.any { !it.fixSuggestion.isNullOrBlank() }
        val icon = getIconForSeverity(maxSeverity, hasAutofix)
        val tooltip = buildTooltipHtml(findings, lineNumber)

        return LineMarkerInfo(
            element,
            element.textRange,
            icon,
            { tooltip },
            GutterClickHandler(findings),
            GutterIconRenderer.Alignment.LEFT,
            { tooltip }
        )
    }

    private fun severityOrder(severity: String): Int = when (severity.lowercase()) {
        "critical" -> 4
        "high" -> 3
        "medium" -> 2
        "low" -> 1
        else -> 0
    }

    private fun getIconForSeverity(severity: String, hasAutofix: Boolean): Icon {
        // Use colored circle icons as fallback (production would use SVG icons from /icons/)
        val color = when (severity.lowercase()) {
            "critical", "high" -> java.awt.Color.RED
            "medium" -> java.awt.Color.ORANGE
            "low" -> java.awt.Color.YELLOW
            else -> java.awt.Color.GREEN
        }
        return createCircleIcon(color, if (hasAutofix) "🔧" else "●")
    }

    private fun createCircleIcon(color: java.awt.Color, text: String): Icon {
        return object : Icon {
            override fun paintIcon(c: java.awt.Component?, g: java.awt.Graphics, x: Int, y: Int) {
                g.color = color
                g.fillOval(x + 2, y + 2, 12, 12)
                g.color = java.awt.Color.WHITE
                g.font = g.font.deriveFont(8f)
                if (text == "🔧") {
                    g.drawString("F", x + 4, y + 12)
                }
            }
            override fun getIconWidth() = 16
            override fun getIconHeight() = 16
        }
    }

    private fun buildTooltipHtml(findings: List<FindingsManager.Finding>, line: Int): String {
        val sb = StringBuilder("<html><body>")
        sb.append("<b>CodeVerify — Line $line</b><br/>")
        sb.append("<table>")
        for (f in findings) {
            val color = when (f.severity.lowercase()) {
                "critical", "high" -> "#dc2626"
                "medium" -> "#ea580c"
                "low" -> "#ca8a04"
                else -> "#16a34a"
            }
            sb.append("<tr>")
            sb.append("<td><font color='$color'><b>[${f.severity.uppercase()}]</b></font></td>")
            sb.append("<td>${f.title}</td>")
            sb.append("</tr>")
            if (!f.fixSuggestion.isNullOrBlank()) {
                sb.append("<tr><td></td><td><i>🔧 Fix available</i></td></tr>")
            }
        }
        sb.append("</table></body></html>")
        return sb.toString()
    }

    /**
     * Handles clicks on gutter icons — shows a popup with finding details and quick-fix options.
     */
    private class GutterClickHandler(
        private val findings: List<FindingsManager.Finding>
    ) : GutterIconNavigationHandler<PsiElement> {
        override fun navigate(e: MouseEvent, elt: PsiElement) {
            val content = buildString {
                for (f in findings) {
                    append("${f.severity.uppercase()}: ${f.title}\n")
                    append("  ${f.description}\n")
                    if (!f.fixSuggestion.isNullOrBlank()) {
                        append("  Fix: ${f.fixSuggestion}\n")
                    }
                    append("\n")
                }
            }
            JBPopupFactory.getInstance()
                .createMessage(content.trim())
                .show(RelativePoint(e))
        }
    }
}

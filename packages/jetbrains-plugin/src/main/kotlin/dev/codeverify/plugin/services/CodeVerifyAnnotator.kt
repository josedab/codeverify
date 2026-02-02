package dev.codeverify.plugin.services

import com.intellij.lang.annotation.AnnotationHolder
import com.intellij.lang.annotation.ExternalAnnotator
import com.intellij.lang.annotation.HighlightSeverity
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiFile
import dev.codeverify.plugin.settings.CodeVerifySettings
import kotlinx.coroutines.runBlocking

/**
 * External annotator that shows CodeVerify findings inline in the editor.
 * Runs asynchronously to avoid blocking the UI.
 */
class CodeVerifyAnnotator : ExternalAnnotator<CodeVerifyAnnotator.InitialInfo, CodeVerifyAnnotator.AnnotationResult>() {
    
    private val logger = Logger.getInstance(CodeVerifyAnnotator::class.java)
    
    /**
     * Collect initial information before analysis.
     */
    override fun collectInformation(file: PsiFile, editor: Editor, hasErrors: Boolean): InitialInfo? {
        val settings = CodeVerifySettings.getInstance()
        
        // Skip if disabled or no API key
        if (!settings.enabled || settings.apiKey.isNullOrBlank()) {
            return null
        }
        
        // Skip unsupported file types
        val language = getLanguage(file) ?: return null
        
        return InitialInfo(
            code = file.text,
            language = language,
            filePath = file.virtualFile?.path ?: "",
            project = file.project
        )
    }
    
    /**
     * Perform the actual analysis.
     */
    override fun doAnnotate(info: InitialInfo): AnnotationResult? {
        if (info.code.isBlank()) {
            return null
        }
        
        return try {
            val service = VerificationService.getInstance(info.project)
            val result = runBlocking {
                service.verifyCode(info.code, info.language, info.filePath)
            }
            
            if (result.success) {
                // Update findings manager
                val findingsManager = FindingsManager.getInstance(info.project)
                findingsManager.updateFindings(info.filePath, result.findings)
                
                AnnotationResult(
                    findings = result.findings,
                    trustScore = result.trustScore,
                    filePath = info.filePath
                )
            } else {
                logger.warn("Verification failed: ${result.error}")
                null
            }
        } catch (e: Exception) {
            logger.error("Annotation failed", e)
            null
        }
    }
    
    /**
     * Apply annotations to the editor.
     */
    override fun apply(file: PsiFile, result: AnnotationResult?, holder: AnnotationHolder) {
        result ?: return
        
        val document = file.viewProvider.document ?: return
        
        for (finding in result.findings) {
            val lineStart = finding.lineStart ?: continue
            val lineEnd = finding.lineEnd ?: lineStart
            
            // Calculate text range
            val startOffset = try {
                document.getLineStartOffset(lineStart - 1)
            } catch (e: Exception) {
                continue
            }
            
            val endOffset = try {
                document.getLineEndOffset(lineEnd - 1)
            } catch (e: Exception) {
                document.getLineEndOffset(lineStart - 1)
            }
            
            val textRange = TextRange(startOffset, endOffset)
            
            // Map severity
            val severity = when (finding.severity.lowercase()) {
                "critical", "high" -> HighlightSeverity.ERROR
                "medium" -> HighlightSeverity.WARNING
                "low" -> HighlightSeverity.WEAK_WARNING
                else -> HighlightSeverity.INFORMATION
            }
            
            // Create annotation
            val message = buildString {
                append("[CodeVerify] ")
                append(finding.title)
                if (finding.proof != null) {
                    append(" (Proven)")
                }
            }
            
            val annotation = holder.newAnnotation(severity, message)
                .range(textRange)
                .tooltip(buildTooltip(finding))
            
            // Add quick fix if available
            if (finding.fixSuggestion != null) {
                annotation.withFix(ApplyFixIntention(finding))
            }
            
            annotation.create()
        }
    }
    
    /**
     * Build tooltip HTML for a finding.
     */
    private fun buildTooltip(finding: Finding): String {
        return buildString {
            append("<html><body>")
            append("<b>${escapeHtml(finding.title)}</b><br/>")
            append("<i>Category:</i> ${escapeHtml(finding.category)}<br/>")
            append("<i>Severity:</i> ${escapeHtml(finding.severity)}<br/>")
            append("<br/>")
            append(escapeHtml(finding.description))
            
            if (finding.proof != null) {
                append("<br/><br/>")
                append("<b>Proof:</b><br/>")
                append("<code>${escapeHtml(finding.proof)}</code>")
            }
            
            if (finding.fixSuggestion != null) {
                append("<br/><br/>")
                append("<b>Suggested Fix Available</b>")
            }
            
            append("</body></html>")
        }
    }
    
    /**
     * Get language identifier for a file.
     */
    private fun getLanguage(file: PsiFile): String? {
        val extension = file.virtualFile?.extension?.lowercase()
        return when (extension) {
            "py" -> "python"
            "ts" -> "typescript"
            "tsx" -> "typescript"
            "js" -> "javascript"
            "jsx" -> "javascript"
            "java" -> "java"
            "kt" -> "kotlin"
            "go" -> "go"
            "rs" -> "rust"
            else -> null
        }
    }
    
    private fun escapeHtml(text: String): String {
        return text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;")
    }
    
    // Data classes
    
    data class InitialInfo(
        val code: String,
        val language: String,
        val filePath: String,
        val project: com.intellij.openapi.project.Project
    )
    
    data class AnnotationResult(
        val findings: List<Finding>,
        val trustScore: Double,
        val filePath: String
    )
}

/**
 * Quick fix intention to apply a suggested fix.
 */
class ApplyFixIntention(private val finding: Finding) : com.intellij.codeInsight.intention.IntentionAction {
    
    override fun getText(): String = "Apply CodeVerify Fix: ${finding.title}"
    
    override fun getFamilyName(): String = "CodeVerify Fixes"
    
    override fun isAvailable(
        project: com.intellij.openapi.project.Project,
        editor: Editor?,
        file: PsiFile?
    ): Boolean = finding.fixSuggestion != null && editor != null
    
    override fun invoke(
        project: com.intellij.openapi.project.Project,
        editor: Editor?,
        file: PsiFile?
    ) {
        editor ?: return
        val service = VerificationService.getInstance(project)
        service.applyFix(finding, editor)
    }
    
    override fun startInWriteAction(): Boolean = false
}

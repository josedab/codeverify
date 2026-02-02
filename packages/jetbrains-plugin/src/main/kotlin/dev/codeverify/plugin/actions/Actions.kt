package dev.codeverify.plugin.actions

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.progress.ProgressIndicator
import com.intellij.openapi.progress.ProgressManager
import com.intellij.openapi.progress.Task
import com.intellij.openapi.ui.Messages
import dev.codeverify.plugin.services.FindingsManager
import dev.codeverify.plugin.services.VerificationService
import kotlinx.coroutines.runBlocking

/**
 * Action to verify the current file.
 */
class VerifyFileAction : AnAction() {
    
    private val logger = Logger.getInstance(VerifyFileAction::class.java)
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        val file = e.getData(CommonDataKeys.VIRTUAL_FILE) ?: return
        val editor = e.getData(CommonDataKeys.EDITOR) ?: return
        
        val language = getLanguage(file.extension) ?: run {
            Messages.showInfoMessage(
                project,
                "Unsupported file type: ${file.extension}",
                "CodeVerify"
            )
            return
        }
        
        val code = editor.document.text
        val filePath = file.path
        
        ProgressManager.getInstance().run(object : Task.Backgroundable(project, "Verifying code...", true) {
            override fun run(indicator: ProgressIndicator) {
                indicator.isIndeterminate = true
                
                val service = VerificationService.getInstance(project)
                val result = runBlocking {
                    service.verifyCode(code, language, filePath)
                }
                
                if (result.success) {
                    val findingsManager = FindingsManager.getInstance(project)
                    findingsManager.updateFindings(filePath, result.findings)
                    
                    // Show summary
                    val message = buildString {
                        append("Verification complete!\n\n")
                        append("Trust Score: ${(result.trustScore * 100).toInt()}%\n")
                        append("Findings: ${result.findings.size}\n")
                        
                        if (result.findings.isNotEmpty()) {
                            append("\nBy severity:\n")
                            result.findings.groupBy { it.severity }.forEach { (severity, findings) ->
                                append("  $severity: ${findings.size}\n")
                            }
                        }
                    }
                    
                    Messages.showInfoMessage(project, message, "CodeVerify Results")
                } else {
                    Messages.showErrorDialog(
                        project,
                        result.error ?: "Unknown error",
                        "CodeVerify Error"
                    )
                }
            }
        })
    }
    
    override fun update(e: AnActionEvent) {
        val file = e.getData(CommonDataKeys.VIRTUAL_FILE)
        e.presentation.isEnabled = file != null && getLanguage(file.extension) != null
    }
    
    private fun getLanguage(extension: String?): String? {
        return when (extension?.lowercase()) {
            "py" -> "python"
            "ts", "tsx" -> "typescript"
            "js", "jsx" -> "javascript"
            "java" -> "java"
            "kt" -> "kotlin"
            "go" -> "go"
            "rs" -> "rust"
            else -> null
        }
    }
}

/**
 * Action to verify selected code.
 */
class VerifySelectionAction : AnAction() {
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        val editor = e.getData(CommonDataKeys.EDITOR) ?: return
        val file = e.getData(CommonDataKeys.VIRTUAL_FILE)
        
        val selection = editor.selectionModel.selectedText
        if (selection.isNullOrBlank()) {
            Messages.showInfoMessage(project, "No code selected", "CodeVerify")
            return
        }
        
        val language = when (file?.extension?.lowercase()) {
            "py" -> "python"
            "ts", "tsx" -> "typescript"
            "js", "jsx" -> "javascript"
            "java" -> "java"
            "kt" -> "kotlin"
            else -> "python"
        }
        
        ProgressManager.getInstance().run(object : Task.Backgroundable(project, "Verifying selection...", true) {
            override fun run(indicator: ProgressIndicator) {
                val service = VerificationService.getInstance(project)
                val result = runBlocking {
                    service.verifyCode(selection, language)
                }
                
                if (result.success) {
                    val message = buildString {
                        append("Selection verified!\n\n")
                        append("Trust Score: ${(result.trustScore * 100).toInt()}%\n")
                        append("Findings: ${result.findings.size}\n")
                        
                        result.findings.forEach { finding ->
                            append("\n• [${finding.severity}] ${finding.title}")
                        }
                    }
                    Messages.showInfoMessage(project, message, "CodeVerify Results")
                } else {
                    Messages.showErrorDialog(project, result.error ?: "Unknown error", "CodeVerify Error")
                }
            }
        })
    }
    
    override fun update(e: AnActionEvent) {
        val editor = e.getData(CommonDataKeys.EDITOR)
        e.presentation.isEnabled = editor?.selectionModel?.hasSelection() == true
    }
}

/**
 * Action to verify the entire project.
 */
class VerifyProjectAction : AnAction() {
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        
        Messages.showInfoMessage(
            project,
            "Project verification is available in CodeVerify Pro.\n\nVisit https://codeverify.dev to upgrade.",
            "CodeVerify Pro Feature"
        )
    }
}

/**
 * Quick verify action with keyboard shortcut.
 */
class QuickVerifyAction : AnAction() {
    
    override fun actionPerformed(e: AnActionEvent) {
        val verifyFileAction = VerifyFileAction()
        verifyFileAction.actionPerformed(e)
    }
    
    override fun update(e: AnActionEvent) {
        val file = e.getData(CommonDataKeys.VIRTUAL_FILE)
        e.presentation.isEnabled = file != null
    }
}

/**
 * Action to apply a fix.
 */
class ApplyFixAction : AnAction() {
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        val editor = e.getData(CommonDataKeys.EDITOR) ?: return
        val file = e.getData(CommonDataKeys.VIRTUAL_FILE) ?: return
        
        val findingsManager = FindingsManager.getInstance(project)
        val caretLine = editor.caretModel.logicalPosition.line + 1
        val findings = findingsManager.getFindingsForLine(file.path, caretLine)
        
        val fixableFindings = findings.filter { it.fixSuggestion != null }
        
        when {
            fixableFindings.isEmpty() -> {
                Messages.showInfoMessage(project, "No fix available for this line", "CodeVerify")
            }
            fixableFindings.size == 1 -> {
                val service = VerificationService.getInstance(project)
                service.applyFix(fixableFindings.first(), editor)
                Messages.showInfoMessage(project, "Fix applied!", "CodeVerify")
            }
            else -> {
                val titles = fixableFindings.map { it.title }.toTypedArray()
                val selected = Messages.showChooseDialog(
                    project,
                    "Multiple fixes available. Choose one:",
                    "Apply Fix",
                    null,
                    titles,
                    titles[0]
                )
                if (selected >= 0) {
                    val service = VerificationService.getInstance(project)
                    service.applyFix(fixableFindings[selected], editor)
                    Messages.showInfoMessage(project, "Fix applied!", "CodeVerify")
                }
            }
        }
    }
}

/**
 * Action to show verification proof.
 */
class ShowProofAction : AnAction() {
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        val editor = e.getData(CommonDataKeys.EDITOR) ?: return
        val file = e.getData(CommonDataKeys.VIRTUAL_FILE) ?: return
        
        val findingsManager = FindingsManager.getInstance(project)
        val caretLine = editor.caretModel.logicalPosition.line + 1
        val findings = findingsManager.getFindingsForLine(file.path, caretLine)
        
        val findingsWithProof = findings.filter { it.proof != null }
        
        if (findingsWithProof.isEmpty()) {
            Messages.showInfoMessage(project, "No proof available for this line", "CodeVerify")
            return
        }
        
        val proofText = buildString {
            findingsWithProof.forEach { finding ->
                append("=== ${finding.title} ===\n\n")
                append(finding.proof)
                append("\n\n")
            }
        }
        
        Messages.showInfoMessage(project, proofText, "Verification Proof")
    }
}

/**
 * Action to view coverage dashboard.
 */
class ViewDashboardAction : AnAction() {
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        
        // Open tool window
        val toolWindow = com.intellij.openapi.wm.ToolWindowManager.getInstance(project)
            .getToolWindow("CodeVerify")
        
        toolWindow?.show()
    }
}

/**
 * Action to open settings.
 */
class OpenSettingsAction : AnAction() {
    
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        
        com.intellij.openapi.options.ShowSettingsUtil.getInstance()
            .showSettingsDialog(project, "dev.codeverify.plugin.settings.CodeVerifyConfigurable")
    }
}

package dev.codeverify.plugin.listeners

import com.intellij.openapi.components.service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.fileEditor.FileDocumentManager
import com.intellij.openapi.fileEditor.FileDocumentManagerListener
import com.intellij.openapi.editor.Document
import com.intellij.openapi.project.ProjectManager
import dev.codeverify.plugin.lsp.LspClient
import dev.codeverify.plugin.services.VerificationService
import dev.codeverify.plugin.settings.CodeVerifySettings

/**
 * Listens for file save events and triggers background Z3 verification.
 * Only runs when "Verify on Save" is enabled in settings.
 */
class SaveVerificationListener : FileDocumentManagerListener {

    private val log = Logger.getInstance(SaveVerificationListener::class.java)

    override fun beforeDocumentSaving(document: Document) {
        val settings = CodeVerifySettings.instance
        if (!settings.enabled || !settings.verifyOnSave) return

        val file = FileDocumentManager.getInstance().getFile(document) ?: return
        val extension = file.extension?.lowercase() ?: return

        val supportedExtensions = setOf("py", "ts", "tsx", "js", "jsx", "java", "kt", "go", "rs")
        if (extension !in supportedExtensions) return

        // Trigger verification in all open projects that contain this file
        for (project in ProjectManager.getInstance().openProjects) {
            if (project.isDisposed) continue

            try {
                // Notify LSP server of save
                val lspClient = project.service<LspClient>()
                if (lspClient.isConnected) {
                    lspClient.didSave("file://${file.path}")
                }

                // Trigger background verification via the API
                val verificationService = project.service<VerificationService>()
                verificationService.verifyFileAsync(file.path, document.text)

                log.info("Background verification triggered on save: ${file.name}")
            } catch (e: Exception) {
                log.warn("Failed to trigger verification on save: ${e.message}")
            }
        }
    }
}

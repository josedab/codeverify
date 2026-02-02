package dev.codeverify.plugin.settings

import com.intellij.openapi.options.Configurable
import com.intellij.ui.components.JBCheckBox
import com.intellij.ui.components.JBLabel
import com.intellij.ui.components.JBPasswordField
import com.intellij.ui.components.JBTextField
import com.intellij.ui.dsl.builder.*
import javax.swing.JComponent
import javax.swing.JPanel

/**
 * Settings UI for CodeVerify plugin.
 */
class CodeVerifyConfigurable : Configurable {
    
    private var panel: JPanel? = null
    
    // UI Components
    private var enabledCheckbox: JBCheckBox? = null
    private var apiKeyField: JBPasswordField? = null
    private var apiUrlField: JBTextField? = null
    private var includeProofsCheckbox: JBCheckBox? = null
    private var includeFixesCheckbox: JBCheckBox? = null
    private var verifyOnSaveCheckbox: JBCheckBox? = null
    private var verifyOnTypeCheckbox: JBCheckBox? = null
    private var offlineModeCheckbox: JBCheckBox? = null
    private var ollamaUrlField: JBTextField? = null
    private var ollamaModelField: JBTextField? = null
    
    override fun getDisplayName(): String = "CodeVerify"
    
    override fun createComponent(): JComponent {
        val settings = CodeVerifySettings.getInstance()
        
        panel = panel {
            group("General") {
                row {
                    enabledCheckbox = checkBox("Enable CodeVerify")
                        .component
                    enabledCheckbox?.isSelected = settings.enabled
                }
                
                row("API Key:") {
                    apiKeyField = cell(JBPasswordField())
                        .columns(COLUMNS_MEDIUM)
                        .comment("Get your API key from https://codeverify.dev/settings")
                        .component
                    apiKeyField?.text = settings.apiKey ?: ""
                }
                
                row("API URL:") {
                    apiUrlField = textField()
                        .columns(COLUMNS_MEDIUM)
                        .component
                    apiUrlField?.text = settings.apiUrl
                }
            }
            
            group("Verification Options") {
                row {
                    includeProofsCheckbox = checkBox("Include formal proofs in results")
                        .component
                    includeProofsCheckbox?.isSelected = settings.includeProofs
                }
                
                row {
                    includeFixesCheckbox = checkBox("Include fix suggestions")
                        .component
                    includeFixesCheckbox?.isSelected = settings.includeFixes
                }
                
                row {
                    verifyOnSaveCheckbox = checkBox("Verify on file save")
                        .component
                    verifyOnSaveCheckbox?.isSelected = settings.verifyOnSave
                }
                
                row {
                    verifyOnTypeCheckbox = checkBox("Verify while typing (may impact performance)")
                        .component
                    verifyOnTypeCheckbox?.isSelected = settings.verifyOnType
                }
            }
            
            group("Offline Mode") {
                row {
                    offlineModeCheckbox = checkBox("Enable offline mode (uses local Ollama)")
                        .component
                    offlineModeCheckbox?.isSelected = settings.offlineMode
                }
                
                row("Ollama URL:") {
                    ollamaUrlField = textField()
                        .columns(COLUMNS_MEDIUM)
                        .enabledIf(offlineModeCheckbox!!.selected)
                        .component
                    ollamaUrlField?.text = settings.ollamaUrl
                }
                
                row("Ollama Model:") {
                    ollamaModelField = textField()
                        .columns(COLUMNS_MEDIUM)
                        .enabledIf(offlineModeCheckbox!!.selected)
                        .comment("Recommended: codellama:7b-instruct, deepseek-coder:6.7b")
                        .component
                    ollamaModelField?.text = settings.ollamaModel
                }
            }
            
            row {
                browserLink("Documentation", "https://docs.codeverify.dev")
                browserLink("Get API Key", "https://codeverify.dev/settings/api")
            }
        }
        
        return panel!!
    }
    
    override fun isModified(): Boolean {
        val settings = CodeVerifySettings.getInstance()
        return enabledCheckbox?.isSelected != settings.enabled ||
                String(apiKeyField?.password ?: charArrayOf()) != (settings.apiKey ?: "") ||
                apiUrlField?.text != settings.apiUrl ||
                includeProofsCheckbox?.isSelected != settings.includeProofs ||
                includeFixesCheckbox?.isSelected != settings.includeFixes ||
                verifyOnSaveCheckbox?.isSelected != settings.verifyOnSave ||
                verifyOnTypeCheckbox?.isSelected != settings.verifyOnType ||
                offlineModeCheckbox?.isSelected != settings.offlineMode ||
                ollamaUrlField?.text != settings.ollamaUrl ||
                ollamaModelField?.text != settings.ollamaModel
    }
    
    override fun apply() {
        val settings = CodeVerifySettings.getInstance()
        
        settings.enabled = enabledCheckbox?.isSelected ?: true
        settings.apiKey = String(apiKeyField?.password ?: charArrayOf()).takeIf { it.isNotBlank() }
        settings.apiUrl = apiUrlField?.text ?: "https://api.codeverify.dev"
        settings.includeProofs = includeProofsCheckbox?.isSelected ?: true
        settings.includeFixes = includeFixesCheckbox?.isSelected ?: true
        settings.verifyOnSave = verifyOnSaveCheckbox?.isSelected ?: true
        settings.verifyOnType = verifyOnTypeCheckbox?.isSelected ?: false
        settings.offlineMode = offlineModeCheckbox?.isSelected ?: false
        settings.ollamaUrl = ollamaUrlField?.text ?: "http://localhost:11434"
        settings.ollamaModel = ollamaModelField?.text ?: "codellama:7b-instruct"
    }
    
    override fun reset() {
        val settings = CodeVerifySettings.getInstance()
        
        enabledCheckbox?.isSelected = settings.enabled
        apiKeyField?.text = settings.apiKey ?: ""
        apiUrlField?.text = settings.apiUrl
        includeProofsCheckbox?.isSelected = settings.includeProofs
        includeFixesCheckbox?.isSelected = settings.includeFixes
        verifyOnSaveCheckbox?.isSelected = settings.verifyOnSave
        verifyOnTypeCheckbox?.isSelected = settings.verifyOnType
        offlineModeCheckbox?.isSelected = settings.offlineMode
        ollamaUrlField?.text = settings.ollamaUrl
        ollamaModelField?.text = settings.ollamaModel
    }
    
    override fun disposeUIResources() {
        panel = null
        enabledCheckbox = null
        apiKeyField = null
        apiUrlField = null
        includeProofsCheckbox = null
        includeFixesCheckbox = null
        verifyOnSaveCheckbox = null
        verifyOnTypeCheckbox = null
        offlineModeCheckbox = null
        ollamaUrlField = null
        ollamaModelField = null
    }
}

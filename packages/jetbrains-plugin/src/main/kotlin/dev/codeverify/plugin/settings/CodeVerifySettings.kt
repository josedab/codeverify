package dev.codeverify.plugin.settings

import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.components.PersistentStateComponent
import com.intellij.openapi.components.State
import com.intellij.openapi.components.Storage

/**
 * Persistent settings for CodeVerify plugin.
 */
@State(
    name = "CodeVerifySettings",
    storages = [Storage("codeverify.xml")]
)
class CodeVerifySettings : PersistentStateComponent<CodeVerifySettings.State> {
    
    private var myState = State()
    
    // Accessors
    var enabled: Boolean
        get() = myState.enabled
        set(value) { myState.enabled = value }
    
    var apiKey: String?
        get() = myState.apiKey
        set(value) { myState.apiKey = value }
    
    var apiUrl: String
        get() = myState.apiUrl
        set(value) { myState.apiUrl = value }
    
    var includeProofs: Boolean
        get() = myState.includeProofs
        set(value) { myState.includeProofs = value }
    
    var includeFixes: Boolean
        get() = myState.includeFixes
        set(value) { myState.includeFixes = value }
    
    var verifyOnSave: Boolean
        get() = myState.verifyOnSave
        set(value) { myState.verifyOnSave = value }
    
    var verifyOnType: Boolean
        get() = myState.verifyOnType
        set(value) { myState.verifyOnType = value }
    
    var verifyDelay: Int
        get() = myState.verifyDelay
        set(value) { myState.verifyDelay = value }
    
    var minSeverity: String
        get() = myState.minSeverity
        set(value) { myState.minSeverity = value }
    
    var offlineMode: Boolean
        get() = myState.offlineMode
        set(value) { myState.offlineMode = value }
    
    var ollamaUrl: String
        get() = myState.ollamaUrl
        set(value) { myState.ollamaUrl = value }
    
    var ollamaModel: String
        get() = myState.ollamaModel
        set(value) { myState.ollamaModel = value }
    
    override fun getState(): State = myState
    
    override fun loadState(state: State) {
        myState = state
    }
    
    /**
     * Settings state class.
     */
    class State {
        var enabled: Boolean = true
        var apiKey: String? = null
        var apiUrl: String = "https://api.codeverify.dev"
        var includeProofs: Boolean = true
        var includeFixes: Boolean = true
        var verifyOnSave: Boolean = true
        var verifyOnType: Boolean = false
        var verifyDelay: Int = 1000 // ms
        var minSeverity: String = "low"
        var offlineMode: Boolean = false
        var ollamaUrl: String = "http://localhost:11434"
        var ollamaModel: String = "codellama:7b-instruct"
    }
    
    companion object {
        fun getInstance(): CodeVerifySettings {
            return ApplicationManager.getApplication().getService(CodeVerifySettings::class.java)
        }
    }
}

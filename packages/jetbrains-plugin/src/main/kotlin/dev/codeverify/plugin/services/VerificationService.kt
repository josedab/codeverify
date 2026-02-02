package dev.codeverify.plugin.services

import com.intellij.openapi.components.Service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.project.Project
import dev.codeverify.plugin.settings.CodeVerifySettings
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Service for communicating with the CodeVerify API.
 * Handles verification requests and caches results.
 */
@Service(Service.Level.PROJECT)
class VerificationService(private val project: Project) {
    
    private val logger = Logger.getInstance(VerificationService::class.java)
    private val gson = Gson()
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()
    
    // Cache for verification results
    private val resultCache = mutableMapOf<String, VerificationResult>()
    
    /**
     * Verify code asynchronously.
     */
    suspend fun verifyCode(
        code: String,
        language: String,
        filePath: String? = null
    ): VerificationResult = withContext(Dispatchers.IO) {
        val settings = CodeVerifySettings.getInstance()
        val apiKey = settings.apiKey
        
        if (apiKey.isNullOrBlank()) {
            return@withContext VerificationResult(
                success = false,
                error = "API key not configured. Please set your API key in Settings > Tools > CodeVerify"
            )
        }
        
        // Check cache
        val cacheKey = "${code.hashCode()}-${language}"
        resultCache[cacheKey]?.let { cached ->
            if (System.currentTimeMillis() - cached.timestamp < CACHE_DURATION_MS) {
                logger.info("Returning cached result for $filePath")
                return@withContext cached
            }
        }
        
        try {
            val request = VerifyRequest(
                code = code,
                language = language,
                includeProof = settings.includeProofs,
                includeFixes = settings.includeFixes
            )
            
            val response = makeApiRequest(settings.apiUrl, apiKey, request)
            
            // Cache the result
            resultCache[cacheKey] = response
            
            response
        } catch (e: Exception) {
            logger.error("Verification failed", e)
            VerificationResult(
                success = false,
                error = "Verification failed: ${e.message}"
            )
        }
    }
    
    /**
     * Make API request to CodeVerify server.
     */
    private fun makeApiRequest(
        baseUrl: String,
        apiKey: String,
        request: VerifyRequest
    ): VerificationResult {
        val json = gson.toJson(request)
        val body = json.toRequestBody("application/json".toMediaType())
        
        val httpRequest = Request.Builder()
            .url("$baseUrl/api/v1/verification/verify")
            .header("X-API-Key", apiKey)
            .header("Content-Type", "application/json")
            .post(body)
            .build()
        
        client.newCall(httpRequest).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("API request failed: ${response.code} ${response.message}")
            }
            
            val responseBody = response.body?.string() 
                ?: throw IOException("Empty response body")
            
            val apiResponse = gson.fromJson(responseBody, ApiVerificationResponse::class.java)
            
            return VerificationResult(
                success = true,
                verified = apiResponse.verified,
                trustScore = apiResponse.trustScore,
                findings = apiResponse.findings.map { finding ->
                    Finding(
                        id = finding.id,
                        category = finding.category,
                        severity = finding.severity,
                        title = finding.title,
                        description = finding.description,
                        lineStart = finding.lineStart,
                        lineEnd = finding.lineEnd,
                        proof = finding.proof,
                        fixSuggestion = finding.fixSuggestion
                    )
                },
                proofSummary = apiResponse.proofSummary,
                timestamp = System.currentTimeMillis()
            )
        }
    }
    
    /**
     * Apply a fix suggestion.
     */
    fun applyFix(finding: Finding, editor: com.intellij.openapi.editor.Editor) {
        val fixCode = finding.fixSuggestion ?: return
        
        com.intellij.openapi.command.WriteCommandAction.runWriteCommandAction(project) {
            val document = editor.document
            val startOffset = document.getLineStartOffset(finding.lineStart - 1)
            val endOffset = document.getLineEndOffset(finding.lineEnd - 1)
            
            document.replaceString(startOffset, endOffset, fixCode)
        }
        
        logger.info("Applied fix for finding: ${finding.id}")
    }
    
    /**
     * Clear the result cache.
     */
    fun clearCache() {
        resultCache.clear()
    }
    
    /**
     * Dispose resources.
     */
    fun dispose() {
        scope.cancel()
        client.dispatcher.executorService.shutdown()
        client.connectionPool.evictAll()
    }
    
    companion object {
        private const val CACHE_DURATION_MS = 5 * 60 * 1000L // 5 minutes
        
        fun getInstance(project: Project): VerificationService {
            return project.getService(VerificationService::class.java)
        }
    }
}

// Data classes

data class VerifyRequest(
    val code: String,
    val language: String,
    val includeProof: Boolean = true,
    val includeFixes: Boolean = true
)

data class ApiVerificationResponse(
    val requestId: String,
    val status: String,
    val verified: Boolean,
    val trustScore: Double,
    val findings: List<ApiFinding>,
    val proofSummary: String?,
    val processingTimeMs: Double,
    val tokensUsed: Int,
    val remainingQuota: Map<String, Int>
)

data class ApiFinding(
    val id: String,
    val category: String,
    val severity: String,
    val title: String,
    val description: String,
    val filePath: String? = null,
    val lineStart: Int? = null,
    val lineEnd: Int? = null,
    val codeSnippet: String? = null,
    val confidence: Double,
    val proof: String? = null,
    val fixSuggestion: String? = null
)

data class VerificationResult(
    val success: Boolean,
    val verified: Boolean = false,
    val trustScore: Double = 0.0,
    val findings: List<Finding> = emptyList(),
    val proofSummary: String? = null,
    val error: String? = null,
    val timestamp: Long = System.currentTimeMillis()
)

data class Finding(
    val id: String,
    val category: String,
    val severity: String,
    val title: String,
    val description: String,
    val lineStart: Int? = null,
    val lineEnd: Int? = null,
    val proof: String? = null,
    val fixSuggestion: String? = null
) {
    val severityLevel: Int
        get() = when (severity.lowercase()) {
            "critical" -> 4
            "high" -> 3
            "medium" -> 2
            "low" -> 1
            else -> 0
        }
}

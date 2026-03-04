package dev.codeverify.plugin.lsp

import com.google.gson.Gson
import com.google.gson.JsonElement
import com.intellij.openapi.Disposable
import com.intellij.openapi.components.Service
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.project.Project
import dev.codeverify.plugin.settings.CodeVerifySettings
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * LSP client that communicates with the CodeVerify LSP server (packages/lsp-server/)
 * over stdio, sending text document notifications and receiving diagnostics.
 */
@Service(Service.Level.PROJECT)
class LspClient(private val project: Project) : Disposable {

    private val log = Logger.getInstance(LspClient::class.java)
    private val gson = Gson()
    private val requestId = AtomicInteger(0)
    private val pendingRequests = ConcurrentHashMap<Int, (JsonElement?) -> Unit>()

    private var process: Process? = null
    private var writer: OutputStreamWriter? = null
    private var readerThread: Thread? = null

    @Volatile
    var isConnected: Boolean = false
        private set

    fun start() {
        if (isConnected) return
        try {
            val settings = CodeVerifySettings.instance
            val serverCommand = settings.lspServerPath.ifBlank { "codeverify-lsp" }

            val pb = ProcessBuilder(serverCommand, "--stdio")
                .redirectErrorStream(false)
            process = pb.start()
            writer = OutputStreamWriter(process!!.outputStream, Charsets.UTF_8)

            readerThread = Thread({
                val reader = BufferedReader(InputStreamReader(process!!.inputStream, Charsets.UTF_8))
                try { readLoop(reader) } catch (e: Exception) { log.warn("LSP reader exited", e) }
            }, "codeverify-lsp-reader").apply { isDaemon = true; start() }

            isConnected = true
            sendRequest("initialize", mapOf(
                "processId" to ProcessHandle.current().pid(),
                "rootUri" to "file://${project.basePath}",
                "capabilities" to mapOf(
                    "textDocument" to mapOf(
                        "publishDiagnostics" to mapOf("relatedInformation" to true),
                        "codeAction" to mapOf("codeActionLiteralSupport" to mapOf(
                            "codeActionKind" to mapOf("valueSet" to listOf("quickfix", "refactor"))
                        ))
                    )
                )
            ))
            log.info("CodeVerify LSP server started")
        } catch (e: Exception) {
            log.warn("Failed to start LSP server: ${e.message}")
            isConnected = false
        }
    }

    fun stop() {
        isConnected = false
        try {
            sendRequest("shutdown", null)
            writer?.close()
            process?.destroyForcibly()
        } catch (_: Exception) {}
        process = null; writer = null
    }

    override fun dispose() = stop()

    // -- Document notifications ---
    fun didOpen(uri: String, languageId: String, version: Int, text: String) {
        sendNotification("textDocument/didOpen", mapOf(
            "textDocument" to mapOf("uri" to uri, "languageId" to languageId, "version" to version, "text" to text)
        ))
    }

    fun didChange(uri: String, version: Int, text: String) {
        sendNotification("textDocument/didChange", mapOf(
            "textDocument" to mapOf("uri" to uri, "version" to version),
            "contentChanges" to listOf(mapOf("text" to text))
        ))
    }

    fun didSave(uri: String) {
        sendNotification("textDocument/didSave", mapOf("textDocument" to mapOf("uri" to uri)))
    }

    fun didClose(uri: String) {
        sendNotification("textDocument/didClose", mapOf("textDocument" to mapOf("uri" to uri)))
    }

    fun requestCodeActions(uri: String, startLine: Int, startChar: Int, endLine: Int, endChar: Int, callback: (JsonElement?) -> Unit) {
        sendRequest("textDocument/codeAction", mapOf(
            "textDocument" to mapOf("uri" to uri),
            "range" to mapOf(
                "start" to mapOf("line" to startLine, "character" to startChar),
                "end" to mapOf("line" to endLine, "character" to endChar)
            ),
            "context" to mapOf("diagnostics" to emptyList<Any>())
        ), callback)
    }

    // -- Transport ---
    private fun sendRequest(method: String, params: Any?, callback: ((JsonElement?) -> Unit)? = null) {
        val id = requestId.incrementAndGet()
        callback?.let { pendingRequests[id] = it }
        send(gson.toJson(mapOf("jsonrpc" to "2.0", "id" to id, "method" to method, "params" to params)))
    }

    private fun sendNotification(method: String, params: Any?) {
        send(gson.toJson(mapOf("jsonrpc" to "2.0", "method" to method, "params" to params)))
    }

    private fun send(json: String) {
        try {
            val w = writer ?: return
            val bytes = json.toByteArray(Charsets.UTF_8)
            synchronized(w) {
                w.write("Content-Length: ${bytes.size}\r\n\r\n")
                w.write(json)
                w.flush()
            }
        } catch (e: Exception) { log.warn("Failed to send LSP message", e) }
    }

    private fun readLoop(reader: BufferedReader) {
        while (isConnected) {
            var contentLength = -1
            while (true) {
                val header = reader.readLine() ?: return
                if (header.isBlank()) break
                if (header.startsWith("Content-Length:"))
                    contentLength = header.substringAfter(":").trim().toIntOrNull() ?: -1
            }
            if (contentLength <= 0) continue
            val buf = CharArray(contentLength)
            var read = 0
            while (read < contentLength) {
                val n = reader.read(buf, read, contentLength - read)
                if (n < 0) return
                read += n
            }
            handleMessage(String(buf))
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun handleMessage(json: String) {
        try {
            val msg = gson.fromJson(json, Map::class.java)
            val id = (msg["id"] as? Number)?.toInt()
            val method = msg["method"] as? String

            if (id != null) pendingRequests.remove(id)?.invoke(gson.toJsonTree(msg["result"]))
            if (method == "textDocument/publishDiagnostics") {
                val params = msg["params"] as? Map<*, *> ?: return
                log.info("Received diagnostics for ${params["uri"]}")
            }
        } catch (e: Exception) { log.warn("Failed to parse LSP message", e) }
    }
}

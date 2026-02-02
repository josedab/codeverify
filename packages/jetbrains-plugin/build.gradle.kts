plugins {
    id("java")
    id("org.jetbrains.kotlin.jvm") version "1.9.21"
    id("org.jetbrains.intellij") version "1.16.1"
}

group = "dev.codeverify"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.google.code.gson:gson:2.10.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    testImplementation("junit:junit:4.13.2")
}

// See https://plugins.jetbrains.com/docs/intellij/tools-gradle-intellij-plugin.html
intellij {
    version.set("2023.3")
    type.set("IC") // IntelliJ IDEA Community Edition
    
    plugins.set(listOf(
        "com.intellij.java",
        "org.jetbrains.kotlin",
        "PythonCore:233.11799.241",
        "JavaScript"
    ))
}

tasks {
    withType<JavaCompile> {
        sourceCompatibility = "17"
        targetCompatibility = "17"
    }
    withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
        kotlinOptions.jvmTarget = "17"
    }

    patchPluginXml {
        sinceBuild.set("233")
        untilBuild.set("241.*")
        
        pluginDescription.set("""
            <h2>CodeVerify - AI-Powered Code Verification</h2>
            <p>Formal verification meets AI-powered code review for JetBrains IDEs.</p>
            
            <h3>Features</h3>
            <ul>
                <li>🔍 Real-time code verification as you type</li>
                <li>🛡️ Formal proofs using Z3 SMT solver</li>
                <li>🤖 AI-powered semantic analysis</li>
                <li>🎯 Trust scoring for code changes</li>
                <li>🔧 One-click fix suggestions</li>
                <li>📊 Proof coverage visualization</li>
            </ul>
            
            <h3>Supported Languages</h3>
            <ul>
                <li>Python</li>
                <li>TypeScript/JavaScript</li>
                <li>Java</li>
                <li>Kotlin</li>
                <li>Go</li>
            </ul>
        """.trimIndent())
        
        changeNotes.set("""
            <h3>0.1.0</h3>
            <ul>
                <li>Initial release</li>
                <li>Real-time verification</li>
                <li>Inline annotations</li>
                <li>Fix suggestions</li>
            </ul>
        """.trimIndent())
    }

    signPlugin {
        certificateChain.set(System.getenv("CERTIFICATE_CHAIN"))
        privateKey.set(System.getenv("PRIVATE_KEY"))
        password.set(System.getenv("PRIVATE_KEY_PASSWORD"))
    }

    publishPlugin {
        token.set(System.getenv("PUBLISH_TOKEN"))
    }
}

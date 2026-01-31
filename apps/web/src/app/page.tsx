import Link from "next/link";
import { Shield, Zap, GitPullRequest, CheckCircle } from "lucide-react";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col">
      {/* Navigation */}
      <nav className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center">
              <Shield className="h-8 w-8 text-primary-600" />
              <span className="ml-2 text-xl font-bold text-gray-900 dark:text-white">
                CodeVerify
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <Link
                href="/docs"
                className="text-gray-600 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white"
              >
                Docs
              </Link>
              <Link
                href="/login"
                className="bg-primary-600 text-white px-4 py-2 rounded-lg hover:bg-primary-700 transition"
              >
                Sign In
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="flex-1 flex items-center justify-center px-4 py-16">
        <div className="max-w-4xl text-center">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 dark:text-white mb-6">
            AI-Powered Code Review with{" "}
            <span className="text-primary-600">Formal Verification</span>
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
            Catch bugs, security vulnerabilities, and logical errors in code—
            especially AI-generated code. Backed by mathematical proofs.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/signup"
              className="bg-primary-600 text-white px-8 py-3 rounded-lg text-lg font-medium hover:bg-primary-700 transition"
            >
              Get Started Free
            </Link>
            <Link
              href="/demo"
              className="border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 px-8 py-3 rounded-lg text-lg font-medium hover:bg-gray-50 dark:hover:bg-gray-800 transition"
            >
              See Demo
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-white dark:bg-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center text-gray-900 dark:text-white mb-12">
            Why CodeVerify?
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <FeatureCard
              icon={<Zap className="h-8 w-8 text-primary-600" />}
              title="Hybrid AI + Formal Methods"
              description="Combines LLM semantic understanding with SMT solver-based verification for mathematically-grounded results."
            />
            <FeatureCard
              icon={<GitPullRequest className="h-8 w-8 text-primary-600" />}
              title="GitHub-Native Integration"
              description="Seamless PR comments, checks, and suggested fixes. Works with your existing workflow."
            />
            <FeatureCard
              icon={<CheckCircle className="h-8 w-8 text-primary-600" />}
              title="Actionable Results"
              description="Every finding includes a clear explanation and one-click fix suggestions. Not just problems—solutions."
            />
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200 dark:border-gray-800 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <Shield className="h-6 w-6 text-gray-400" />
              <span className="ml-2 text-gray-600 dark:text-gray-400">
                © 2026 CodeVerify
              </span>
            </div>
            <div className="flex space-x-6">
              <Link href="/privacy" className="text-gray-600 dark:text-gray-400 hover:text-gray-900">
                Privacy
              </Link>
              <Link href="/terms" className="text-gray-600 dark:text-gray-400 hover:text-gray-900">
                Terms
              </Link>
              <Link href="https://github.com/codeverify" className="text-gray-600 dark:text-gray-400 hover:text-gray-900">
                GitHub
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="p-6 rounded-xl border border-gray-200 dark:border-gray-700 hover:shadow-lg transition">
      <div className="mb-4">{icon}</div>
      <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
        {title}
      </h3>
      <p className="text-gray-600 dark:text-gray-300">{description}</p>
    </div>
  );
}

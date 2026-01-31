import Link from "next/link";
import { Shield, LayoutDashboard, Settings, LogOut, GitPullRequest, GitBranch, CreditCard } from "lucide-react";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-900 text-white">
        <div className="p-4">
          <Link href="/" className="flex items-center space-x-2">
            <Shield className="h-8 w-8 text-primary-500" />
            <span className="text-xl font-bold">CodeVerify</span>
          </Link>
        </div>

        <nav className="mt-8 px-4">
          <ul className="space-y-2">
            <NavItem href="/dashboard" icon={<LayoutDashboard />} label="Dashboard" />
            <NavItem href="/dashboard/analyses" icon={<GitPullRequest />} label="Analyses" />
            <NavItem href="/dashboard/repositories" icon={<GitBranch />} label="Repositories" />
            <NavItem href="/dashboard/usage" icon={<CreditCard />} label="Usage & Billing" />
            <NavItem href="/dashboard/settings" icon={<Settings />} label="Settings" />
          </ul>
        </nav>

        <div className="absolute bottom-0 w-64 p-4 border-t border-gray-800">
          <button className="flex items-center space-x-2 text-gray-400 hover:text-white w-full">
            <LogOut className="h-5 w-5" />
            <span>Sign Out</span>
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 bg-gray-50 dark:bg-gray-900">
        <div className="p-8">{children}</div>
      </main>
    </div>
  );
}

function NavItem({
  href,
  icon,
  label,
}: {
  href: string;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <li>
      <Link
        href={href}
        className="flex items-center space-x-3 px-4 py-2 rounded-lg text-gray-300 hover:bg-gray-800 hover:text-white transition"
      >
        {icon}
        <span>{label}</span>
      </Link>
    </li>
  );
}

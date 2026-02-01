import { Bell, Key, Shield, Users, Building2 } from "lucide-react";

export default function SettingsPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">Settings</h1>

      <div className="space-y-6">
        <SettingsSection icon={<Building2 />} title="Organization" description="Manage organization settings">
          <SettingsRow label="Organization Name" value="Acme Corp" />
          <SettingsRow label="Plan" value="Team ($79/user/month)" />
          <SettingsRow label="Members" value="24 users" />
        </SettingsSection>

        <SettingsSection icon={<Shield />} title="Verification" description="Configure analysis settings">
          <ToggleSetting label="Formal Verification" description="Use Z3 SMT solver" defaultChecked />
          <ToggleSetting label="Security Scanning" description="Scan for vulnerabilities" defaultChecked />
          <ToggleSetting label="AI Analysis" description="Use LLM for understanding" defaultChecked />
        </SettingsSection>

        <SettingsSection icon={<Bell />} title="Notifications" description="Configure notifications">
          <ToggleSetting label="PR Comments" description="Post analysis as PR comment" defaultChecked />
          <ToggleSetting label="Slack" description="Post to Slack channel" defaultChecked={false} />
        </SettingsSection>

        <SettingsSection icon={<Key />} title="API Access" description="Manage API keys">
          <div className="flex justify-between items-center py-2">
            <div>
              <p className="font-medium text-gray-900 dark:text-white">Production Key</p>
              <p className="text-sm text-gray-500">cv_prod_****3f7a</p>
            </div>
            <button className="text-red-600 hover:text-red-700">Revoke</button>
          </div>
          <button className="px-4 py-2 bg-primary-600 text-white rounded-lg">Generate New Key</button>
        </SettingsSection>
      </div>
    </div>
  );
}

function SettingsSection({ icon, title, description, children }: { icon: React.ReactNode; title: string; description: string; children: React.ReactNode }) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-start space-x-4 mb-6">
        <div className="p-2 bg-primary-100 dark:bg-primary-900 rounded-lg text-primary-600">{icon}</div>
        <div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">{title}</h2>
          <p className="text-sm text-gray-500">{description}</p>
        </div>
      </div>
      <div className="space-y-4">{children}</div>
    </div>
  );
}

function SettingsRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between py-2">
      <span className="text-gray-600 dark:text-gray-300">{label}</span>
      <span className="font-medium text-gray-900 dark:text-white">{value}</span>
    </div>
  );
}

function ToggleSetting({ label, description, defaultChecked }: { label: string; description: string; defaultChecked?: boolean }) {
  return (
    <div className="flex justify-between items-center py-2">
      <div>
        <p className="font-medium text-gray-900 dark:text-white">{label}</p>
        <p className="text-sm text-gray-500">{description}</p>
      </div>
      <label className="relative inline-flex items-center cursor-pointer">
        <input type="checkbox" defaultChecked={defaultChecked} className="sr-only peer" />
        <div className="w-11 h-6 bg-gray-200 peer-checked:bg-primary-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all"></div>
      </label>
    </div>
  );
}

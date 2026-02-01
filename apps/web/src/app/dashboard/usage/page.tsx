import { BarChart3, TrendingUp, CreditCard, Zap } from "lucide-react";

// Mock data
const usageData = {
  current_period: {
    start: "2026-01-01",
    end: "2026-01-31",
  },
  analyses: {
    used: 127,
    limit: 500,
    remaining: 373,
  },
  plan: {
    tier: "team",
    name: "Team",
    price: 79,
  },
  history: [
    { month: "Aug", analyses: 89 },
    { month: "Sep", analyses: 112 },
    { month: "Oct", analyses: 98 },
    { month: "Nov", analyses: 134 },
    { month: "Dec", analyses: 156 },
    { month: "Jan", analyses: 127 },
  ],
};

export default function UsagePage() {
  const usagePercent = Math.round(
    (usageData.analyses.used / usageData.analyses.limit) * 100
  );

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Usage & Billing
        </h1>
        <button className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700">
          Upgrade Plan
        </button>
      </div>

      {/* Current Plan */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-primary-100 dark:bg-primary-900 rounded-lg">
              <CreditCard className="h-6 w-6 text-primary-600" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                {usageData.plan.name} Plan
              </h2>
              <p className="text-sm text-gray-500">
                ${usageData.plan.price}/user/month
              </p>
            </div>
          </div>
          <button className="text-primary-600 hover:text-primary-700 text-sm font-medium">
            Manage subscription →
          </button>
        </div>
      </div>

      {/* Usage Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <UsageCard
          icon={<Zap className="h-5 w-5" />}
          title="Analyses This Month"
          value={usageData.analyses.used}
          max={usageData.analyses.limit}
          percent={usagePercent}
        />
        <UsageCard
          icon={<BarChart3 className="h-5 w-5" />}
          title="Remaining"
          value={usageData.analyses.remaining}
          subtitle="analyses available"
        />
        <UsageCard
          icon={<TrendingUp className="h-5 w-5" />}
          title="Avg. Per Day"
          value={Math.round(usageData.analyses.used / 28)}
          subtitle="analyses/day"
        />
      </div>

      {/* Usage History */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-8">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Usage History
        </h3>
        <div className="flex items-end justify-between h-48 gap-4">
          {usageData.history.map((month, index) => (
            <div key={month.month} className="flex-1 flex flex-col items-center">
              <div className="flex-1 w-full flex items-end">
                <div
                  className="w-full bg-primary-500 rounded-t transition-all"
                  style={{
                    height: `${(month.analyses / 200) * 100}%`,
                    minHeight: "8px",
                  }}
                />
              </div>
              <p className="mt-2 text-xs text-gray-500">{month.month}</p>
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                {month.analyses}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Plan Comparison */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
          Available Plans
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <PlanCard
            name="Free"
            price="$0"
            features={["50 analyses/month", "5 repositories", "3 users", "7 day retention"]}
            current={false}
          />
          <PlanCard
            name="Team"
            price="$79"
            priceUnit="/user/month"
            features={["500 analyses/month", "50 repositories", "25 users", "90 day retention", "Priority support"]}
            current={true}
            highlighted={true}
          />
          <PlanCard
            name="Enterprise"
            price="Custom"
            features={["Unlimited analyses", "Unlimited repos", "SSO/SAML", "1 year retention", "Dedicated support", "SLA"]}
            current={false}
            cta="Contact Sales"
          />
        </div>
      </div>
    </div>
  );
}

function UsageCard({
  icon,
  title,
  value,
  max,
  percent,
  subtitle,
}: {
  icon: React.ReactNode;
  title: string;
  value: number;
  max?: number;
  percent?: number;
  subtitle?: string;
}) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center space-x-2 text-gray-500 mb-2">
        {icon}
        <span className="text-sm">{title}</span>
      </div>
      <p className="text-3xl font-bold text-gray-900 dark:text-white">
        {value}
        {max && <span className="text-lg text-gray-500">/{max}</span>}
      </p>
      {percent !== undefined && (
        <div className="mt-3">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full ${
                percent > 80 ? "bg-red-500" : percent > 60 ? "bg-yellow-500" : "bg-green-500"
              }`}
              style={{ width: `${percent}%` }}
            />
          </div>
          <p className="mt-1 text-xs text-gray-500">{percent}% used</p>
        </div>
      )}
      {subtitle && <p className="mt-1 text-sm text-gray-500">{subtitle}</p>}
    </div>
  );
}

function PlanCard({
  name,
  price,
  priceUnit,
  features,
  current,
  highlighted,
  cta,
}: {
  name: string;
  price: string;
  priceUnit?: string;
  features: string[];
  current: boolean;
  highlighted?: boolean;
  cta?: string;
}) {
  return (
    <div
      className={`rounded-xl border p-6 ${
        highlighted
          ? "border-primary-500 ring-2 ring-primary-500"
          : "border-gray-200 dark:border-gray-700"
      }`}
    >
      {current && (
        <span className="text-xs bg-primary-100 text-primary-700 px-2 py-1 rounded-full">
          Current Plan
        </span>
      )}
      <h4 className="text-xl font-bold text-gray-900 dark:text-white mt-2">{name}</h4>
      <p className="text-3xl font-bold text-gray-900 dark:text-white mt-2">
        {price}
        {priceUnit && <span className="text-sm text-gray-500">{priceUnit}</span>}
      </p>
      <ul className="mt-4 space-y-2">
        {features.map((feature) => (
          <li key={feature} className="flex items-center text-sm text-gray-600 dark:text-gray-300">
            <span className="text-green-500 mr-2">✓</span>
            {feature}
          </li>
        ))}
      </ul>
      <button
        className={`mt-6 w-full py-2 rounded-lg ${
          current
            ? "bg-gray-100 text-gray-500 cursor-default"
            : "bg-primary-600 text-white hover:bg-primary-700"
        }`}
        disabled={current}
      >
        {current ? "Current" : cta || "Upgrade"}
      </button>
    </div>
  );
}

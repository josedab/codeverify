"use client";

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { format } from "date-fns";
import {
  Download,
  Search,
  Filter,
  Clock,
  User as UserIcon,
  Activity,
  AlertCircle,
} from "lucide-react";
import api, { AuditLog, AuditLogStats } from "@/lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Button,
  DataTable,
  Select,
  Input,
  DateRangePicker,
  SlideOver,
  Badge,
  Column,
} from "@/components/ui";

export default function AuditLogsPage() {
  const [page, setPage] = useState(1);
  const [filters, setFilters] = useState({
    action: "",
    resource_type: "",
    search: "",
    start_date: "",
    end_date: "",
  });
  const [selectedLog, setSelectedLog] = useState<AuditLog | null>(null);

  // Fetch audit logs
  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ["auditLogs", page, filters],
    queryFn: () =>
      api.getAuditLogs({
        page,
        page_size: 20,
        action: filters.action || undefined,
        resource_type: filters.resource_type || undefined,
        search: filters.search || undefined,
        start_date: filters.start_date || undefined,
        end_date: filters.end_date || undefined,
      }),
  });

  // Fetch stats
  const { data: stats } = useQuery({
    queryKey: ["auditLogStats"],
    queryFn: () => api.getAuditLogStats(),
  });

  // Fetch filter options
  const { data: actions } = useQuery({
    queryKey: ["auditLogActions"],
    queryFn: () => api.getAuditLogActions(),
  });

  const { data: resourceTypes } = useQuery({
    queryKey: ["auditLogResourceTypes"],
    queryFn: () => api.getAuditLogResourceTypes(),
  });

  const handleExport = async (format: "csv" | "json") => {
    try {
      const blob = await api.exportAuditLogs({
        format,
        start_date: filters.start_date || undefined,
        end_date: filters.end_date || undefined,
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `audit-logs.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Export failed:", error);
    }
  };

  const columns: Column<AuditLog>[] = [
    {
      key: "created_at",
      header: "Timestamp",
      sortable: true,
      render: (row) => (
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-gray-400" />
          <span className="text-sm">
            {format(new Date(row.created_at), "MMM d, yyyy HH:mm:ss")}
          </span>
        </div>
      ),
    },
    {
      key: "username",
      header: "User",
      render: (row) => (
        <div className="flex items-center gap-2">
          <UserIcon className="h-4 w-4 text-gray-400" />
          <span>{row.username || "System"}</span>
        </div>
      ),
    },
    {
      key: "action",
      header: "Action",
      sortable: true,
      render: (row) => <ActionBadge action={row.action} />,
    },
    {
      key: "resource_type",
      header: "Resource",
      render: (row) =>
        row.resource_type ? (
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {row.resource_type}
          </span>
        ) : (
          <span className="text-gray-400">-</span>
        ),
    },
    {
      key: "ip_address",
      header: "IP Address",
      render: (row) => (
        <span className="font-mono text-sm">{row.ip_address || "-"}</span>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Audit Logs
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Track all security and compliance events
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => handleExport("csv")}>
            <Download className="h-4 w-4 mr-2" />
            Export CSV
          </Button>
          <Button variant="outline" onClick={() => handleExport("json")}>
            <Download className="h-4 w-4 mr-2" />
            Export JSON
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatsCard
            title="Total Events"
            value={stats.total_events.toLocaleString()}
            icon={<Activity className="h-5 w-5" />}
          />
          <StatsCard
            title="Today"
            value={stats.events_today.toLocaleString()}
            icon={<Clock className="h-5 w-5" />}
          />
          <StatsCard
            title="This Week"
            value={stats.events_this_week.toLocaleString()}
            icon={<Calendar className="h-5 w-5" />}
          />
          <StatsCard
            title="Top Users"
            value={stats.top_users.length.toString()}
            icon={<UserIcon className="h-5 w-5" />}
          />
        </div>
      )}

      {/* Filters */}
      <Card padding="md">
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex-1 min-w-[200px]">
            <Input
              label="Search"
              placeholder="Search actions, details..."
              value={filters.search}
              onChange={(e) =>
                setFilters({ ...filters, search: e.target.value })
              }
            />
          </div>
          <div className="w-48">
            <Select
              label="Action"
              value={filters.action}
              onChange={(e) =>
                setFilters({ ...filters, action: e.target.value })
              }
            >
              <option value="">All Actions</option>
              {actions?.map((action) => (
                <option key={action} value={action}>
                  {action}
                </option>
              ))}
            </Select>
          </div>
          <div className="w-48">
            <Select
              label="Resource Type"
              value={filters.resource_type}
              onChange={(e) =>
                setFilters({ ...filters, resource_type: e.target.value })
              }
            >
              <option value="">All Resources</option>
              {resourceTypes?.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </Select>
          </div>
          <DateRangePicker
            label="Date Range"
            startDate={filters.start_date}
            endDate={filters.end_date}
            onStartDateChange={(date) =>
              setFilters({ ...filters, start_date: date })
            }
            onEndDateChange={(date) =>
              setFilters({ ...filters, end_date: date })
            }
          />
          <Button
            variant="ghost"
            onClick={() =>
              setFilters({
                action: "",
                resource_type: "",
                search: "",
                start_date: "",
                end_date: "",
              })
            }
          >
            Clear
          </Button>
        </div>
      </Card>

      {/* Logs Table */}
      <Card padding="none">
        <DataTable
          data={logsData?.items || []}
          columns={columns}
          keyExtractor={(row) => row.id}
          onRowClick={setSelectedLog}
          loading={logsLoading}
          emptyMessage="No audit logs found"
          showPagination={false}
        />
        {logsData && logsData.total_pages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200 dark:border-gray-700">
            <span className="text-sm text-gray-500">
              Page {page} of {logsData.total_pages} ({logsData.total} total)
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page === 1}
                onClick={() => setPage((p) => p - 1)}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={page === logsData.total_pages}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </Card>

      {/* Detail Slide-over */}
      <SlideOver
        open={!!selectedLog}
        onClose={() => setSelectedLog(null)}
        title="Audit Log Details"
        width="lg"
      >
        {selectedLog && <AuditLogDetail log={selectedLog} />}
      </SlideOver>
    </div>
  );
}

function StatsCard({
  title,
  value,
  icon,
}: {
  title: string;
  value: string;
  icon: React.ReactNode;
}) {
  return (
    <Card padding="md">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg text-gray-500 dark:text-gray-400">
          {icon}
        </div>
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">{title}</p>
          <p className="text-xl font-semibold text-gray-900 dark:text-white">
            {value}
          </p>
        </div>
      </div>
    </Card>
  );
}

function Calendar({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
      />
    </svg>
  );
}

function ActionBadge({ action }: { action: string }) {
  const variant = action.includes("delete")
    ? "danger"
    : action.includes("create")
    ? "success"
    : action.includes("update")
    ? "warning"
    : "default";

  return <Badge variant={variant}>{action}</Badge>;
}

function AuditLogDetail({ log }: { log: AuditLog }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
            Timestamp
          </label>
          <p className="mt-1 text-gray-900 dark:text-white">
            {format(new Date(log.created_at), "MMMM d, yyyy HH:mm:ss")}
          </p>
        </div>
        <div>
          <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
            User
          </label>
          <p className="mt-1 text-gray-900 dark:text-white">
            {log.username || "System"}
          </p>
        </div>
        <div>
          <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
            Action
          </label>
          <div className="mt-1">
            <ActionBadge action={log.action} />
          </div>
        </div>
        <div>
          <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
            Resource Type
          </label>
          <p className="mt-1 text-gray-900 dark:text-white">
            {log.resource_type || "-"}
          </p>
        </div>
        <div>
          <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
            Resource ID
          </label>
          <p className="mt-1 font-mono text-sm text-gray-900 dark:text-white">
            {log.resource_id || "-"}
          </p>
        </div>
        <div>
          <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
            IP Address
          </label>
          <p className="mt-1 font-mono text-sm text-gray-900 dark:text-white">
            {log.ip_address || "-"}
          </p>
        </div>
      </div>

      <div>
        <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
          User Agent
        </label>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-300 break-all">
          {log.user_agent || "-"}
        </p>
      </div>

      {Object.keys(log.details).length > 0 && (
        <div>
          <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
            Details
          </label>
          <pre className="mt-2 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg text-sm overflow-auto max-h-64">
            {JSON.stringify(log.details, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

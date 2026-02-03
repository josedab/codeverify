"use client";

import { HTMLAttributes, forwardRef } from "react";
import { clsx } from "clsx";

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "success" | "warning" | "danger" | "info";
  size?: "sm" | "md";
}

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = "default", size = "sm", children, ...props }, ref) => {
    const variants = {
      default:
        "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200",
      success:
        "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
      warning:
        "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
      danger:
        "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
      info:
        "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    };

    const sizes = {
      sm: "px-2 py-0.5 text-xs",
      md: "px-2.5 py-1 text-sm",
    };

    return (
      <span
        ref={ref}
        className={clsx(
          "inline-flex items-center font-medium rounded-full",
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      >
        {children}
      </span>
    );
  }
);

Badge.displayName = "Badge";

export function SeverityBadge({ severity }: { severity: string }) {
  const variantMap: Record<string, BadgeProps["variant"]> = {
    critical: "danger",
    high: "danger",
    medium: "warning",
    low: "info",
  };

  return (
    <Badge variant={variantMap[severity] || "default"}>
      {severity}
    </Badge>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const variantMap: Record<string, BadgeProps["variant"]> = {
    passed: "success",
    completed: "success",
    failed: "danger",
    error: "danger",
    running: "info",
    pending: "default",
    cancelled: "default",
  };

  return (
    <Badge variant={variantMap[status] || "default"}>
      {status}
    </Badge>
  );
}

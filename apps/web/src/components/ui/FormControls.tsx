"use client";

import { forwardRef, SelectHTMLAttributes, InputHTMLAttributes } from "react";
import { clsx } from "clsx";

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, label, error, children, ...props }, ref) => {
    return (
      <div className="space-y-1">
        {label && (
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
          </label>
        )}
        <select
          ref={ref}
          className={clsx(
            "block w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm",
            "focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500",
            "dark:border-gray-600 dark:bg-gray-800 dark:text-white",
            error && "border-red-500 focus:border-red-500 focus:ring-red-500",
            className
          )}
          {...props}
        >
          {children}
        </select>
        {error && <p className="text-sm text-red-500">{error}</p>}
      </div>
    );
  }
);

Select.displayName = "Select";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, type = "text", ...props }, ref) => {
    return (
      <div className="space-y-1">
        {label && (
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
          </label>
        )}
        <input
          ref={ref}
          type={type}
          className={clsx(
            "block w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm",
            "focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500",
            "dark:border-gray-600 dark:bg-gray-800 dark:text-white",
            "placeholder:text-gray-400 dark:placeholder:text-gray-500",
            error && "border-red-500 focus:border-red-500 focus:ring-red-500",
            className
          )}
          {...props}
        />
        {error && <p className="text-sm text-red-500">{error}</p>}
      </div>
    );
  }
);

Input.displayName = "Input";

export interface DateRangePickerProps {
  startDate: string;
  endDate: string;
  onStartDateChange: (date: string) => void;
  onEndDateChange: (date: string) => void;
  label?: string;
}

export function DateRangePicker({
  startDate,
  endDate,
  onStartDateChange,
  onEndDateChange,
  label,
}: DateRangePickerProps) {
  return (
    <div className="space-y-1">
      {label && (
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          {label}
        </label>
      )}
      <div className="flex items-center gap-2">
        <input
          type="date"
          value={startDate}
          onChange={(e) => onStartDateChange(e.target.value)}
          className={clsx(
            "block rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm",
            "focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500",
            "dark:border-gray-600 dark:bg-gray-800 dark:text-white"
          )}
        />
        <span className="text-gray-500">to</span>
        <input
          type="date"
          value={endDate}
          onChange={(e) => onEndDateChange(e.target.value)}
          className={clsx(
            "block rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm",
            "focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500",
            "dark:border-gray-600 dark:bg-gray-800 dark:text-white"
          )}
        />
      </div>
    </div>
  );
}

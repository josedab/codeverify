import { test, expect } from '@playwright/test';

// Mock authentication for dashboard tests
test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // In a real scenario, we'd mock the auth session
    // For now, test the public parts of dashboard
    await page.goto('/dashboard');
  });

  test('should display dashboard layout', async ({ page }) => {
    // Check for main navigation elements
    await expect(page.getByRole('navigation')).toBeVisible();
  });

  test('should have sidebar navigation links', async ({ page }) => {
    await expect(page.getByRole('link', { name: /Dashboard/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /Analyses/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /Repositories/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /Settings/i })).toBeVisible();
  });
});

test.describe('Analyses Page', () => {
  test('should display analyses list', async ({ page }) => {
    await page.goto('/dashboard/analyses');
    
    await expect(page.getByRole('heading', { name: /Analyses/i })).toBeVisible();
    await expect(page.getByPlaceholder(/Search analyses/i)).toBeVisible();
  });

  test('should have filter button', async ({ page }) => {
    await page.goto('/dashboard/analyses');
    
    await expect(page.getByRole('button', { name: /Filters/i })).toBeVisible();
  });
});

test.describe('Repositories Page', () => {
  test('should display repositories list', async ({ page }) => {
    await page.goto('/dashboard/repositories');
    
    await expect(page.getByRole('heading', { name: /Repositories/i })).toBeVisible();
  });

  test('should have add repository button', async ({ page }) => {
    await page.goto('/dashboard/repositories');
    
    await expect(page.getByRole('button', { name: /Add Repository/i })).toBeVisible();
  });
});

test.describe('Settings Page', () => {
  test('should display settings sections', async ({ page }) => {
    await page.goto('/dashboard/settings');
    
    await expect(page.getByRole('heading', { name: /Settings/i })).toBeVisible();
    await expect(page.getByText(/Organization/i)).toBeVisible();
    await expect(page.getByText(/Verification/i)).toBeVisible();
    await expect(page.getByText(/Notifications/i)).toBeVisible();
  });

  test('should have toggle switches', async ({ page }) => {
    await page.goto('/dashboard/settings');
    
    // Check for toggle switches
    const toggles = page.locator('input[type="checkbox"]');
    await expect(toggles.first()).toBeVisible();
  });
});

import { test, expect } from '@playwright/test';

// Mock authentication for dashboard tests
test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display dashboard layout', async ({ page }) => {
    await expect(page.getByRole('navigation')).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Dashboard', exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Recent Analyses' }),
    ).toBeVisible();
  });

  test('should have sidebar navigation links', async ({ page }) => {
    const navigation = page.getByRole('navigation');
    await expect(
      navigation.getByRole('link', { name: 'Dashboard', exact: true }),
    ).toHaveAttribute('href', '/dashboard');
    await expect(
      navigation.getByRole('link', { name: 'Analyses', exact: true }),
    ).toHaveAttribute('href', '/dashboard/analyses');
    await expect(
      navigation.getByRole('link', { name: 'Repositories', exact: true }),
    ).toHaveAttribute('href', '/dashboard/repositories');
    await expect(
      navigation.getByRole('link', { name: 'Settings', exact: true }),
    ).toHaveAttribute('href', '/dashboard/settings');
  });
});

test.describe('Analyses Page', () => {
  test('should display analyses list', async ({ page }) => {
    await page.goto('/dashboard/analyses');

    await expect(
      page.getByRole('heading', { name: 'Analyses', exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole('textbox', { name: /search analyses/i }),
    ).toBeVisible();
    await expect(
      page.getByRole('row', { name: /acme\/api-service.*#423/i }),
    ).toBeVisible();
  });

  test('should have filter button', async ({ page }) => {
    await page.goto('/dashboard/analyses');

    await expect(
      page.getByRole('button', { name: 'Filters', exact: true }),
    ).toBeVisible();
  });
});

test.describe('Repositories Page', () => {
  test('should display repositories list', async ({ page }) => {
    await page.goto('/dashboard/repositories');

    await expect(
      page.getByRole('heading', { name: 'Repositories', exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'acme/api-service' }),
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'acme/web-app' }),
    ).toBeVisible();
  });

  test('should have add repository button', async ({ page }) => {
    await page.goto('/dashboard/repositories');

    await expect(
      page.getByRole('button', { name: 'Add Repository', exact: true }),
    ).toBeVisible();
  });
});

test.describe('Settings Page', () => {
  test('should display settings sections', async ({ page }) => {
    await page.goto('/dashboard/settings');

    await expect(
      page.getByRole('heading', { name: 'Settings', exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Organization', exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Verification', exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Notifications', exact: true }),
    ).toBeVisible();
  });

  test('should have toggle switches', async ({ page }) => {
    await page.goto('/dashboard/settings');

    const toggles = page.getByRole('checkbox');
    await expect(toggles).toHaveCount(5);
    await expect(toggles.first()).toBeChecked();
    await expect(toggles.last()).not.toBeChecked();
  });
});

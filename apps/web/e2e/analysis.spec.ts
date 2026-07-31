import { test, expect } from '@playwright/test';

test.describe('Analysis Detail Page', () => {
  test('should display analysis header', async ({ page }) => {
    await page.goto('/analysis/test-analysis-id');

    await expect(
      page.getByRole('heading', { name: 'acme/api-service #423' }),
    ).toBeVisible();
    await expect(
      page.getByText('Add user authentication endpoints'),
    ).toBeVisible();
    await expect(page.getByText('Passed', { exact: true })).toBeVisible();
    await expect(
      page.locator('header a[href="/dashboard/analyses"]'),
    ).toBeVisible();
  });

  test('should display summary cards', async ({ page }) => {
    await page.goto('/analysis/test-analysis-id');

    await expect(page.getByText('Total Issues', { exact: true })).toBeVisible();
    await expect(page.getByText('Critical/High', { exact: true })).toBeVisible();
    await expect(page.getByText('Medium/Low', { exact: true })).toBeVisible();
    await expect(page.getByText('Duration', { exact: true })).toBeVisible();
  });

  test('should display pipeline stages', async ({ page }) => {
    await page.goto('/analysis/test-analysis-id');

    await expect(
      page.getByRole('heading', { name: 'Analysis Pipeline' }),
    ).toBeVisible();
    await expect(page.getByText('fetch', { exact: true })).toBeVisible();
    await expect(page.getByText('parse', { exact: true })).toBeVisible();
    await expect(page.getByText('semantic', { exact: true })).toBeVisible();
    await expect(page.getByText('verify', { exact: true })).toBeVisible();
  });

  test('should display findings section', async ({ page }) => {
    await page.goto('/analysis/test-analysis-id');

    await expect(
      page.getByRole('heading', { name: 'Findings (2)' }),
    ).toBeVisible();
    await expect(
      page.getByRole('heading', {
        name: 'Potential SQL injection in user query',
      }),
    ).toBeVisible();
  });

  test('should show severity badges', async ({ page }) => {
    await page.goto('/analysis/test-analysis-id');

    await expect(page.getByText('medium', { exact: true })).toBeVisible();
    await expect(page.getByText('low', { exact: true })).toBeVisible();
  });

  test('should display fix suggestions', async ({ page }) => {
    await page.goto('/analysis/test-analysis-id');

    await expect(page.getByText('Suggested fix:', { exact: true })).toHaveCount(
      2,
    );
    await expect(
      page.getByText(/SELECT \* FROM users WHERE id = \?/i),
    ).toBeVisible();
  });
});

test.describe('Analysis Flow', () => {
  test('should navigate from analyses list to detail', async ({ page }) => {
    await page.goto('/dashboard/analyses');

    const analysisRow = page.getByRole('row', {
      name: /acme\/api-service.*#423/i,
    });
    const analysisLink = analysisRow.getByRole('link');
    await expect(analysisLink).toHaveAttribute('href', '/analysis/a1b2c3d4');

    await analysisLink.click();

    await expect(page).toHaveURL('/analysis/a1b2c3d4');
    await expect(
      page.getByRole('heading', { name: 'Analysis Pipeline' }),
    ).toBeVisible();
  });
});

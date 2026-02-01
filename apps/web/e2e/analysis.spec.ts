import { test, expect } from '@playwright/test';

test.describe('Analysis Detail Page', () => {
  test('should display analysis header', async ({ page }) => {
    // Using a mock analysis ID
    await page.goto('/analysis/test-analysis-id');
    
    // Check for main components
    await expect(page.getByRole('link', { name: /back/i })).toBeVisible();
  });

  test('should display summary cards', async ({ page }) => {
    await page.goto('/analysis/test-analysis-id');
    
    await expect(page.getByText(/Total Issues/i)).toBeVisible();
    await expect(page.getByText(/Duration/i)).toBeVisible();
  });

  test('should display pipeline stages', async ({ page }) => {
    await page.goto('/analysis/test-analysis-id');
    
    await expect(page.getByText(/Analysis Pipeline/i)).toBeVisible();
    await expect(page.getByText(/fetch/i)).toBeVisible();
    await expect(page.getByText(/parse/i)).toBeVisible();
    await expect(page.getByText(/semantic/i)).toBeVisible();
    await expect(page.getByText(/verify/i)).toBeVisible();
  });

  test('should display findings section', async ({ page }) => {
    await page.goto('/analysis/test-analysis-id');
    
    await expect(page.getByText(/Findings/i)).toBeVisible();
  });

  test('should show severity badges', async ({ page }) => {
    await page.goto('/analysis/test-analysis-id');
    
    // Check that severity badges are rendered
    const findings = page.locator('[class*="rounded-full"]');
    await expect(findings.first()).toBeVisible();
  });

  test('should display fix suggestions', async ({ page }) => {
    await page.goto('/analysis/test-analysis-id');
    
    await expect(page.getByText(/Suggested fix/i).first()).toBeVisible();
  });
});

test.describe('Analysis Flow', () => {
  test('should navigate from analyses list to detail', async ({ page }) => {
    await page.goto('/dashboard/analyses');
    
    // Click on the first analysis link
    const analysisLink = page.locator('a[href^="/analysis/"]').first();
    if (await analysisLink.isVisible()) {
      await analysisLink.click();
      await expect(page.getByText(/Analysis Pipeline/i)).toBeVisible();
    }
  });
});

import { test, expect } from '@playwright/test';

test.describe('Landing Page', () => {
  test('should display the hero section', async ({ page }) => {
    await page.goto('/');
    
    await expect(page.getByRole('heading', { name: /CodeVerify/i })).toBeVisible();
    await expect(page.getByText(/AI-powered code review/i)).toBeVisible();
  });

  test('should have navigation links', async ({ page }) => {
    await page.goto('/');
    
    await expect(page.getByRole('link', { name: /Features/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /Pricing/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /Docs/i })).toBeVisible();
  });

  test('should navigate to login page', async ({ page }) => {
    await page.goto('/');
    
    await page.getByRole('link', { name: /Sign In/i }).click();
    await expect(page).toHaveURL('/login');
  });

  test('should display feature cards', async ({ page }) => {
    await page.goto('/');
    
    await expect(page.getByText(/Formal Verification/i)).toBeVisible();
    await expect(page.getByText(/AI Analysis/i)).toBeVisible();
    await expect(page.getByText(/GitHub Integration/i)).toBeVisible();
  });
});

test.describe('Login Page', () => {
  test('should display GitHub login button', async ({ page }) => {
    await page.goto('/login');
    
    await expect(page.getByRole('link', { name: /Continue with GitHub/i })).toBeVisible();
  });

  test('should have terms and privacy links', async ({ page }) => {
    await page.goto('/login');
    
    await expect(page.getByRole('link', { name: /Terms of Service/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /Privacy Policy/i })).toBeVisible();
  });
});

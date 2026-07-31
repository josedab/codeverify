import { test, expect } from '@playwright/test';

test.describe('Landing Page', () => {
  test('should display the hero section', async ({ page }) => {
    await page.goto('/');

    await expect(
      page.getByRole('heading', {
        name: 'AI-Powered Code Review with Formal Verification',
      }),
    ).toBeVisible();
    await expect(
      page.getByText(/Catch bugs, security vulnerabilities, and logical errors/i),
    ).toBeVisible();
  });

  test('should have navigation links', async ({ page }) => {
    await page.goto('/');

    const navigation = page.getByRole('navigation');
    await expect(navigation.getByRole('link', { name: 'Docs' })).toHaveAttribute(
      'href',
      '/docs',
    );
    await expect(
      navigation.getByRole('link', { name: 'Sign In' }),
    ).toHaveAttribute('href', '/login');
  });

  test('should navigate to login page', async ({ page }) => {
    await page.goto('/');

    await page.getByRole('link', { name: 'Sign In' }).click();
    await expect(page).toHaveURL('/login');
  });

  test('should display feature cards', async ({ page }) => {
    await page.goto('/');

    await expect(
      page.getByRole('heading', { name: 'Hybrid AI + Formal Methods' }),
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'GitHub-Native Integration' }),
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Actionable Results' }),
    ).toBeVisible();
  });
});

test.describe('Login Page', () => {
  test('should display GitHub login button', async ({ page }) => {
    await page.goto('/login');

    await expect(
      page.getByRole('link', { name: 'Continue with GitHub' }),
    ).toBeVisible();
  });

  test('should have terms and privacy links', async ({ page }) => {
    await page.goto('/login');

    await expect(
      page.getByRole('link', { name: 'Terms of Service' }),
    ).toHaveAttribute('href', '/terms');
    await expect(
      page.getByRole('link', { name: 'Privacy Policy' }),
    ).toHaveAttribute('href', '/privacy');
  });
});

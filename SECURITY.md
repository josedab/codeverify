# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security seriously at CodeVerify. If you discover a security vulnerability, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report vulnerabilities through one of these channels:

1. **GitHub Security Advisories** (Preferred)
   - Go to [Security Advisories](https://github.com/codeverify/codeverify/security/advisories/new)
   - Create a new private security advisory

2. **Email**
   - Send details to: security@codeverify.dev
   - Use our PGP key for sensitive information (available at https://codeverify.dev/.well-known/security.txt)

### What to Include

Please include the following in your report:

- **Description**: A clear description of the vulnerability
- **Impact**: What an attacker could achieve by exploiting this
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Affected Versions**: Which versions are affected
- **Suggested Fix**: If you have ideas on how to fix it (optional)

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt within 48 hours
2. **Initial Assessment**: We will provide an initial assessment within 5 business days
3. **Regular Updates**: We will keep you informed of our progress
4. **Resolution**: We aim to resolve critical vulnerabilities within 30 days
5. **Credit**: We will credit you in our security advisories (unless you prefer to remain anonymous)

### Bug Bounty

We currently do not have a formal bug bounty program. However, we recognize and appreciate security researchers who help us improve CodeVerify's security.

## Security Best Practices

When using CodeVerify:

### API Keys & Secrets
- Never commit API keys or secrets to version control
- Use environment variables for sensitive configuration
- Rotate API keys regularly
- Use the minimum required permissions

### GitHub App Installation
- Only install on repositories that need analysis
- Review the permissions requested by the app
- Regularly audit app installations

### Self-Hosted Deployments
- Keep all dependencies up to date
- Use TLS for all connections
- Implement network segmentation
- Enable audit logging
- Follow the principle of least privilege

## Security Features

CodeVerify includes several security features:

- **Webhook Signature Verification**: All GitHub webhooks are verified using HMAC signatures
- **JWT Authentication**: Secure token-based authentication with expiration
- **Rate Limiting**: Protects against abuse and DoS attacks
- **Input Sanitization**: Prevents XSS and injection attacks
- **Security Headers**: CSP, HSTS, X-Frame-Options, etc.
- **Encrypted Data**: Sensitive data encrypted at rest and in transit

## Disclosure Policy

We follow a coordinated disclosure process:

1. Reporter submits vulnerability privately
2. We acknowledge and investigate
3. We develop and test a fix
4. We release the fix
5. We publish a security advisory
6. Reporter may publish their findings after advisory is public

We request a 90-day disclosure window for critical vulnerabilities to allow users time to update.

## Contact

- Security Team: security@codeverify.dev
- General Inquiries: hello@codeverify.dev

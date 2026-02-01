import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
        'getting-started/first-analysis',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'concepts/how-it-works',
        'concepts/verification-types',
        'concepts/findings',
        'concepts/trust-scores',
      ],
    },
    {
      type: 'category',
      label: 'Configuration',
      items: [
        'configuration/overview',
        'configuration/repository-config',
        'configuration/verification-settings',
        'configuration/ai-settings',
        'configuration/custom-rules',
      ],
    },
    {
      type: 'category',
      label: 'Integrations',
      items: [
        'integrations/github',
        'integrations/gitlab',
        'integrations/bitbucket',
        'integrations/vscode',
        'integrations/ci-cd',
        'integrations/slack-teams',
      ],
    },
    {
      type: 'category',
      label: 'Verification',
      items: [
        'verification/overview',
        'verification/null-safety',
        'verification/array-bounds',
        'verification/integer-overflow',
        'verification/division-by-zero',
        'verification/debugger',
      ],
    },
    {
      type: 'category',
      label: 'Advanced Features',
      items: [
        'advanced/copilot-interceptor',
        'advanced/monorepo-support',
        'advanced/test-generation',
        'advanced/proof-carrying-prs',
        'advanced/team-learning',
        'advanced/semantic-diff',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/overview',
        'api/authentication',
        'api/analyses',
        'api/findings',
        'api/webhooks',
      ],
    },
    {
      type: 'category',
      label: 'Self-Hosting',
      items: [
        'self-hosting/overview',
        'self-hosting/docker',
        'self-hosting/kubernetes',
        'self-hosting/configuration',
      ],
    },
    {
      type: 'category',
      label: 'Resources',
      items: [
        'resources/troubleshooting',
        'resources/faq',
        'resources/comparison',
        'resources/changelog',
      ],
    },
  ],
};

export default sidebars;

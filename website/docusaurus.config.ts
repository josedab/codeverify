import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'CodeVerify',
  tagline: 'AI-powered code review with formal verification',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://docs.codeverify.dev',
  baseUrl: '/',

  organizationName: 'codeverify',
  projectName: 'codeverify',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  themes: [
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        language: ['en'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/codeverify/codeverify/tree/main/website/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/codeverify/codeverify/tree/main/website/',
          blogTitle: 'CodeVerify Blog',
          blogDescription: 'Updates, tutorials, and insights about AI-powered code verification',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/codeverify-social-card.png',
    metadata: [
      {name: 'keywords', content: 'code review, formal verification, AI, static analysis, GitHub, Z3, security'},
      {name: 'twitter:card', content: 'summary_large_image'},
    ],
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    announcementBar: {
      id: 'v0_3_release',
      content: '🚀 CodeVerify v0.3.0 is here with Copilot Trust Score, Team Learning Mode, and more! <a href="/blog/v0.3-release">Learn more</a>',
      backgroundColor: '#6366f1',
      textColor: '#ffffff',
      isCloseable: true,
    },
    navbar: {
      title: 'CodeVerify',
      logo: {
        alt: 'CodeVerify Logo',
        src: 'img/logo.svg',
        srcDark: 'img/logo-dark.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/docs/api-reference/overview',
          label: 'API',
          position: 'left',
        },
        {to: '/blog', label: 'Blog', position: 'left'},
        {
          href: 'https://github.com/codeverify/codeverify',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Getting Started', to: '/docs/getting-started/quick-start'},
            {label: 'Configuration', to: '/docs/configuration/overview'},
            {label: 'Verification', to: '/docs/verification/overview'},
            {label: 'API Reference', to: '/docs/api-reference/overview'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub Discussions', href: 'https://github.com/codeverify/codeverify/discussions'},
            {label: 'Discord', href: 'https://discord.gg/codeverify'},
            {label: 'Twitter', href: 'https://twitter.com/codeverify'},
          ],
        },
        {
          title: 'More',
          items: [
            {label: 'Blog', to: '/blog'},
            {label: 'GitHub', href: 'https://github.com/codeverify/codeverify'},
            {label: 'Changelog', to: '/docs/changelog'},
            {label: 'Roadmap', href: 'https://github.com/codeverify/codeverify/blob/main/ROADMAP.md'},
          ],
        },
        {
          title: 'Legal',
          items: [
            {label: 'Privacy Policy', href: 'https://codeverify.dev/privacy'},
            {label: 'Terms of Service', href: 'https://codeverify.dev/terms'},
            {label: 'Security', href: 'https://github.com/codeverify/codeverify/blob/main/SECURITY.md'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} CodeVerify. MIT License.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'yaml', 'json', 'typescript', 'diff'],
    },
  } satisfies Preset.ThemeConfig,
};
export default config;

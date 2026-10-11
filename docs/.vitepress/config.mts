import { defineConfig } from 'vitepress'

const base = (process.env.BASE_PATH || '/') as `/${string}/` | '/'

export default defineConfig({
  title: 'aeonq',
  description: 'Semantics-Preserving Quantum Amplitude Estimation Across Heterogeneous Quantum SDKs',
  base,
  cleanUrls: true,
  srcExclude: ['README.md'],
  head: [
    // Square icon: the 187x45 wordmark would be squashed into the tab's square.
    ['link', { rel: 'icon', type: 'image/png', sizes: '40x40', href: '/favicon.png' }]
  ],
  markdown: {
    math: true,
  },
  themeConfig: {
    logo: '/logo.png',
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Quickstart', link: '/quickstart' },
      { text: 'Capabilities', link: '/capabilities' },
      { text: 'Algorithms', link: '/algorithms' },
      {
        text: 'Core & Testing',
        items: [
          { text: 'Semantic IR & Synthesis', link: '/intermediate_representation' },
          { text: 'Conformance & Mutation Testing', link: '/conformance_testing' },
          { text: 'Translator Fidelity & Oracles', link: '/translator_fidelity' },
        ],
      },
    ],
    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Overview', link: '/' },
          { text: 'Quickstart Guide', link: '/quickstart' },
        ],
      },
      {
        text: 'Architecture & Theory',
        items: [
          { text: 'Capabilities & Adapters', link: '/capabilities' },
          { text: 'Algorithms & Guarantees', link: '/algorithms' },
          { text: 'Semantic IR & Synthesis', link: '/intermediate_representation' },
        ],
      },
      {
        text: 'Verification',
        items: [
          { text: 'Conformance & Mutation Testing', link: '/conformance_testing' },
          { text: 'Translator Fidelity & Oracles', link: '/translator_fidelity' },
        ],
      },
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/username/aeonq' },
    ],
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026 aeonq contributors',
    },
    search: {
      provider: 'local',
    },
  },
})

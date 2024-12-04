import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Topic Modeling Docs",
  description: "Documentation for Topic Modeling Project",
  base: '/', // Assuming this is your repository name
  themeConfig: {
    logo: '/logo.png',
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'API', link: '/api/overview' }
    ],
    sidebar: {
      '/guide/': [
        {
          text: 'Introduction',
          items: [
            { text: 'Getting Started', link: '/guide/getting-started' },
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Preprocessing', link: '/guide/preprocessing' },
            { text: 'Filtering', link: '/guide/filtering' }, // Fixed path
            { text: 'Training', link: '/guide/training' }, // Fixed path
            { text: 'Visualization', link: '/guide/visualization' }, // Fixed path
            { text: 'TopicModeler', link: '/guide/TopicModeler' }, // Fixed path
            { text: 'Coherences', link: '/guide/coherences' }, // Fixed path
            // Fixed path
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Overview', link: '/api/overview' },
            { text: 'Model Evaluator', link: '/api/model-evaluator' },
            { text: 'Usage Examples', link: '/api/usage-examples' },
          ]
        }
      ]
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/venvis/topic_modelling' }
    ],
    search: {
      provider: 'local',
      options: {
        detailedView: true,
        searchFields: ['title', 'text', 'headers'],
        maxResults: 10
      }
    }
  },
  head: [
    [
      'link',
      { rel: 'icon', type: 'image/x-icon', href: '../icon.ico' } // Replace with your favicon path
    ]
  ],
  ignoreDeadLinks: true
})
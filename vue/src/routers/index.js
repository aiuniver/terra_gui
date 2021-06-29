export default [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/Datasets')
  },
  {
    path: '/datasets',
    name: 'Datasets',
    component: () => import('@/views/Datasets'),
  },
  {
    path: '/modeling',
    name: 'Modeling',
    component: () => import('@/views/Modeling'),
  },
  {
    path: '/training',
    name: 'Training',
    component: () => import('@/views/Training'),
  },
]

export default [
  {
    path: '/',
    name: 'Home',
    redirect: '/datasets'
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
  {
    path: '/deploy',
    name: 'Deploy',
    component: () => import('@/views/Deploy'),
  },
]

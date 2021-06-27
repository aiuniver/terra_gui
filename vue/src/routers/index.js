export default [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/Home')
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
]

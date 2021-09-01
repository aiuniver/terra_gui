import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router)

const router = new Router({
  mode: "history",
  routes: [
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
      path: '/cascades',
      name: 'Cascades',
      component: () => import('@/views/Cascades'),
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
    {
      path: '/marking',
      name: 'Marking',
      component: () => import('@/views/Marking'),
    },
  ]
})

// router.beforeEach((to, from, next) => {
//   next()
// })

export default router
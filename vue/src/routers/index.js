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
})

router.beforeEach((to, from, next) => {
  // console.log(router.app.$store)
  // if (router?.app?.$store?.getters['projects/getProject']?.datatset) {
  //   next()
  // } else {
  //   next('/dataset')
  // }
  // if (to.matched.some(route => route.meta.requiresAuth)) {
  //   if (store.state.authStatus)
  //     next()
  //   else
  //     next({name: 'home'})
  // } else {
    next()
  // }
})

export default router
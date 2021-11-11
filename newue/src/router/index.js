import Vue from 'vue'
import Router from 'vue-router'
Vue.use(Router)

const router = new Router({
  mode: "history",
  routes: [
    {
      path: '/',
      name: 'Home',
      meta: { 
        access: true,
        text: '' 
      },
      redirect: '/datasets'
    },
    {
      path: '/datasets',
      name: 'Datasets',
      meta: { 
        access: true,
        text: '' 
      },
      component: () => import('@/views/Datasets'),
    },
    {
      path: '/test',
      name: 'Test',
      meta: { 
        title: process.env.NODE_ENV === 'development' ? 'Test' : null,
        access: true,
        text: `Для перехода на страницу деплоя необходимо загрузить датасет.`, 
      },
      component: () => import('@/views/Test'),
    },
    {
      path: "*",
      name: '404',
      meta: { 
        access: true,
        text: '' 
      },
      component: () => import('@/views/404')
    }
  ]
})

// router.beforeEach((to, from, next) => {
  // console.log(to)
  // console.log(from)
  // next()
// })

export default router
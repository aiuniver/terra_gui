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
        title: 'Данные',
        access: true,
        text: '' 
      },
      component: () => import('@/views/Datasets'),
    },
    {
      path: '/marking',
      name: 'Marking',
      meta: { 
        title: process.env.NODE_ENV === 'development' ? 'Разметка' : null,
        access: true,
        text: `Для перехода на страницу разметки необходимо загрузить датасет.`, 
      },
      component: () => import('@/views/Marking'),
    },
    {
      path: '/modeling',
      name: 'Modeling',
      meta: { 
        title: 'Проектирование',
        access: true,
        text: '' 
      },
      component: () => import('@/views/Modeling'),
    },
    {
      path: '/training',
      name: 'Training',
      meta: { 
        title: 'Обучение',
        access: false,
        text: `Для перехода на страницу обучения необходимо загрузить датасет.`, 
      },
      component: () => import('@/views/Training'),
    },
    {
      path: '/cascades',
      name: 'Cascades',
      meta: { 
        title: 'Каскады', //process.env.NODE_ENV === 'development' ? 'Каскады' : null,
        access: false,
        text: `Для перехода на страницу каскадов необходимо загрузить датасет.`, 
      },
      component: () => import('@/views/Cascades'),
    },
    {
      path: '/deploy',
      name: 'Deploy',
      meta: { 
        title: 'Деплой',
        access: false,
        text: `Для перехода на страницу деплоя необходимо загрузить датасет.`, 
      },
      component: () => import('@/views/Deploy'),
    },
    {
      path: '/servers',
      name: 'Servers',
      meta: { 
        title: 'Серверы',
        access: true,
        text: '', 
      },
      component: () => import('@/views/Servers'),
    },
    {
      path: '/profile',
      name: 'Profile',
      meta: { 
        title: null,
        access: true,
        text: ``, 
      },
      component: () => import('@/views/Profile'),
    },
    {
      path: '/projects',
      name: 'Projects',
      meta: { 
        title: process.env.NODE_ENV === 'development' ? 'Проекты' : null,
        access: true,
        text: ``, 
      },
      component: () => import('@/views/Projects'),
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
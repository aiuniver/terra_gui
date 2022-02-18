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
        parent: 'data',
        title: 'Датасеты',
        access: true,
        text: ''
      },
      component: () => import('@/views/new/datasets/Choice'),
    },
    {
      path: '/create',
      name: 'Create',
      meta: {
        parent: 'data',
        title: 'Создание датасета',
        access: true,
        text: ''
      },
      component: () => import('@/views/new/datasets/Create'),
    },
    {
      path: '/marking',
      name: 'Marking',
      meta: {
        parent: 'data',
        title: 'Разметка',
        access: true,
        text: ``,
      },
      component: () => import('@/views/Marking'),
    },
    {
      path: '/view',
      name: 'View',
      meta: {
        parent: 'data',
        title: 'Просмотр датасета',
        access: true,
        text: ''
      },
      component: () => import('@/views/new/datasets/View'),
    },
    {
      path: '/modeling',
      name: 'Modeling',
      meta: {
        parent: 'modeling',
        title: 'Проектирование',
        access: true,
        text: ''
      },
      component: () => import('@/views/new/design/Modeling'),
    },
    {
      path: '/cascades',
      name: 'Cascades',
      meta: {
        parent: 'modeling',
        title: 'Каскады', //process.env.NODE_ENV === 'development' ? 'Каскады' : null,
        access: true,
        text: `Для перехода на страницу каскадов необходимо загрузить датасет.`,
      },
      component: () => import('@/views/new/design/Cascades'),
    },

    {
      path: '/training',
      name: 'Training',
      meta: {
        parent: 'training',
        title: 'Обучение',
        access: false,
        text: `Для перехода на страницу обучения необходимо загрузить датасет.`,
      },
      component: () => import('@/views/new/completion/Training'),
    },
    {
      path: '/deploy',
      name: 'Deploy',
      meta: {
        parent: 'training',
        title: 'Деплой',
        access: true,
        text: `Для перехода на страницу деплоя необходимо загрузить датасет.`,
      },
      component: () => import('@/views/new/completion/Deploy'),
    },
    {
      path: '/servers',
      name: 'Servers',
      meta: {
        parent: 'training',
        title: 'Серверы',
        access: true,
        text: '',
      },
      component: () => import('@/views/new/completion/Servers'),
    },
    {
      path: '/profile',
      name: 'Profile',
      meta: {
        // parent: 'project',
        title: 'Профиль',
        access: true,
        text: ``,
      },
      component: () => import('@/views/new/Profile'),
    },
    {
      path: '/projects',
      name: 'Projects',
      meta: {
        parent: 'project',
        title: 'Проекты',
        access: true,
        text: ``,
      },
      component: () => import('@/views/new/project/Projects'),
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
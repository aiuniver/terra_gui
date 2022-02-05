import Vue from 'vue'
import Router from 'vue-router'
const originalPush = Router.prototype.push;
Router.prototype.push = function push (location) {
  return originalPush.call(this, location).catch(err => err)
};

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
      redirect: '/datasets/choice'
    },
    {
      path: '/datasets/',
      component: () => import('@/views/datasets/index'),
      meta: {
        title: 'Датасеты',
        access: true,
        text: ''
      },
      children: [
        {
          path: 'choice',
          name: 'Choice',
          component: () => import('@/views/datasets/Choice'),
          meta: {
            title: 'Датасеты',
            access: true,
            text: ''
          },
        },
        {
          path: 'create',
          name: 'Create',
          component: () => import('@/views/datasets/Create'),
          meta: {
            title: 'Создание датасета',
            access: true,
            text: ''
          },
        },
        {
          path: 'marker',
          name: 'Marker',
          component: () => import('@/views/datasets/Marking'),
          meta: {
            title: 'Разметка',
            access: true,
            text: ''
          },
        },
        {
          path: 'view',
          name: 'View',
          component: () => import('@/views/datasets/View'),
          meta: {
            title: 'Просмотр датасета',
            access: true,
            text: ''
          },
        },
      ]
    },
    {
      path: '/design/',
      component: () => import('@/views/design/index'),
      meta: {
        title: 'Проектирование',
        access: true,
        text: ''
      },
      children: [
        {
          path: 'model',
          name: 'Model',
          component: () => import('@/views/design/Modeling'),
          meta: {
            title: 'Проектирование',
            access: true,
            text: ''
          },
        },
        {
          path: 'cascade',
          name: 'Cascade',
          component: () => import('@/views/design/Cascades'),
          meta: {
            title: 'Каскады',
            access: true,
            text: ''
          },
        },
      ]
    },
    {
      path: '/completion/',
      component: () => import('@/views/datasets/index'),
      meta: {
        title: 'Обучение',
        access: true,
        text: ''
      },
      children: [
        {
          path: 'training',
          name: 'Training',
          component: () => import('@/views/completion/Training.vue'),
          meta: {
            title: 'Обучение',
            access: true,
            text: ''
          },
        },
        {
          path: 'deploy',
          name: 'Deploy',
          component: () => import('@/views/completion/Deploy.vue'),
          meta: {
            title: 'Деплой',
            access: true,
            text: ''
          },
        },
      ]
    },
    {
      path: '/project/',
      name: 'Projects',
      meta: {
        title: 'Проекты',
        access: true,
        text: `Для перехода на страницу деплоя необходимо загрузить датасет.`,
      },
      component: () => import('@/views/project/index'),
      children: [
        {
          path: 'training',
          name: 'Training',
          component: () => import('@/views/project/Projects'),
          meta: {
            title: 'Проекты',
            access: true,
            text: ''
          },
        },
      ]
    },
    // {
    //   path: '/test',
    //   name: 'Test',
    //   meta: {
    //     title: process.env.NODE_ENV === 'development' ? 'Test' : null,
    //     access: true,
    //     text: `Для перехода на страницу деплоя необходимо загрузить датасет.`,
    //   },
    //   component: () => import('@/views/other/Test'),
    // },
    {
      path: "*",
      name: '404',
      meta: {
        access: true,
        text: ''
      },
      component: () => import('@/views/other/404')
    }
  ]
})

// router.beforeEach((to, from, next) => {
// console.log(to)
// console.log(from)
// next()
// })

export default router
const newRouter = [
    {
        path: '/new',
        name: 'Home',
        meta: {
            access: true,
            text: ''
        },
        redirect: '/new/datasets'
    },

    {
        path: '/new/datasets',
        name: 'Datasets',
        meta: {
            parent: 'data',
            title: 'Датасеты',
            access: true,
            text: ''
        },
        component: () => import('@/views/Datasets'),
    },
    {
        path: '/new/create',
        name: 'Create',
        meta: {
            parent: 'data',
            title: 'Создание датасета',
            access: true,
            text: ''
        },
        component: () => import('@/views/Datasets'),
    },
    {
        path: '/new/marking',
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
        path: '/new/modeling',
        name: 'View',
        meta: {
            parent: 'data',
            title: 'Просмотр датасета',
            access: true,
            text: ''
        },
        component: () => import('@/views/Modeling'),
    },
    {
        path: '/new/modeling',
        name: 'Modeling',
        meta: {
            parent: 'modeling',
            title: 'Проектирование',
            access: true,
            text: ''
        },
        component: () => import('@/views/Modeling'),
    },
    {
        path: '/new/cascades',
        name: 'Cascades',
        meta: {
            parent: 'modeling',
            title: 'Каскады', //process.env.NODE_ENV === 'development' ? 'Каскады' : null,
            access: true,
            text: `Для перехода на страницу каскадов необходимо загрузить датасет.`,
        },
        component: () => import('@/views/Cascades'),
    },

    {
        path: '/new/training',
        name: 'Training',
        meta: {
            parent: 'training',
            title: 'Обучение',
            access: false,
            text: `Для перехода на страницу обучения необходимо загрузить датасет.`,
        },
        component: () => import('@/views/Training'),
    },
    {
        path: '/new/deploy',
        name: 'Deploy',
        meta: {
            parent: 'training',
            title: 'Деплой',
            access: true,
            text: `Для перехода на страницу деплоя необходимо загрузить датасет.`,
        },
        component: () => import('@/views/Deploy'),
    },
    {
        path: '/new/servers',
        name: 'Servers',
        meta: {
            parent: 'training',
            title: 'Серверы',
            access: true,
            text: '',
        },
        component: () => import('@/views/Servers'),
    },
    {
        path: '/new/profile',
        name: 'Profile',
        meta: {
            parent: 'project',
            title: 'Профиль',
            access: true,
            text: ``,
        },
        component: () => import('@/views/Profile'),
    },
    {
        path: '/new/projects',
        name: 'Projects',
        meta: {
            parent: 'project',
            title: 'Проекты',
            access: true,
            text: ``,
        },
        component: () => import('@/views/Projects'),
    },

]

export { newRouter }
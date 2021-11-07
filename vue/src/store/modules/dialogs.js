export default {
  namespaced: true,
  state: () => ({
  }),
  mutations: {
  },
  actions: {
    async confirm (_, { ctx, title = 'Внимание!', content, width = 300 }) {
      try {
        return await ctx.$Modal.confirm({ title, content, width });
      } catch (error) {
        console.log(error)
        return null
      }
    },
    async trining ({ dispatch }, { ctx, page }) {
      const res = await dispatch('projects/get', {}, { root: true });
      if (res) {
        const { project: { training: { state: { status } } } } = res
        // console.log(status)
        if (['addtrain', 'training'].includes(status)) {
          const res = await dispatch('confirm', {
            ctx,
            content: `Для загрузки ${page} остановите обучение. Перейти на страницу обучения ?`
          });
          if (res == 'confirm') ctx.$router.push('/training');
          return false
        } else if (['trained', 'stopped'].includes(status)) {
          const res = await dispatch('confirm', {
            ctx,
            content: `Для загрузки ${page} необходимо сбросить обучение. Сбросить обучения ?`
          });
          if (res == 'confirm') {
            // ctx.$router.push('/training');
            await dispatch('trainings/clear', {}, {root: true})
            return true
          } else {
            return false
          }
        }
        return true
      }
    }
  },
  getters: {
    getErrors: ({ errors }) => errors,
  },
};

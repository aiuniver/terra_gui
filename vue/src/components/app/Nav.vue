<template>
  <nav class="nav">
    <ul class="nav__menu">
      <template v-for="(route, i) in items">
        <li :class="['nav__menu--item', { active: $route.path === route.path }]" :key="i" @click="nav(route)">
          {{ route.title }}
        </li>
      </template>
    </ul>
  </nav>
</template>

<script>
import { mapGetters } from 'vuex';
export default {
  computed: {
    ...mapGetters({
      project: 'projects/getProject',
    }),
    isTrain() {
      const state = this.$store.getters['trainings/getStatus'];
      return ['addtrain', 'training'].includes(state);
    },
    items() {
      return this.$router.options.routes
        .filter(item => item?.meta?.title)
        .map(item => {
          return {
            title: item.meta.title,
            path: item.path,
            access: item.meta.access,
            text: item.meta.text,
          };
        });
    },
  },
  methods: {
    async message({ text }, showClose) {
      try {
        const data = await this.$Modal.alert({
          title: 'Предупреждение!',
          width: 300,
          content: text,
          showClose,
          okText: 'Загрузить датасет',
        });
        if (data === 'confirm') {
          if (this.$route.path !== '/datasets') {
            this.$router.push('/datasets');
          }
        }
      } catch (error) {
        console.log(error);
      }
    },
    async nav({ path, access, text }) {
      if (!this.project.dataset && access === false) {
        this.message({ text }, true);
      } else {
        if (this.$route.path !== path) {
          this.$router.push(path);
        }
      }
      this.$store.dispatch('messages/resetProgress');
    },
  },
};
</script>

<style lang="scss" scoped>
.nav {
  width: 100%;
  // position: fixed;
  // left: 0;
  // top: 54px;
  z-index: 700;
  &__menu {
    list-style: none;
    display: -webkit-box;
    display: -moz-box;
    display: -ms-flexbox;
    display: -webkit-flex;
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    justify-content: flex-start;
    align-content: flex-start;
    align-items: center;

    &--item {
      color: #6c7883;
      padding-right: 1px;
      display: block;
      padding: 0 40px;
      line-height: 40px;
      font-size: 0.875rem;
      text-decoration: none;
      border-radius: 5px 5px 0 0;
      transition: color 0.3s ease-in-out;
      white-space: nowrap;
      user-select: none;
      cursor: pointer;
    }
    .active {
      background-color: #17212b;
      color: #ffffff;
    }
    :hover {
      color: #ffffff;
    }
  }
}
</style>
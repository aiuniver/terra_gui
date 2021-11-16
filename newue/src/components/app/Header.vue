<template>
  <div class="header">
    <div class="header__left">
      <d-dropdown>
        <template v-slot:activator="{ on }">
          <img src="@/assets/images/logo.png" width="28" height="28" alt="Logo" v-on="on" />
        </template>
        <ul class="list">
          <template v-for="({ title, path, children }, i) of menu">
            <hr v-if="path === '/project/'" :key="'hr' + i" />
            <li
              :class="['list__item', { 'list__item--active': active(path) }]"
              :key="title"
              @click="to(path, children)"
            >
              {{ title }}
            </li>
          </template>
        </ul>
      </d-dropdown>
    </div>
    <div class="app__center">
      <ul class="nav">
        <li
          v-for="({ title, path }, i) of children"
          :class="['nav__item', { 'nav__item--active': active(path) }]"
          :key="i"
          @click="to(path)"
        >
          {{ title }}
        </li>
      </ul>
    </div>
    <div class="header__right"></div>
  </div>
</template>

<script>
export default {
  name: 'app-header',
  components: {},
  methods: {
    to(path, children) {
      children = children?.[0]?.path || '';
      path = children ? path + children : path;
      if (path !== this.$route.fullPath) {
        this.$router.push(path);
      }
    },
    active(path) {
      const fullPath = this.$route.fullPath || '';
      return fullPath.includes(path);
    },
  },
  computed: {
    routes() {
      return this.$router?.options?.routes || [];
    },
    menu() {
      return this.routes
        .filter(item => item?.meta?.title)
        .map(item => {
          return {
            title: item.meta.title,
            path: item.path,
            access: item.meta.access,
            text: item.meta.text,
            children: item.children,
          };
        });
    },
    children() {
      const children = this.menu.find(i => this.$route.fullPath.includes(i.path))?.children || [];
      return children
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
};
</script>

<style lang="scss" scoped>
.header {
  width: 100%;
  height: 52px;
  padding: 0 10px;
  display: flex;
  align-items: center;
  border-bottom: 1px solid var(--color-border);
}
.nav {
  position: relative;
  display: flex;
  align-items: center;
  user-select: none;
  height: 100%;

  &__item {
    padding: 15px 30px;
    color: var(--color-gray-blue);
    &--active,
    &:hover {
      color: var(--color-light-blue);
    }
  }
}
</style>

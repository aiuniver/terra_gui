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
    <div class="header__right">
      <ul>
        <li>
          <i class="ci-icon ci-pie_chart_50" @click="theme()"/>
        </li>
        <li>
          <i class="ci-icon ci-notification_outline" />
        </li>
        <li>
          <i class="ci-icon ci-user_circle" />
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
import { mapActions } from 'vuex';
export default {
  name: 'app-header',
  components: {},
  methods: {
    ...mapActions({
      theme: 'themes/changeTheme',
    }),
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
  background-color: #17212B;
  &__right {
    height: 100%;
    margin-left: auto;
    align-items: center;
    position: relative;
    &::before {
      content: '';
      display: block;
      position: absolute;
      top: 11px;
      height: 30px;
      width: 1px;
      background-color: var(--color-dark-gray);
    }
    ul {
      height: 100%;
      li {
        cursor: pointer;
        display: inline-block;
        height: 100%;
        // margin: 0 13px;
        padding: 13px;
        i {
          font-size: 20px;
          color: var(--color-gray-blue);
        }
      }
    }
  }
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

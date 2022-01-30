<template>
  <div class="header">
    <div class="header__left">
      <d-dropdown>
        <template v-slot:activator="{ on }">
          <img src="@/assets/images/logo.png" width="28" height="28" alt="Logo" v-on="on" />
        </template>
        <ul class="list">
          <template v-for="({ title, meta }, i) of menu">
            <hr v-if="meta === 'project'" :key="'hr' + i" />
            <li :class="['list__item', { 'list__item--active': activeRoot(meta) }]" :key="title" @click="onSelect(meta)">
              {{ title }}
            </li>
          </template>
        </ul>
      </d-dropdown>
    </div>
    <div class="app__center">
      <ul class="nav">
        <li v-for="({ title, path }, i) of children" :class="['nav__item', { 'nav__item--active': active(path) }]" :key="i" @click="to(path)">
          {{ title }}
        </li>
      </ul>
    </div>
    <div class="header__right">
      <ul>
        <li>
          <i class="ci-icon ci-pie_chart_50" @click="theme()" />
        </li>
        <li>
          <i class="ci-icon ci-notification_outline" />
        </li>
        <li>
          <i class="ci-icon ci-user_circle" @click="to('/new/profile')"/>
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
  data: () => ({
    menu: [
      { title: 'Данные', path: '', meta: 'data' },
      { title: 'Проектирование', path: '', meta: 'modeling' },
      { title: 'Обучение', path: '', meta: 'training' },
      { title: 'Проекты', path: '', meta: 'project' },
    ],
    select: 'data', // this.menu[0]
  }),
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
    onSelect(meta) {
      this.select = meta;
      const router = this.children[0];
      if (router?.path) this.$router.push(router.path);
    },
    activeRoot(meta) {
      return Boolean(this.select === meta);
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
    children() {
      const routes = this.routes.filter(r => r.meta.parent === this.select) || [];
      console.log(routes);
      return routes.map(item => {
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
  background-color: #17212b;
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
  hr {
    margin: 0;
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
    cursor: pointer;
    &--active,
    &:hover {
      color: var(--color-light-blue);
    }
  }
}
</style>

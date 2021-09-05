<template>
  <nav class="nav">
    <ul class="nav__menu">
      <template v-for="({ title, path }, i) in items">
        <li
          :class="['nav__menu--item', { active: $route.path === path }]"
          :key="i"
          @click="nav(path)"
        >
          {{ title }}
        </li>
      </template>
    </ul>
  </nav>
</template>

<script>
import { mapGetters } from "vuex";
export default {
  data: () => ({
    items: [
      { title: "Данные", path: "/datasets", access: true },
      { title: "Разметка", path: "/marking", access: true },
      { title: "Проектирование", path: "/modeling", access: true },
      { title: "Обучение", path: "/training", access: false },
      { title: "Каскады", path: "/cascades", access: false },
      { title: "Деплой", path: "/deploy", access: true },
    ],
  }),
  computed: {
    ...mapGetters({
      project: "projects/getProject",
    }),
  },
  methods: {
    nav(path) {
      if (!this.project.dataset) {
        if (this.items.find(item => item.path === path)?.access) {
          if (this.$route.path !== path) {
            this.$router.push(path);
          }
          return;
        }
        const text = {
          "/modeling": "редактирования модели",
          "/training": "обучения",
          "/deploy": "деплоя",
        };
        const self = this
        this.$Modal.alert({
          title: "Предупреждение!",
          width: 300,
          content: `Для ${text[path]} необходимо загрузить датасет.`,
          // showClose: false,
          maskClosable: true,
          okText: "Загрузить датасет",
          callback: function (action) {
            console.log(action);
            if (self.$route.path !== '/datasets') {
              self.$router.push('/datasets');
            }
          },
        });
      } else {
        if (this.$route.path !== path) {
          this.$router.push(path);
        }
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.nav {
  width: 100%;
  position: fixed;
  left: 0;
  top: 54px;
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
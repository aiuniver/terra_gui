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
      { title: "Данные", path: "/datasets" },
      { title: "Проектирование", path: "/modeling" },
      { title: "Обучение", path: "/training" },
      { title: "Деплой", path: "/deploy" },
    ],
  }),
  computed: {
    ...mapGetters({
      project: "projects/getProject",
    }),
  },
  methods: {
    nav(path) {
      if (this.project.dataset) {
        if (this.$route.path !== path) {
          this.$router.push(path);
        }
      } else {
        if (this.$route.path !== "/datasets") {
          this.$router.push("/datasets");
        }
        this.$Modal.alert({
          title: "Предупреждение!",
          width: 300,
          content: "Для редактирования модели необходимо загрузить датасет.",
          callback: function (action) {
            console.log(action)
          },
        });
      }
    },
  },
  created() {
    console.log(this.$router.history.current.fullPath === "/datasets");
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
    -webkit-box-direction: normal;
    -moz-box-direction: normal;
    -webkit-box-orient: horizontal;
    -moz-box-orient: horizontal;
    -webkit-flex-direction: row;
    -ms-flex-direction: row;
    flex-direction: row;
    -webkit-flex-wrap: nowrap;
    -ms-flex-wrap: nowrap;
    flex-wrap: nowrap;
    -webkit-box-pack: start;
    -moz-box-pack: start;
    -webkit-justify-content: flex-start;
    -ms-flex-pack: start;
    justify-content: flex-start;
    -webkit-align-content: flex-start;
    -ms-flex-line-pack: start;
    align-content: flex-start;
    -webkit-box-align: center;
    -moz-box-align: center;
    -webkit-align-items: center;
    -ms-flex-align: center;
    align-items: center;

    &--item {
      color: #6C7883;
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
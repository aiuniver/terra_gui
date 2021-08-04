<template>
  <div class="toolbar">
    <ul class="toolbar__menu">
      <template v-for="({ title, name, disabled, icon, hr }, i) of lists">
        <li :key="i" :disabled="disabled" @click.prevent="click(name)">
          <span :title="title" :class="icon"></span>
        </li>
        <hr v-if="hr" :key="`hr_${i}`" />
      </template>
    </ul>
  </div>
</template>

<script>
export default {
  name: "Toolbar",
  data: () => ({
    lists: [
      {
        title: "Загрузить модель",
        name: "load",
        disabled: false,
        icon: "icon-model-load",
      },
      {
        title: "Сохранить модель",
        name: "save",
        disabled: false,
        icon: "icon-model-save",
      },
      {
        title: "Валидация",
        name: "validation",
        disabled: false,
        icon: "icon-model-validation",
      },
      {
        title: "Очистить",
        name: "clear",
        disabled: false,
        icon: "icon-clear-model",
        hr: true,
      },
      {
        title: "Входящий слой",
        name: "input",
        disabled: true,
        icon: "icon-layer-input",
      },
      {
        title: "Промежуточный слой",
        name: "middle",
        disabled: false,
        icon: "icon-layer-middle",
      },
      {
        title: "Исходящий слой",
        name: "output",
        disabled: true,
        icon: "icon-layer-output",
        hr: true,
      },
      {
        title: "Код на Keras",
        name: "keras",
        disabled: true,
        icon: "icon-keras-code",
      },
    ],
  }),
  methods: {
    click(event) {
      this.$store.dispatch("modeling/setToolbarEvent", { event });
    },
  },
};
</script>

<style lang="scss" scoped>
.toolbar {
  z-index: 10;
  width: 41px;
  flex-shrink: 0;
  position: relative;
  border-right: #0e1621 1px solid;
  &__menu {
    padding: 10px 0;
    list-style: none;
    li {
      &[disabled="disabled"] > span {
        opacity: 0.1;
        cursor: default;
      }
      span {
        display: block;
        padding: 8px;
        user-select: none;
        cursor: pointer;
      }
      span:after {
        display: block;
        content: "";
        width: 24px;
        height: 24px;
        background-position: center;
        background-repeat: no-repeat;
        background-size: contain;
      }
    }
  }
}

hr {
  border: none;
  color: #0e1621;
  background-color: #0e1621;
  height: 1px;
  margin: 10px 0px;
}
</style>
<template>
  <div class="toolbar project-modeling-toolbar">
    <div class="wrapper">
      <ul class="menu-section models">
        <template v-for="({ title, name, disabled, icon, hr }, i) of lists">
          <li
            :key="i"
            :disabled="disabled"
            @click.prevent="$emit('click', name)"
          >
            <span :title="title" :class="icon"></span>
          </li>
          <hr v-if="hr" :key="`hr_${i}`" />
        </template>
      </ul>
    </div>
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
        name: "save_model",
        disabled: true,
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
};
</script>

<style scoped>
hr {
  border: none;
  color: #0e1621;
  background-color: #0e1621;
  height: 1px;
  margin: 10px 0px;
}
</style>
<template>
<div class="toolbar project-modeling-toolbar">
    <div class="wrapper">
        <ul class="menu-section models">
            <li data-type="load" @click="OpenModalWindow">
                <span title="Загрузить модель" class="icon-model-load"></span>
            </li>
            <li data-type="save_model" disabled="disabled">
                <span title="Сохранить модель" class="icon-model-save"></span>
            </li>
            <li data-type="save" class="hidden" disabled="disabled">
                <span class="icon-model-save"></span>
            </li>
            <li data-type="validation">
                <span title="Валидация" class="icon-model-validation"></span>
            </li>
            <li data-type="clear">
                <span title="Очистить" class="icon-clear-model"></span>
            </li>
        </ul>
        <ul class="menu-section layers">
            <li data-type="input" disabled="disabled">
                <span title="Входящий слой" class="icon-layer-input"></span>
            </li>
            <li data-type="middle">
                <span title="Промежуточный слой" class="icon-layer-middle"></span>
            </li>
            <li data-type="output" disabled="disabled">
                <span title="Исходящий слой" class="icon-layer-output"></span>
            </li>
        </ul>
        <ul class="menu-section others">
            <li data-type="keras" disabled="disabled">
                <span title="Код на Keras" class="icon-keras-code"></span>
            </li>
        </ul>
    </div>
</div>
</template>

<script>
export default {
  name: "Toolbar",
  props: {
    load_model_flag: Boolean
  },
  methods: {
    OpenModalWindow(){
      this.load_model_flag = !this.load_model_flag
      this.$emit("OpenModalWindow", this.load_model_flag)
    }
  }

}
</script>

<style scoped>

</style>
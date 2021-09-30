<template>
  <div class="toolbar">
    <ul class="toolbar__menu">
      <li class="toolbar__menu--item" @click.prevent="click('load')" title="Загрузить модель">
        <i class="t-icon icon-model-load"></i>
      </li>
      <li class="toolbar__menu--item" @click.prevent="click('save')" title="Сохранить модель">
        <i class="t-icon icon-model-save"></i>
      </li>
      <li class="toolbar__menu--item" @click.prevent="click('validation')" title="Валидация">
        <i class="t-icon icon-model-validation"></i>
      </li>
      <li class="toolbar__menu--item" @click.prevent="click('clear')" title="Очистить">
        <i class="t-icon icon-model-clear"></i>
      </li>
      <hr />
      <li class="toolbar__menu--item" @click.prevent="click('input')" title="Входящий слой">
        <i class="t-icon icon-layer-input-casc"></i>
      </li>
      <li class="toolbar__menu--item" @click.prevent="click('model')" title="Model">
        <i class="t-icon icon-layer-model"></i>
      </li>
      <li class="toolbar__menu--item" @click.prevent="click('function')" title="Function">
        <i class="t-icon icon-layer-function"></i>
      </li>
      <li class="toolbar__menu--item" @click.prevent="click('custom')" title="Custom">
        <i class="t-icon icon-layer-custom"></i>
      </li>
      <li class="toolbar__menu--item" @click.prevent="click('output')" title="Исходящий слой">
        <i class="t-icon icon-layer-output"></i>
      </li>
      <hr />
    </ul>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
export default {
  name: 'Toolbar',
  data: () => ({}),
  computed: {
    ...mapGetters({
      blocks: 'cascades/getBlocks',
      project: 'projects/getProject',
    }),
  },
  methods: {
    click(event, idDisebled) {
      if (!idDisebled) {
        this.$emit('actions', event);
      }
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
    &--item {
      width: 40px;
      height: 40px;
      width: 40px;
      height: 40px;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor: pointer;
      &[disabled='disabled'] {
        opacity: 0.1;
        cursor: default;
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
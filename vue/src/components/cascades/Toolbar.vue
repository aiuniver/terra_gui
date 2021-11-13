<template>
  <div class="toolbar">
    <ul class="toolbar__menu">
      <li :class="['toolbar__menu--item', { disabled: false }]" @click="click($event, 'load')" title="Загрузить каскад">
        <i class="t-icon icon-model-load"></i>
      </li>
      <!-- <li
        :class="['toolbar__menu--item', { disabled: isSave }]"
        @click="click($event, 'save')"
        title="Сохранить каскад"
      >
        <i class="t-icon icon-model-save"></i>
      </li> -->
      <li :class="['toolbar__menu--item', { disabled: isClear }]" @click="click($event, 'clear')" title="Очистить">
        <i class="t-icon icon-model-clear"></i>
      </li>
      <hr />
      <li :class="['toolbar__menu--item', { disabled: false }]" @click.prevent="click($event, 'InputData')" title="Входящий слой">
        <i class="t-icon icon-layer-input-casc"></i>
      </li>
      <li class="toolbar__menu--item" @click.prevent="click($event, 'Model')" title="Model">
        <i class="t-icon icon-layer-model"></i>
      </li>
      <li class="toolbar__menu--item" @click.prevent="click($event, 'Function')" title="Function">
        <i class="t-icon icon-layer-function"></i>
      </li>
      <li class="toolbar__menu--item" @click.prevent="click($event, 'Custom')" title="Custom">
        <i class="t-icon icon-layer-custom"></i>
      </li>
      <li :class="['toolbar__menu--item', { disabled: isOutput }]" @click.prevent="click($event, 'OutputData')" title="Исходящий слой">
        <i class="t-icon icon-layer-output"></i>
      </li>
      <hr />
      <li :class="['toolbar__menu--item', { disabled: false }]" @click="click($event, 'validation')" title="Валидация">
        <i class="t-icon icon-model-validation"></i>
      </li>
      <li :class="['toolbar__menu--item', { disabled: false }]" @click="click($event, 'start')" title="Запустить">
        <i class="t-icon icon-model-play"></i>
      </li>
      <li :class="['toolbar__menu--item', { disabled: false }]" @click="click($event, 'save')" title="Сохранить каскад">
        <i class="t-icon icon-model-save"></i>
      </li>
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
      errors: 'cascades/getErrorsBlocks',
      project: 'projects/getProject',
      status: 'cascades/getStatus',
    }),
    isSave() {
      return this.isKeras;
      // || this.status.isUpdate
    },
    isClear() {
      return !this.blocks.length;
    },
    isKeras() {
      const errors = Object.values(this.errors);
      return !errors.length || errors.filter(item => item !== null).length;
    },
    isValidation() {
      const blocks = this.blocks.map(item => item.group);
      return !(blocks.includes('input') && blocks.includes('output'));
    },
    isInput() {
      return !!this.blocks.find(item => item.group === 'InputData') && !!this.project?.dataset;
    },
    isOutput() {
      return !!this.blocks.find(item => item.group === 'OutputData') && !!this.project?.dataset;
    },
  },
  methods: {
    click({ currentTarget }, comm) {
      const classList = [...currentTarget?.classList] || [];
      // console.log(classList);
      if (!classList.includes('disabled')) {
        this.$emit('actions', comm);
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
      &.disabled {
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
<template>
  <div class="toolbar">
    <ul class="toolbar__menu">
      <li :class="['toolbar__menu--item', { disabled: false }]" @click="click($event, 'load')" title="Загрузить модель">
        <i class="t-icon icon-model-load"></i>
      </li>
    </ul>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
export default {
  name: 'Toolbar',
  data: () => ({
  }),
  computed: {
    ...mapGetters({
      blocks: 'modeling/getBlocks',
      errors: 'modeling/getErrorsBlocks',
      project: 'projects/getProject',
      status: 'modeling/getStatus',
    }),
    isSave() {
      return this.isKeras
      // || this.status.isUpdate
    },
    isClear() {
      return !this.blocks.length 
    },
    isKeras() {
      const errors = Object.values(this.errors)
      return !errors.length || errors.filter(item => item !== null).length
    },
    isValidation() {
      const blocks = this.blocks.map(item => item.group)
      return !(blocks.includes('input') && blocks.includes('output'))
    },
    isInput() {
      return !!this.blocks.find(item => item.group === 'input') && !!this.project?.dataset;
    },
    isOutput() {
      return !!this.blocks.find(item => item.group === 'output') && !!this.project?.dataset;
    },
  },
  methods: {
    click({ currentTarget }, comm) {
      const classList = [...currentTarget?.classList] || [];
      console.log(classList);
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
  height: 100%;
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
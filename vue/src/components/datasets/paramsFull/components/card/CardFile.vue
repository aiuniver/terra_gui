<template>
  <div class="t-card-file" :style="bc">
    <div v-if="id" class="t-card-file__header" :style="bg">{{title}}</div>
    <div :class="['t-card-file__body', 'icon-file-' + type]"></div>
    <div class="t-card-file__footer">{{ label }}</div>
  </div>
</template>

<script>
export default {
  name: 't-card-file',
  props: {
    label: String,
    type: String,
    id: Number,
  },
  computed: {
    selectInputData() {
      return this.$store.getters['datasets/getInputDataByID'](this.id) || {}
    },
    title() {
      const card = this.selectInputData
      return card.name || card.layer === 'input' ? 'Входные данные ' + card.id : 'Выходные данные ' + card.id
    },
    color() {
      return this.selectInputData.color || ''
    },
    bg() {
      return { backgroundColor: this.id ? this.color : '' };
    },
    bc() {
      return { borderColor: this.id ? this.color : '' };
    },
  },
};
</script>

<style lang="scss" scoped>
.t-card-file {
  flex: 0 0 152px;
  position: relative;
  width: 152px;
  height: 152px;
  border: 1px solid #6c7883;
  box-sizing: border-box;
  border-radius: 4px;
  background-color: #17212b;
  margin: 0 5px;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  &__footer,
  &__header {
    position: relative;
    height: 24px;
    background-color: #242f3d;
    width: 100%;
    border-radius: 0 0 3px 3px;
    font-family: Open Sans;
    font-style: normal;
    font-weight: normal;
    font-size: 12px;
    line-height: 16px;
  }
  &__header {
    top: 0;
    border-radius: 3px 3px 0 0;
    padding: 2px 6px 4px 6px;
  }
  &__body {
    height: 100%;
    width: 100%;
    display: inline-block;
    background-repeat: no-repeat;
    background-position: 50% 40%;
    background-size: 39px 39px;
  }
  &__footer {
    bottom: 0;
    border-radius: 0 0 3px 3px;
    padding: 4px 6px 2px 6px;
  }
}

</style>
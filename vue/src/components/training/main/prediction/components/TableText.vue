<template>
  <div class="t-predict-text">
    <p class="t-predict-text__text">{{ value }}</p>
  </div>
</template>

<script>
export default {
  name: 'table-text',
  props: {
    value: {
      type: String,
      default: '',
    },
    color_mark: {
      type: Array,
      default: () => [],
    },
  },
};
</script>

<style lang="scss" scoped>
.t-predict-text {
  display: flex;
  height: 100%;
  justify-content: center;
  align-items: center;
  width: 400px;
  padding: 5px;
  &__text {
    text-align: center;
  }
}
</style>
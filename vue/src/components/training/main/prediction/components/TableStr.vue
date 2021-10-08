<template>
  <div class="t-predict-test">
    <p :class="['t-predict-test__text', color]">
      {{ value }}
    </p>
  </div>
</template>

<script>
export default {
  name: 'TableStr',
  props: {
    value: {
      type: [String, Number],
      default: '',
    },
    color_mark: {
      type: String,
      default: '',
    },
    tagsColor: {
      type: String,
      default: '',
    },
  },
  computed: {
    color() {
      return `t-predict-test__text--${this.color_mark}`
    }
  }
};
</script>

<style lang="scss" scoped>
.t-predict-test {
  display: flex;
  height: 100%;
  justify-content: center;
  align-items: center;
  &__text {
    &--success {
      color: green;
    }
    &--wrong {
      color: orange;
    }
  }
}
</style>

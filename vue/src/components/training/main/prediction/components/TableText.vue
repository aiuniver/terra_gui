<template>
  <p
    v-if="type === 'str' || type === 'number'"
    :class="['table-prediction__text', { [`table-prediction__text--marked-${color}`]: color }]"
  >
    {{ text }}
  </p>
</template>

<script>
export default {
  name: 'TableText',
  props: {
    data: {
      type: [Object, Array],
      default: () => {},
    },
  },
  computed: {
    text() {
      return this.data?.data || '';
    },
    type() {
      return this.data?.type || null;
    },
    color() {
      return this.data?.color_mark || '';
    },
  },
};
</script>

<style lang="scss" scoped>
.table-prediction {
  &__text {
    &--marked-success {
      color: green;
    }
    &--marked-wrong {
      color: orange;
    }
  }
}
</style>

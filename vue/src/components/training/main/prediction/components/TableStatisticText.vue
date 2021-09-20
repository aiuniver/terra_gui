<template>
  <div class="table-prediction">
    <p :class="['table-prediction__text']">
      {{ value }}
    </p>
    <div :class="['table-prediction__color', { [`table-prediction__color--marked-${color}`]: color }]"></div>
  </div>
</template>

<script>
export default {
  name: 'TableStatisticText',
  props: {
    data: {
      type: [Object, Array],
      default: () => {},
    },
    value: String,
  },
  computed: {
    color() {
      return this.data?.color_mark || '';
    },
    percent() {
      return this.data?.value || '';
    },
  },
};
</script>

<style lang="scss" scoped>
.table-prediction {
  display: flex;
  &__text {
    margin-right: 10px;
    &--marked-success {
      color: green;
    }
    &--marked-wrong {
      color: orange;
    }
  }
  &__color {
    width: 77px;
    height: 24px;
    border-radius: 4px;
    &--marked-success {
      background: green;
    }
    &--marked-wrong {
      background: orange;
    }
  }
}
</style>

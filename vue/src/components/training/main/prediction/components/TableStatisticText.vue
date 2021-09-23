<template>
  <div class="t-predict-table">
    <div v-if="index === '1'" :class="['t-predict-table__text']">
      {{ name }}
    </div>
    <div class="t-predict-table__empty"></div>
    <div :class="['t-predict-table__color', color]">{{ value }}</div>
  </div>
</template>

<script>
export default {
  name: 't-predict-table',
  props: {
    name: {
      type: String,
      default: '',
    },
  },
  computed: {
    value() {
      return this.data.name || '';
    },
    color() {
      console.log(this.index);
      return `t-predict-table__color--${this.color_mark}`;
    },
  },
};
</script>

<style lang="scss" scoped>
.t-predict-table {
  display: flex;
  flex-direction: column;
  height: 100%;
  position: relative;
  &__text {
    position: absolute;
    width: 100%;
    top: 0;
    flex: 0 0 22px;
    display: flex;
    justify-content: center;
    background-color: #242f3d;
    &--marked-success {
      color: green;
    }
    &--marked-wrong {
      color: orange;
    }
  }
  &__empty {
    flex: 1 1 auto;
  }
  &__color {
    flex: 0 0 80px;
    width: 77px;
    height: 24px;
    border-radius: 4px;
    display: flex;
    justify-content: center;
    &--marked-success {
      background: green;
    }
    &--marked-wrong {
      background: orange;
    }
  }
}
</style>

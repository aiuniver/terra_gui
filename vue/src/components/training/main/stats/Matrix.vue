<template>
  <div class="t-matrix">
    <div class="t-matrix__gradient">
      <div class="colors"></div>
      <div class="values">
        <div v-for="(item, idx) in stepValues" :key="idx">{{ item }}</div>
      </div>
    </div>
    <div class="t-matrix__table">
      <div class="t-matrix__label--top">{{ graph_name }}</div>
      <div class="t-matrix__grid--wrapper">
        <div class="legend--left">
          <div v-for="val in labels" :key="val">{{ val }}</div>
        </div>
        <div
          class="t-matrix__grid"
          :style="{ gridTemplate: `repeat(${labels.length}, 40px) / repeat(${labels.length}, 40px)` }"
        >
          <div
            class="t-matrix__grid--item"
            v-for="(item, i) in values"
            :key="'col_' + i"
            :style="{ background: getColor(percent[i]) }"
          >
            {{ `${item}, ${percent[i]}%` }}
          </div>
        </div>
        <div class="legend--bottom">
          <div v-for="val in labels" :key="val">{{ val }}</div>
        </div>
        <div class="t-matrix__label--left">{{ y_label }}</div>
      </div>
      <div class="t-matrix__label--bottom">{{ x_label }}</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-matrix',
  props: {
    id: Number,
    task_type: String,
    graph_name: String,
    x_label: String,
    y_label: String,
    labels: Array,
    data_array: Array,
    data_percent_array: Array,
  },
  data: () => ({}),
  computed: {
    values() {
      return [].concat(...this.data_array);
    },
    percent() {
      return [].concat(...this.data_percent_array);
    },
    stepValues() {
      return [4, 3, 2, 1, 0].map(item => (this.max / 4) * item).reverse();
    },
    max() {
      return Math.round(this.maxValue / 100) * 100;
    },
    maxValue() {
      return Math.max(...this.values);
    },
  },
  methods: {
    getColor(value) {
      return `rgb(${0 + value}, ${50 + value}, ${150 + value})`;
    },
  },
};
</script>

<style lang="scss" scoped>
.t-matrix {
  display: flex;
  align-items: center;
  color: #a7bed3;
  width: fit-content;
  gap: 30px;
  &__gradient {
    display: flex;
    gap: 5px;
    margin-bottom: 10px;
    height: 100%;
    .colors {
      background: linear-gradient(180deg, #003b7f 0%, #54a3ff 100%);
      border-radius: 4px;
      width: 24px;
      flex-shrink: 0;
    }
    .values {
      display: flex;
      flex-direction: column-reverse;
      justify-content: space-between;
      font-size: 9px;
      line-height: 14px;
    }
  }
  &__table {
    position: relative;
  }
  &__grid {
    display: grid;
    border-radius: 4px;
    overflow: hidden;
    &--wrapper {
      display: flex;
      flex-wrap: wrap;
      max-width: 500px;
      margin: 10px auto 25px;
      justify-content: flex-end;
      position: relative;
      .legend--left,
      .legend--bottom {
        font-size: 9px;
        line-height: 11px;
      }
      .legend--left {
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        * {
          flex-basis: 10%;
          display: flex;
          justify-content: flex-end;
          align-items: center;
          padding: 0 5px;
        }
      }
      .legend--bottom {
        display: flex;
        justify-content: flex-end;
        position: absolute;
        right: 0;
        top: 100%;
        * {
          flex: 0 0 40px;
          // word-wrap: break-word;
          text-align: center;
          max-width: 40px;
          padding: 3px 0;
        }
      }
    }
    &--item {
      font-size: 10px;
      line-height: 14px;
      display: flex;
      text-align: center;
      justify-content: center;
      align-items: center;
    }
  }
  &__label--top {
    text-align: center;
    font-size: 14px;
    line-height: 17px;
    font-weight: 600;
  }
  &__label--bottom,
  &__label--left {
    text-align: center;
    font-weight: 600;
    font-size: 12px;
    line-height: 16px;
  }
  &__label--left {
    position: absolute;
    /* top: 50%; */
    bottom: 0px;
    left: -50px;
    transform: rotate(-90deg) translateX(50%);
  }
}
</style>
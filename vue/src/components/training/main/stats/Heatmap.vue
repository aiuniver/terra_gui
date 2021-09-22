<template>
  <div class="t-heatmap">
    <div class="t-heatmap__scale">
      <div class="t-heatmap__scale--gradient"></div>
      <div class="t-heatmap__scale--values">
        <span class="value" v-for="(item, idx) in stepValues" :key="idx">{{ item }}</span>
      </div>
    </div>
    <div class="t-heatmap__body">
      <scrollbar :ops="ops">
        <div class="t-heatmap__wrapper">
          <p class="t-heatmap__title">{{ graph_name }}</p>
          <div class="t-heatmap__x-label">{{ x_label }}</div>
          <div class="t-heatmap__y-label">{{ y_label }}</div>
          <div class="t-heatmap__grid--y-labels">
            <span v-for="(item, idx) in labels" :key="idx">{{ item }}</span>
          </div>
          <div class="t-heatmap__grid--x-labels">
            <span v-for="(item, idx) in labels" :key="idx" :title="item">{{ item }}</span>
          </div>
          <div
            class="t-heatmap__grid"
            :style="{ gridTemplate: `repeat(${data_array.length}, 40px) / repeat(${data_array.length}, 40px)` }"
          >
            <div
              class="t-heatmap__grid--item"
              v-for="(item, i) in values"
              :key="'col_' + i"
              :style="{ background: getColor(percent[i]) }"
              :title="`${item} / ${percent[i]}%`"
            >
              {{ `${item}` }}
              <br />
              {{ `${percent[i]}%` }}
            </div>
          </div>
        </div>
      </scrollbar>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-heatmap',
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
  data: () => ({
    ops: {
      scrollPanel: {
        scrollingX: true,
        scrollingY: false,
      },
    },
  }),
  computed: {
    values() {
      return [].concat(...this.data_array);
    },
    percent() {
      return [].concat(...this.data_percent_array);
    },
    averageVal() {
      return this.values.reduce((prev, cur) => prev + cur) / this.values.length;
    },
    stepValues() {
      return [4, 3, 2, 1, 0].map(item => (this.max / 4) * item);
    },
    max() {
      return Math.round(this.maxValue / 100) * 100;
    },
    maxValue() {
      return Math.max(...this.values);
    },
  },
  methods: {
    getColor(val) {
      const light = 66 - (val / 100) * 41;
      return `hsl(212, 100%, ${light}%)`;
    },
  },
};
</script>

<style lang="scss" scoped>
.t-heatmap {
  display: flex;
  justify-content: flex-start;
  margin: 35px 0 70px;
  gap: 35px;
  width: 100%;
  &__body {
    overflow: hidden;
    position: relative;
    width: 100%;
  }
  &__title {
    color: #a7bed3;
    font-size: 14px;
    line-height: 17px;
    font-weight: 600;
    position: absolute;
    bottom: calc(100% + 10px);
    left: 50%;
    transform: translate(-40%);
    white-space: nowrap;
  }
  &__x-label,
  &__y-label {
    position: absolute;
    font-size: 12px;
    line-height: 16px;
    font-weight: 600;
    color: #a7bed3;
  }
  &__x-label {
    top: 100%;
    left: 50%;
    transform: translate(-50%, 200%);
  }
  &__y-label {
    bottom: 50%;
    right: 100%;
    transform: rotate(-90deg) translateY(200%);
    white-space: nowrap;
  }
  &__wrapper {
    display: flex;
    gap: 5px;
    position: relative;
  }
  &__grid {
    display: grid;
    font-size: 10px;
    text-align: center;
    border-radius: 4px;
    overflow: hidden;
    &--item {
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    &--y-labels {
      flex-direction: column;
      width: fit-content;
      overflow: hidden;
      * {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        flex-basis: 40px;
      }
    }
    &--x-labels {
      position: absolute;
      top: 100%;
      justify-content: flex-end;
      width: 100%;
      * {
        text-overflow: ellipsis;
        overflow: hidden;
        padding: 0 2px;
        width: 40px;
        text-align: center;
      }
    }
    &--y-labels,
    &--x-labels {
      display: flex;
      color: #a7bed3;
      font-size: 9px;
      line-height: 14px;
    }
  }
  &__scale {
    display: flex;
    height: 100%;
    gap: 5px;
    &--values {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      color: #a7bed3;
      font-size: 9px;
      line-height: 14px;
    }
    &--gradient {
      background: linear-gradient(180deg, #003b7f 0%, #54a3ff 100%);
      border-radius: 4px;
      width: 24px;
    }
  }
}
</style>
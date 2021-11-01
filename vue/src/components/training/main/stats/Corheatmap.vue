<template>
  <div class="t-cor-heatmap" :class="['t-cor-heatmap', { 't-cor-heatmap--small': isSmallMap }]">
    <div class="t-cor-heatmap__scale" ref="scale">
      <div class="t-cor-heatmap__scale--gradient"></div>
      <div class="t-cor-heatmap__scale--values">
        <span class="value" v-for="(item, idx) in stepValues" :key="idx">{{ item }}</span>
      </div>
      <div class="t-cor-heatmap__y-label">{{ y_label }}</div>
    </div>
    <div class="t-cor-heatmap__grid--y-labels" ref="label">
      <span v-for="(item, idx) in labels" :key="idx">{{ item }}</span>
    </div>
    <div class="t-cor-heatmap__body" :style="{ maxWidth: bodyWidth }">
      <p class="t-cor-heatmap__title">{{ graph_name }}</p>
      <div class="t-cor-heatmap__x-label">{{ x_label }}</div>
      <scrollbar :ops="ops">
        <div class="t-cor-heatmap__wrapper">
          <div class="t-cor-heatmap__grid" :style="gridTemplate">
            <div class="t-cor-heatmap__grid--x-labels">
              <span v-for="(item, idx) in labels" :key="idx" :title="item">{{ item }}</span>
            </div>
            <div
              class="t-cor-heatmap__grid--item"
              v-for="(item, i) in values"
              :key="'col_' + i"
              :style="{ background: getColor(item) }"
              :title="`${item}`"
            >
              {{ `${item}` }}
            </div>
          </div>
        </div>
      </scrollbar>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-corheatmap',
  props: {
    id: Number,
    task_type: String,
    graph_name: String,
    x_label: String,
    y_label: String,
    labels: Array,
    data_array: Array,
    data_percent_array: [Array, String],
  },
  data: () => ({
    ops: {
      scrollPanel: {
        scrollingX: true,
        scrollingY: false,
      },
    },
    width: null,
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
      return [1, 0.5, 0, -0.5, -1];
    },
    max() {
      return Math.ceil(this.maxValue / 1) * 1;
    },
    maxValue() {
      return Math.max(...this.values);
    },
    bodyWidth() {
      return `calc(100% - ${this.width - 10}px)`;
    },
    isSmallMap() {
      return this.data_array.length < 5;
    },
    gridTemplate() {
      const width = this.isSmallMap ? '80px' : '40px';
      return {
        gridTemplate: `repeat(${this.data_array.length}, ${width}) / repeat(${this.data_array.length}, ${width})`,
      };
    },
  },
  methods: {
    getColor(value) {
      let blue, red, green = 0;
      if (value > 0) {
        blue = (value / 1) * 50;
        red = (value / 1) * 250;
        green = (value / 1) * 153;
      } else {
        blue = (value / -1) * 255;
        red = (value / -1) * 82;
        green = (value / -1) * 160;
      }
      return `rgb(${red}, ${green}, ${blue})`;
    },
  },
  mounted() {
    this.width = this.$refs.label.offsetWidth + this.$refs.scale.offsetWidth;
  },
};
</script>

<style lang="scss" scoped>
.t-cor-heatmap {
  display: flex;
  justify-content: flex-start;
  margin: 35px 0;
  gap: 5px;
  max-width: 100%;
  position: relative;
  &__body {
    position: relative;
    max-width: calc(100% - 60px);
  }
  &__title {
    color: #a7bed3;
    font-size: 14px;
    line-height: 17px;
    font-weight: 600;
    top: calc(-2em - 10px);
    position: absolute;
    left: 50%;
    transform: translate(-50%);
    width: 100%;
    text-align: center;
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
    transform: translate(-50%, 100%);
  }
  &__y-label {
    bottom: 50%;
    left: 30px;
    transform: rotate(-90deg);
    white-space: nowrap;
  }
  &__wrapper {
    display: flex;
    gap: 5px;
    position: relative;
    width: fit-content;
    padding-bottom: 30px;
  }
  &__grid {
    display: grid;
    font-size: 10px;
    text-align: center;
    border-radius: 4px;
    position: relative;
    &--item {
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    &--y-labels {
      flex-direction: column;
      width: fit-content;
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
  &.t-cor-heatmap--small {
    .t-cor-heatmap__grid {
      font-size: 14px;
    }
    .t-cor-heatmap__grid--x-labels,
    .t-cor-heatmap__grid--y-labels {
      font-size: 12px;
      line-height: 16px;
    }
    .t-cor-heatmap__grid--y-labels {
      * {
        flex-basis: 80px;
      }
    }
    .t-cor-heatmap__grid--x-labels * {
      width: 80px;
    }
  }
  &__scale {
    display: flex;
    gap: 5px;
    padding-right: 35px;
    padding-bottom: 30px;
    &--values {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      color: #a7bed3;
      font-size: 9px;
      line-height: 14px;
    }
    &--gradient {
      background: linear-gradient(180deg, #f99f35 0%, #030202 50%, #003b7f 100%);
      border-radius: 4px;
      width: 24px;
    }
  }
}
</style>
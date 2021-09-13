<template>
  <div class="t-graph">
    <p>{{ graph_name || '' }}</p>
    <div class="t-graph__wrapper">
      <div class="t-graph__x-label">{{ x_label }}</div>
      <div class="t-graph__y-label">{{ y_label }}</div>
      <div class="t-graph__values">
        <div v-for="(val, idx) in stepValues" :key="idx">{{ val }}</div>
      </div>
      <div class="t-graph__diagram">
        <div class="t-graph__diagram-item" v-for="(val, idx) in values" :key="idx">
          <span>{{ val }}</span>
          <div class="t-graph__diagram-fill" :style="{ height: `${((val / maxValue) * 100).toFixed()}%` }"></div>
          <div class="t-graph__diagram-label">{{ labels[idx] }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-graph',
  props: {
    id: Number,
    graph_name: String,
    x_label: String,
    y_label: String,
    plot_data: Array,
  },
  data: () => ({}),
  computed: {
    values() {
      console.log(this.plot_data[0]);
      return this.plot_data?.[0]?.values || [];
    },
    labels() {
      return this.plot_data?.[0]?.labels || [];
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
};
</script>

<style lang="scss" scoped>
.t-graph {
  width: max-content;
  margin-bottom: 25px;
  position: relative;
  &__wrapper {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 5px;
    max-width: 420px;
  }
  &__x-label {
    position: absolute;
    top: 50%;
    left: -25px;
    transform: rotate(-90deg);
    color: #a7bed3;
    font-size: 12px;
    line-height: 14px;
  }
  &__y-label {
    position: absolute;
    left: 50%;
    bottom: -30px;
    color: #a7bed3;
    font-size: 12px;
    line-height: 14px;
  }
  &__values {
    display: flex;
    color: #a7bed3;
    font-size: 9px;
    line-height: 14px;
    flex-direction: column;
    justify-content: space-between;
    text-align: right;
    div {
      transform: translateY(50%);
    }
  }
  &__diagram {
    background: #242f3d;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.25);
    border-radius: 4px;
    height: 220px;
    display: flex;
    gap: 18px;
    padding: 0 10px;
    flex: 0 0 auto;
    &-item {
      font-size: 9px;
      line-height: 14px;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      position: relative;
      text-align: center;
    }
    &-fill {
      background: #2a8cff;
      border-radius: 4px 4px 0px 0px;
      height: 100%;
      width: 21px;
    }
    &-label {
      position: absolute;
      right: 50%;
      transform: translateX(50%);
      top: 100%;
      max-width: 40px;
      // word-wrap: break-word;
      text-align: center;
      color: #a7bed3;
    }
  }
  p {
    color: #a7bed3;
    font-weight: 600;
    font-size: 14px;
    line-height: 17px;
    text-align: center;
    margin-bottom: 10px;
  }
}
</style>
<template>
  <div class="t-graphics">
    <div class="t-graphics__main">
      <Plotly :data="result" :layout="defLayout" :display-mode-bar="false" />
      <div class="t-graphics__title">{{ name }}</div>
    </div>
  </div>
</template>

<script>
import { Plotly } from 'vue-plotly';

export default {
  name: 'GraphicCard',
  components: {
    Plotly,
  },
  props: {
    data: Object,
  },
  data: () => ({
    name: null,
    line: {
      dash: 'dot',
      width: 4,
      color: '#89D764',
    },
    test: {
      plot_bgcolor: '#d3d3d3',
      paper_bgcolor: '#d3d3d3',
    },
    defLayout: {
      height: 350,
      width: 600,
      plot_bgcolor: '#fff0',
      paper_bgcolor: '#242F3D',
      showlegend: true,
      legend: {
        orientation: 'h',
      },
      font: {
        color: '#A7BED3',
        size: 9,
      },
      margin: {
        pad: 1,
        t: 5,
        r: 5,
        b: 30,
        l: 30,
      },
      xaxis: {
        gridcolor: '#17212B',
        showline: false,
        linecolor: '#A7BED3',
        linewidth: 1,
        title: {
          standoff: 0,
          font: { size: 10 },
        },
      },
      yaxis: {
        gridcolor: '#17212B',
        showline: false,
        linecolor: '#A7BED3',
        linewidth: 1,
        title: {
          font: { size: 10 },
        },
      },
    },
  }),
  computed: {
    result() {
      const obj = this.data;
      const arr = [];
      for (let key in obj) {
        arr.push({
          type: 'scatter',
          y: this.data[key] || [],
          x: this.data[key].map((_, i) => i) || [],
          name: key || null,
          mode: 'line',
          marker: { size: 10 },
        });
      }
      return arr
    },
  },
};
</script>

<style lang="scss" scoped>
.t-graphics {
  height: 400px;
  width: 630px;
  display: flex;
  flex-direction: column;
  user-select: none;
  &__title {
    position: absolute;
    top: 0;
    right: calc(50% - 30px);
    color: #7e7e7e;
  }
  &__main {
    flex: 1 1 auto;
    border: 1px solid #6c7883;
    border-radius: 4px;
    position: relative;
  }
  &__extra {
    //flex: 0 0 110px;
    //padding: 0 7px;
    //height: 230px;
    display: flex;
  }
  &__item {
    height: 40px;
    width: 100px;
    color: #a7bed3;
    background-color: #242f3d;
    border: 1px solid #6c7883;
    border-radius: 4px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease-in-out;
    &:hover {
      background-color: #65b9f4;
    }
    &--active {
      border-color: #65b9f4;
      color: #ebeeee;
    }
  }
}
</style>

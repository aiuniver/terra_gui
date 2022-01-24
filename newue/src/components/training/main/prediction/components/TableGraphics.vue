<template>
  <div class="t-graphics">
    <div class="t-graphics__main">
      <Plotly :data="data(plot_data)" :layout="layout(plot_data)" :display-mode-bar="false" />
      <div class="t-graphics__title">{{ plot_data.short_name }}</div>
    </div>
    <div class="t-graphics__extra">
      <scrollbar>
        <div
          v-for="chart of length"
          :key="chart.short_name"
          class="t-graphics__item"
          :title="chart.short_name"
          @click="name = chart.short_name"
        >
          <!-- {{ chart.short_name }} -->
        </div>
      </scrollbar>
    </div>
  </div>
</template>

<script>
import { Plotly } from 'vue-plotly';

export default {
  name: 't-graphics',
  components: {
    Plotly,
  },
  props: {
    value: Array,
  },
  data: () => ({
    name: null,
    defLayout: {
      height: 230,
      width: 400,
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
        showline: true,
        linecolor: '#A7BED3',
        linewidth: 1,
        title: {
          // text: 'Время',
          standoff: 0,
          font: { size: 10 },
        },
      },
      yaxis: {
        gridcolor: '#17212B',
        showline: true,
        linecolor: '#A7BED3',
        linewidth: 1,
        title: {
          // text: 'Значение',
          font: { size: 10 },
        },
      },
    },
  }),
  computed: {
    length() {
      return this.value
      // return this.value.filter(item => item.short_name !== this.name)
    },
    all() {
      return 0;
    },
    plot_data() {
      const name = this.name ?? (this.value?.[0]?.short_name || '')
      return this.value.find(item => item.short_name === name);
    },
  },
  methods: {
    layout({ x_label, y_label }) {
      const layout = this.defLayout;
      if (this.plot_data) {
        // layout.title.text = this.graph_name;
        layout.xaxis.title.text = x_label;
        layout.yaxis.title.text = y_label;
      }
      return layout;
    },
    data({ plot_data }) {
      return plot_data.map((el, i) => {
        return {
          type: 'scatter',
          x: el.x,
          y: el.y,
          mode: 'lines',
          name: el.label,
          line: {
            width: 2,
            color: (i === 0) ? '#89D764' : null,
          },
        };
      });
    },
  },
};
</script>

<style lang="scss" scoped>
.t-graphics {
  height: 230px;
  width: 400;
  display: flex;
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
    flex: 0 0 110px;
    padding: 0 7px;
    height: 230px;
  }
  &__item {
    height: 60px;
    width: 100px;
    margin: 0 0 5px 0;
    color: #ebeeee;
    background: #242f3d;
    border: 1px solid #6c7883;
    border-radius: 4px;
    background-repeat: no-repeat;
    background-image: url('/images/trainings/graphic_mini.png');
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease-in-out;
    &:hover {
      border-color: #65b9f4;
    }
  }
}
</style>

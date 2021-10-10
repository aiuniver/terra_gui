<template>
  <div class="t-bar">
    <p class="t-bar__title">{{ graph_name }}</p>
    <Plotly class="t-bar__plotly" :data="data" :layout="layout" :display-mode-bar="false" />
  </div>
</template>

<script>
import { Plotly } from 'vue-plotly';

export default {
  name: 't-bar',
  components: {
    Plotly,
  },
  props: {
    id: Number,
    task_type: String,
    graph_name: String,
    x_label: String,
    y_label: String,
    plot_data: Array,
    data_array: Array,
    labels: Array,
  },

  computed: {
    layout() {
      const layout = this.defLayout;
      if (this.plot_data) {
        // layout.title.text = this.graph_name;
        layout.xaxis.title.text = this.x_label;
        layout.yaxis.title.text = this.y_label;
      }
      return layout;
    },
    data() {
      return [
        {
          z: this.data_array,
          x: this.labels,
          y: ['Morning', 'Afternoon', 'Evening'],
          type: 'heatmap',
          hoverongaps: false,
          showscale: false,
          colorscale: [
            [0, '#003B7F'],
            [1, '#F99F35'],
          ],
        },
      ];
    },
  },
  data: () => ({
    defLayout: {
      width: 636,
      height: 352,
      plot_bgcolor: '#fff0',
      paper_bgcolor: '#242F3D',
      showlegend: true,
      bargap: 0.1,
      legend: {
        orientation: 'h',
        yanchor: 'top',
      },
      font: {
        color: '#A7BED3',
        size: 12,
      },
      margin: {
        pad: 1,
        t: 10,
        r: 20,
        b: 70,
        l: 60,
      },
      xaxis: {
        gridcolor: '#17212B',
        showline: true,
        linecolor: '#A7BED3',
        linewidth: 1,
        title: {
          text: 'Истинные значения',
          standoff: 0,
          font: { size: 12 },
        },
      },
      yaxis: {
        gridcolor: '#17212B',
        showline: true,
        linecolor: '#A7BED3',
        linewidth: 1,
        title: {
          text: 'Предсказанные значения',
          font: { size: 12 },
        },
      },
    },
  }),
};
</script>

<style lang="scss" scoped>
.t-bar {
  &__title {
    font-size: 14px;
    line-height: 17px;
    font-weight: 600;
    text-align: center;
    color: #a7bed3;
    margin: 10px auto;
  }
  &__plotly {
    border: 1px solid #6c7883;
    border-radius: 4px;
  }
}
</style>

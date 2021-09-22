<template>
  <div class="t-scatter">
    <p class="t-scatter__title">{{ graph_name }}</p>
    <Plotly class="t-scatter__plotly" :data="plot_data" :layout="layout" :display-mode-bar="false" />
  </div>
</template>

<script>
import { Plotly } from 'vue-plotly';

export default {
  name: 't-scatter',
  components: {
    Plotly,
  },
  props: {
    id: Number,
    task_type: String,
    graph_name: String,
    x_label: String,
    y_label: String,
    labels: Array,
    plot_data: Array,
  },
  data: () => ({
    // data: [{
    //     type: 'scatter',
    //     x: [1,2,3,4,5,6,7],
    //     y: [10,15,13,17,20,15,23],
    //     mode: 'markers',
    //     name: 'Регрессия',
    //     marker: { size: 10 }
    // }],
    layout: {
      width: 636,
      height: 352,
      plot_bgcolor: '#fff0',
      paper_bgcolor: '#242F3D',
      showlegend: true,
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
.t-scatter {
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

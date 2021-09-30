<template>
  <div class="t-char-temp">
    <div class="t-char-temp__overflow">
      <i class="t-icon icon-add-chart"></i>
      Добавить график
    </div>
    <div class="t-char-temp__header"></div>
    <div class="t-char-temp__main" v-if="graphicShow">
      <Plotly :data="data" :layout="layout" :display-mode-bar="false"></Plotly>
    </div>
  </div>
</template>

<script>
import { Plotly } from 'vue-plotly';
export default {
  name: 't-char-temp',
  components: {
    Plotly,
  },
  data: () => ({
    dataList: ['loss', 'Metrics'],
    metrics: ['Accuracy', 'CategoricalHinge', 'KLDivergence'],
    char: {
      graph_name: 'График',
      x_label: 'Эпоха',
      y_label: 'Значение',
      plot_data: [],
      progress_state: 'underfitting',
    },
    graphicShow: true,
    popMenuShow: false,
    defLayout: {
      autosize: true,
      margin: {
        l: 50,
        r: 20,
        t: 10,
        b: 35,
        pad: 0,
        autoexpand: true,
      },
      font: {
        color: '#A7BED3',
      },
      showlegend: true,
      legend: {
        y: -0.2,
        itemsizing: 'constant',
        orientation: 'h',
        font: {
          family: 'Open Sans',
          color: '#A7BED3',
        },
      },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      title: {
        text: '',
      },
      xaxis: {
        title: '',
        showgrid: true,
        zeroline: false,
        linecolor: '#A7BED3',
        gridcolor: '#0E1621',
        gridwidth: 1,
      },
      yaxis: {
        title: '',
        showgrid: true,
        zeroline: false,
        linecolor: '#A7BED3',
        gridcolor: '#0E1621',
        gridwidth: 1,
      },
    },
  }),
  computed: {
    layout() {
      const layout = this.defLayout;
      if (this.char) {
        // layout.title.text = this.char?.title || '';
        // layout.xaxis.title = this.char?.x_label || '';
        // layout.yaxis.title = this.char?.y_label || '';
      }
      return layout;
    },
    data() {
      const data = this.char.plot_data || [];
      const arr = data.map(({ epochs: x, values: y, mode = 'lines', label: name }) => {
        return { x, y, mode, name };
      });
      return arr;
    },
  },
  methods: {
    handleMenu(e) {
      console.log(e);
      this.popMenuShow = false;
    },
  },
  mounted() {
    console.log(this.char);
  },
};
</script>

<style lang="scss" scoped>
.normal {
  span {
    color: #65ca35;
  }
  .indicator {
    background: #65ca35;
  }
}
.undertraining {
  span {
    color: #f3d11d;
  }
  .indicator {
    background: #f3d11d;
  }
}
.retraining {
  span {
    color: #ca5035;
  }
  .indicator {
    background: #ca5035;
  }
}
.t-char-temp {
  width: 100%;
  height: 100%;
  // background: #242f3d;
  background: #242f3d3d;

  border-radius: 4px;
  // box-shadow: 0 2px 10px 0 rgb(0 0 0 / 25%);
  position: relative;
  order: 999;
  &__overflow {
    user-select: none;
    position: absolute;
    width: 100%;
    height: 100%;
    z-index: 5;
    // background-color: #242f3da1;
    display: flex;
    justify-content: center;
    align-items: center;
    font-style: normal;
    font-weight: bold;
    font-size: 14px;
    line-height: 24px;
    color: aliceblue;
    opacity: 0.5;
    transition: opacity 0.3s ease-in-out, color 0.3s ease-in-out;
    i {
      margin-right: 20px;
    }
    &:hover {
      opacity: 0.9;
    }
  }
  &__main {
    opacity: 0.3;
  }
  &__header {
    opacity: 0.3;
    padding: 12px;
    display: flex;
    justify-content: space-between;
    &-title {
      padding-left: 12px;
    }
    &-condition {
      border: 1px solid #6c7883;
      box-sizing: border-box;
      border-radius: 4px;
      display: flex;
      padding: 0 8px;
      margin-left: auto;
      span {
        font-size: 12px;
        line-height: 24px;
      }
      .indicator {
        width: 12px;
        height: 12px;
        border-radius: 7px;
        margin: 6px;
      }
    }
    &-additionally {
      padding-left: 10px;
      position: relative;
      cursor: pointer;
      z-index: 10;
    }
  }
}
</style>
<template>
  <div class="t-char" :style="order">
    <div class="t-char__header" v-click-outside="outside">
      <div class="t-char__header--roll">
        <i :class="['t-icon', 'icon-training-roll-down']" :title="'roll down'" @click="show"></i>
      </div>
      <div class="t-char__header--title">{{ graph_name }}</div>
      <div v-if="progress_state" :class="['t-char__header--condition', progress_state]">
        <span>{{ progress_state }}</span>
        <div class="indicator"></div>
      </div>
      <div class="t-char__header--additionally">
        <i
          :class="['t-icon', 'icon-training-additionally']"
          :title="'roll down'"
          @click="popMenuShow = !popMenuShow"
        ></i>
        <PopUpMenu v-if="popMenuShow" :settings="settings" :menus="menus" :show="graphicShow" @event="event" />
      </div>
    </div>
    <div class="t-char__main" v-if="graphicShow">
      <Plotly :data="data" :layout="layout" :display-mode-bar="false"></Plotly>
      <div v-if="!data.length" class="t-char__empty">
        <LoadSpiner v-if="start" text="Загрузка данных..." />
      </div>
    </div>
    <div></div>
  </div>
</template>

<script>
import { Plotly } from 'vue-plotly';
import PopUpMenu from './menu/PopUpMenu';
import LoadSpiner from '@/components/forms/LoadSpiner';
export default {
  name: 't-char',
  props: {
    id: Number,
    progress_state: {
      type: String,
      default: '',
    },
    graph_name: {
      type: String,
      default: '',
    },
    x_label: {
      type: String,
      default: '',
    },
    y_label: {
      type: String,
      default: '',
    },
    best: {
      type: Array,
      default: () => [],
    },
    plot_data: {
      type: Array,
      default: () => [],
    },
    epochs: {
      type: Array,
      default: () => [],
    },
    type: {
      type: String,
      default: 'lines',
    },
    settings: {
      type: Object,
      default: () => {},
    },
    menus: {
      type: Object,
      default: () => {},
    },
    start: Boolean,
  },
  components: {
    Plotly,
    PopUpMenu,
    LoadSpiner,
  },
  data: () => ({
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
    order() {
      return { order: !this.graphicShow ? 998 : this.settings.id };
    },
    layout() {
      const layout = this.defLayout;
      if (this.plot_data) {
        // layout.title.text = this.graph_name;
        layout.xaxis.title = this.x_label;
        layout.yaxis.title = this.y_label;
      }
      return layout;
    },
    minValue() {
      return Math.min(...[].concat(...this.plot_data.map(item => item.y)));
    },
    maxValue() {
      return Math.max(...[].concat(...this.plot_data.map(item => item.y)));
    },
    endpointX() {
      return this.epochs.map(item => {
        return [item, item, null];
      });
    },
    endpointY() {
      return this.epochs.map(() => {
        return [this.maxValue, this.minValue, null];
      });
    },
    changeEpochs() {
      return [
        {
          type: 'scatter',
          x: [].concat(...this.endpointX),
          y: [].concat(...this.endpointY),
          mode: 'line',
          name: 'Остановка обучения',
          // showlegend: false,
          hoverinfo: 'name',
          line: {
            color: 'grey',
            width: 2,
            dash: 'dash',
          },
        },
      ];
    },
    changeBest() {
      return !this.best
        ? []
        : this.best.map(({ x, y, mode = 'markers', label }, i) => {
            return {
              x,
              y,
              mode,
              name: `${label} ${y[0]}`,
              marker: {
                color: ['#1f77b4', '#ff7f0e'][i],
                symbol: 'circle',
                size: 10,
              },
            };
          });
    },
    changePlotData() {
      return this.plot_data.map(({ x, y, mode = 'lines', label }) => {
        return { x, y, mode, name: label.replace(/[<>]/g, '') };
      });
    },
    data() {
      return [...this.changePlotData, ...this.changeBest, ...this.changeEpochs];
    },
  },
  mounted() {
    // console.log(this.menus);
  },
  methods: {
    event({ name, data }) {
      if (data === 'hide') {
        this.show();
      }
      this.$emit('event', { name, data });
      this.popMenuShow = false;
    },
    show() {
      if (this.graphicShow) {
        this.popMenuShow = false;
      }
      this.graphicShow = !this.graphicShow;
    },
    outside() {
      this.popMenuShow = false;
    },
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
.underfitting {
  span {
    color: #f3d11d;
  }
  .indicator {
    background: #f3d11d;
  }
}
.overfitting {
  span {
    color: #d42e22;
  }
  .indicator {
    background: #d42e22;
  }
}
.t-char {
  width: 100%;
  height: 100%;
  background: #242f3d;
  border-radius: 4px;
  box-shadow: 0 2px 10px 0 rgb(0 0 0 / 25%);
  position: relative;
  &--order {
    order: 100;
  }
  &__main {
    position: relative;
  }
  &__empty {
    user-select: none;
    position: absolute;
    top: 0;
    height: 100%;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 16px;
    opacity: 0.7;
  }
  &__header {
    padding: 12px;
    display: flex;
    justify-content: space-between;

    &--title {
      overflow: hidden;
      text-overflow: ellipsis;
      padding-left: 12px;
      white-space: nowrap;
    }
    &--condition {
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
    &--additionally {
      position: relative;
      padding-left: 10px;
    }
  }
}
</style>
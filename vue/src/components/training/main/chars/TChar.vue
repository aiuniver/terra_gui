<template>
  <div class="t-char">
    <div class="t-char__header" v-click-outside="outside">
      <div class="t-char__header--roll">
        <i :class="['t-icon', 'icon-training-roll-down']" :title="'roll down'" @click="show"></i>
      </div>
      <div class="t-char__header--title">{{ char.graph_name || '' }}</div>
      <div v-if="char.progress_state" :class="['t-char__header--condition', char.progress_state]">
        <span>{{ char.progress_state || '' }}</span>
        <div class="indicator"></div>
      </div>
      <div class="t-char__header--additionally">
        <i
          :class="['t-icon', 'icon-training-additionally']"
          :title="'roll down'"
          @click="popMenuShow = !popMenuShow"
        ></i>
        <PopUpMenu v-if="popMenuShow" :data="['Loss', 'Metrics']" :metrics="['Accuracy', 'Hinge']" />
      </div>
    </div>
    <div class="t-char__main" v-if="graphicShow">
      <Plotly :data="data" :layout="layout" :display-mode-bar="false"></Plotly>
    </div>
  </div>
</template>

<script>
import { Plotly } from 'vue-plotly';
import PopUpMenu from './menu/PopUpMenu';
export default {
  name: 't-char',
  props: {
    char: {
      type: Object,
      default: () => {},
    },
  },
  components: {
    Plotly,
    PopUpMenu,
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
    layout() {
      const layout = this.defLayout;
      if (this.char) {
        layout.title.text = this.char?.title || '';
        layout.xaxis.title = this.char?.x_label || '';
        layout.yaxis.title = this.char?.y_label || '';
      }
      return layout;
    },
    data() {
      const data = this.char.plot_data || [];
      const arr = data.map(({ x, y, mode = 'lines', label }) => {
        return { x, y, mode, label };
      });
      return arr;
    },
  },
  mounted() {
    console.log(this.char);
  },
  methods: {
    show() {
      if (this.graphicShow) {
        this.popMenuShow = false
      }
      this.graphicShow = !this.graphicShow;
    },
    outside() {
      console.log('sdsdsd');
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
  &__main {
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
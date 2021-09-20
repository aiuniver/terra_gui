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
        <PopUpMenu
          v-if="popMenuShow"
          :menus="menus"
          @event="event"
        />
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
    plot_data: {
      type: Array,
      default: () => [],
    },
    type: {
      type: String,
      default: 'lines',
    },
    menus: {
      type: Array,
      default: () => []
    },
    settings: {
      type: Object,
      default: () => {}
    }
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
    data() {
      return this.plot_data.map(({ x, y, mode = this.type, label }) => {
        return { x, y, mode, label };
      });
    },
  },
  mounted() {
    console.log(this.menus);
  },
  methods: {
    event({ name, data }) {
      if (data === 'hide') {
        this.show()
      }
      this.$emit('event', { name, data })
      this.popMenuShow = false;
    },
    show() {
      if (this.graphicShow) {
        this.popMenuShow = false;
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
  &--order {
    order: 100;
  }
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
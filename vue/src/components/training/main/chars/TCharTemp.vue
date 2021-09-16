<template>
  <div class="content">
    <div class="char-overflow">
      <i class="t-icon icon-add-chart"></i>
      Добавить график
    </div>
    <div class="item">
      <div class="item__header">
        <div class="item__header-roll">
          <i
            :class="['t-icon', 'icon-training-roll-down']"
            :title="'roll down'"
            @click="graphicShow = !graphicShow"
          ></i>
        </div>
        <div class="item__header-title">{{ char.graph_name || '' }}</div>
        <div class="item__header-condition normal">
          <span>{{ char.progress_state || '' }}</span>
          <div class="indicator"></div>
        </div>
        <div class="item__header-additionally">
          <i
            :class="['t-icon', 'icon-training-additionally']"
            :title="'roll down'"
            @click="popMenuShow = !popMenuShow"
          ></i>
          <PopUpMenu v-if="popMenuShow" />
        </div>
      </div>
      <div class="item__main" v-if="graphicShow">
        <Plotly :data="data" :layout="layout" :display-mode-bar="false"></Plotly>
      </div>
    </div>
  </div>
</template>

<script>
import { Plotly } from 'vue-plotly';
import PopUpMenu from './menu/PopUpMenu';
export default {
  name: 'Tchar',
  components: {
    Plotly,
    PopUpMenu,
  },
  data: () => ({
    char: {
      graph_name: 'График',
      x_label: 'Эпоха',
      y_label: 'Значение',
      plot_data: [
        {
          label: 'Тренировочная выборка',
          epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          values: [
            0.9713165163993835, 0.8578746318817139, 0.802897572517395, 0.7554795145988464, 0.7013662457466125,
            0.6513573527336121, 0.6137965321540833, 0.5739713907241821, 0.5361674427986145, 0.5122222900390625,
            0.4707668423652649, 0.4400102198123932,
          ],
        },
        {
          label: 'Проверочная выборка',
          epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          values: [
            0.942412257194519, 0.8653479218482971, 0.8113749623298645, 0.7803205251693726, 0.7638459205627441,
            0.7569182515144348, 0.7524999380111694, 0.7603248357772827, 0.757815957069397, 0.7624929547309875,
            0.7704657912254333, 0.7801728844642639,
          ],
        },
      ],
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
        layout.title.text = this.char?.title || '';
        layout.xaxis.title = this.char?.x_label || '';
        layout.yaxis.title = this.char?.y_label || '';
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
  mounted() {
    console.log(this.char);
  },
};
</script>

<style lang="scss" scoped>
.content {
  width: 50%;
  padding: 0;
  position: relative;
}
.char-overflow {
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
  i {
    margin-right: 20px;
  }
}
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
.item {
  opacity: 0.3;
  width: 100%;
  height: 100%;
  background: #242f3d;
  border-radius: 4px;
  box-shadow: 0 2px 10px 0 rgb(0 0 0 / 25%);
  overflow: hidden;
  &__header {
    padding: 12px;
    display: flex;
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
    }
  }
}
</style>
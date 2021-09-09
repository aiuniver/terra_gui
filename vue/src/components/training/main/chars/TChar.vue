<template>
  <div class="content">
    <div class="item">
      <div class="item__header">
        <div class="item__header-roll">
          <i :class="['t-icon', 'icon-training-roll-down']" :title="'roll down'" @click="graphicShow = !graphicShow"></i>
        </div>
        <div class="item__header-title">Output_1 – График ошибки обучения – Эпоха №18</div>
        <div class="item__header-condition normal">
          <span>Состояние в норме</span>
          <div class="indicator"></div>
        </div>
        <div class="item__header-additionally">
          <i :class="['t-icon', 'icon-training-additionally']" :title="'roll down'" @click="popMenuShow = !popMenuShow"></i>
          <PopUpMenu v-if="popMenuShow"/>
        </div>
      </div>
      <div class="item__main" v-if="graphicShow">
        <Plotly :data="data" :layout="layout" :display-mode-bar="false"></Plotly>
      </div>
    </div>
  </div>
</template>

<script>
import { Plotly } from "vue-plotly";
import PopUpMenu from "../PopUpMenu";
export default {
  name: "Tchar",
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
    graphicShow: false,
    popMenuShow: false,
    defLayout: {
      autosize: true,
      margin: {
        l: 70,
        r: 20,
        t: 10,
        b: 20,
        pad: 0,
        autoexpand: true,
      },
      font: {
        color: "#A7BED3",
      },
      showlegend: true,
      legend: {
        y: -0.25,
        itemsizing: "constant",
        orientation: "h",
        font: {
          family: "Open Sans",
          color: "#A7BED3",
        },
      },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      title: {
        text: "",
      },
      xaxis: {
        title: "",
        showgrid: true,
        zeroline: false,
        linecolor: "#A7BED3",
        gridcolor: "#0E1621",
        gridwidth: 1,
      },
      yaxis: {
        title: "",
        showgrid: true,
        zeroline: false,
        linecolor: "#A7BED3",
        gridcolor: "#0E1621",
        gridwidth: 1,
      },
    },
  }),
  computed: {
    layout() {
      const layout = this.defLayout;
      if (this.char) {
      layout.title.text = this.char.title || "";
      layout.xaxis.title = this.char.xaxis.title || "";
      layout.yaxis.title = this.char.yaxis.title || "";
      }
      return layout;
    },
    data() {
      const data = this.char.list || [];
      return data;
    },
  },
};
</script>

<style lang="scss" scoped>
.content {
  width: 50%;
  padding: 0 10px 20px 10px;
}
.normal{
  span{
    color: #65CA35
  }
  .indicator{
    background: #65CA35;
  }
}
.item{
  width: 100%;
  height: 100%;
  background: #242f3d;
  border-radius: 4px;
  box-shadow: 0 2px 10px 0 rgb(0 0 0 / 25%);
  overflow: hidden;
  &__header{
    padding: 12px;
    display: flex;
    &-title{
      padding-left: 12px;
    }
    &-condition{
      border: 1px solid #6C7883;
      box-sizing: border-box;
      border-radius: 4px;
      display: flex;
      padding: 0 8px;
      margin-left: auto;
      span{
        font-size: 12px;
        line-height: 24px;
      }
      .indicator{
        width: 12px;
        height: 12px;
        border-radius: 7px;
        margin: 6px;
      }
    }
    &-additionally{
      padding-left: 10px;
    }
  }
}

</style>
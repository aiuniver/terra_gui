<template>
  <div class="item">
    <div class="item__main">
      <Plotly :data="data" :layout="layout" :display-mode-bar="false"></Plotly>
    </div>
  </div>
</template>

<script>
import { Plotly } from "vue-plotly";
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
  },
  data: () => ({
    defLayout: {
      autosize: true,
      margin: {
        l: 70,
        r: 20,
        t: 60,
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
.item {
  width: 50%;
  padding: 0 10px 20px 10px;
  &__main {
    width: 100%;
    height: 100%;
    background: #242f3d;
    border-radius: 4px;
    box-shadow: 0 2px 10px 0 rgb(0 0 0 / 25%);
    overflow: hidden;
  }
}
</style>
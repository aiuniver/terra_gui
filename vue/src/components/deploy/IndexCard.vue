<template>
<div class="card">
  <div class="card__content">
    <div v-if="type != 'graphic'">
      <div class="card__original" >
        <ImgCard v-if="original.type == 'image'"/>
        <TextCard v-if="original.type == 'text'" :style="originaltextStyle">{{ original.data }}</TextCard>
      </div>
      <div class="card__result">
        <ImgCard v-if="result.type == 'image'"/>
        <TextCard v-if="result.type == 'text'">{{ result.data }}</TextCard>
      </div>
    </div>
    <div class="card__graphic" v-if="type == 'graphic'">
       <Plotly :data="data" :layout="layout" :display-mode-bar="false"></Plotly>
    </div>
  </div>
  <div class="card__reload"><button class="btn-reload"><i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i></button></div>
</div>
</template>

<script>
import ImgCard from "./cards/ImgCard";
import TextCard from "./cards/TextCard";
import { Plotly } from "vue-plotly";
import {mapGetters} from "vuex";
export default {
  name: "IndexCard",
  components: {
    ImgCard,
    TextCard,
    Plotly
  },
  data: () => ({
    originaltextStyle: {
      width: "600px",
      height: "300px",
      color: "#A7BED3",
      padding: "10px 25px 12px 12px"
    },
    defLayout: {
      autosize: true,
      margin: {
        l: 62,
        r: 20,
        t: 20,
        b: 67,
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
        title: "Эпоха",
        showgrid: true,
        zeroline: false,
        linecolor: "#A7BED3",
        gridcolor: "#0E1621",
        gridwidth: 1,
      },
      yaxis: {
        title: "accuracy",
        showgrid: true,
        zeroline: false,
        linecolor: "#A7BED3",
        gridcolor: "#0E1621",
        gridwidth: 1,
      },
    },
  }),
  props: {
    original: {
      type: Object,
      default: () => ({})
    },
    result: {
      type: Object,
      default: () => ({})
    },
    type: {
      type: String,
      default: ""
    },
  },
  mounted() {
    console.log(this.graphicData)
  },
  computed: {
    ...mapGetters({
      graphicData : 'deploy/getGraphicData'
    }),
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
      const data = [this.graphicData] || [];
      return data;
    },
  },
}
</script>

<style lang="scss" scoped>
.card__reload{
  padding-left: 5px;
}
.card{
  padding: 15px 15px 15px 0;
  display: flex;
}
.card__graphic{
  background: #242F3D;
  border: 1px solid #6C7883;
  box-sizing: border-box;
  border-radius: 4px;
}
.card__original{
  background: #242F3D;
}
.card__result{
  padding-top: 6px;
}
.btn-reload{
    width: 32px;
    height: 32px;
    i{
      position: absolute;
      margin-left: 7px;
      margin-top: -13px;
      width: 16px;
    }
  }
</style>
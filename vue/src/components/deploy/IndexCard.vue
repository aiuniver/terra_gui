<template>
<div class="card">
  <div class="card__content">
    <div v-if="deployType == 'image_classification'">
      <div class="card__original" >
        <ImgCard v-if="deployType == 'image_classification'" :imgUrl="source"/>
        <TextCard v-if="deployType == 'text'" :style="origTextStyle">{{ data }}</TextCard>
      </div>
      <div class="card__result">
        <ImgCard v-if="deployType == 'text'" :imgUrl="source"/>
        <TextCard v-if="deployType == 'image_classification'" :style="deployType == 'image_classification' ? { width: '224px' } : {}">{{ data }}</TextCard>
      </div>
    </div>
    <div class="card__graphic" v-if="deployType == 'graphic'">
       <Plotly :data="data" :layout="layout" :display-mode-bar="false"></Plotly>
    </div>
<!--    <div class="card__table" v-if="type == 'table'">-->
<!--      <Table/>-->
<!--    </div>-->
  </div>
  <div class="card__reload" v-if="deployType != 'table'"><button class="btn-reload" @click="ReloadCard"><i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i></button></div>
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
    Plotly,
  },
  data: () => ({}),
  props: {
    source: {
      type: Object, String,
      default: () => ({})
    },
    data: {
      type: Object, String,
      default: () => ({})
    },
  },
  mounted() {
    console.log(this.deployType)
  },
  methods: {
    ReloadCard(){
      console.log("RELOAD_CARD")
    }
  },
  computed: {
    ...mapGetters({
      graphicData: 'deploy/getGraphicData',
      defaultLayout: 'deploy/getDefaultLayout',
      origTextStyle: 'deploy/getOrigTextStyle',
      deployType: "deploy/getDeployType",
    }),
    layout() {
      const layout = this.defaultLayout;
      if (this.char) {
        layout.title.text = this.char.title || "";
        layout.xaxis.title = this.char.xaxis.title || "";
        layout.yaxis.title = this.char.yaxis.title || "";
      }
      return layout;
    },
    // data() {
    //   const data = [this.graphicData] || [];
    //   return data;
    // },
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
.card__table{
  width: 100%;
}
</style>
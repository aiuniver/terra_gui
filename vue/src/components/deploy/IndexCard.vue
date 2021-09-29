<template>
<div class="card">
  <div class="card__content">
    <div v-if="deployType == 'image_classification'">
      <div class="card__original" >
        <ImgCard :imgUrl="source"/>
      </div>
      <div class="card__result">
        <TextCard  :style="{ width: '224px' }">{{ imageClassificationText }}</TextCard>
      </div>
    </div>
    <div v-if="deployType == 'image_segmentation'">
      <div class="card__original" >
        <ImgCard :imgUrl="source"/>
      </div>
      <div class="card__result">
        <ImgCard :imgUrl="segment"/>
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
      type: String,
      default: ""
    },
    segment: {
      type: String,
      default: ""
    },
    data: {
      type: [Array, Object, String],
      default: () => ({})
    },
    block: String,
    index: [String, Number],
  },
  methods: {
    ReloadCard(){
      this.$emit('reload', { id: this.block, indexes: [this.index.toString()]})
    },
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
    imageClassificationText(){
      let text = this.data;
      let prepareText = "";
      text.sort((a, b) => a[1] < b[1] ? 1 : -1);
      for(let i=0; i<text.length; i++){
        if(i > 2) break;
        prepareText = prepareText + `${text[i][0]} - вероятность ${text[i][1]}% \n`;
      }
      return prepareText;
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
.card__table{
  width: 100%;
}
</style>
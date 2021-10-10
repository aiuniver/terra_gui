<template>
<div class="card">
  <div class="card__content">
    <div v-if="type == 'ImageClassification'">
      <div class="card__original" >
        <ImgCard :imgUrl="source"/>
      </div>
      <div class="card__result">
        <TextCard :style="{ width: '224px' }">{{ imageClassificationText }}</TextCard>
      </div>
    </div>
     <div v-if="type == 'TextClassification'">
      <div class="card__original" >
        <TextCard :style="{ width: '600px', color: '#A7BED3', height: '324px' }">{{ source }}</TextCard>
      </div>
      <div class="card__result">
        <TextCard  :style="{ width: '600px', height: '80px' }">{{ imageClassificationText }}</TextCard>
      </div>
    </div>
    <div v-if="type == 'TextTextSegmentation'">
      <div class="card__original" >
        <TextCard :style="{ width: '600px', color: '#A7BED3', height: '324px' }">{{ source }}</TextCard>
      </div>
      <div class="card__result">
        <TextCard  :style="{ width: '600px', height: '80px' }">
           <p v-for="(tag, index) in data" :key="'tag-'+index" class="p-segmentation">
             <s1  :style="{'background-color': rgbToHex(tag[2])}">{{ tag[0] }}</s1> - Название {{ index+1 }}
           </p>
        </TextCard>
        <s1></s1>
        <s2></s2>
        <s3></s3>
        <s4></s4>
        <s5></s5>
        <s6></s6>
      </div>
    </div>
    <div v-if="type == 'ImageSegmentation'">
      <div class="card__original" >
        <ImgCard :imgUrl="source"/>
      </div>
      <div class="card__result">
        <ImgCard :imgUrl="segment"/>
      </div>
    </div>
    <div class="card__graphic" v-if="type == 'graphic'">
       <Plotly :data="data" :layout="layout" :display-mode-bar="false"></Plotly>
    </div>
<!--    <div class="card__table" v-if="type == 'table'">-->
<!--      <Table/>-->
<!--    </div>-->
  </div>
  <div class="card__reload" v-if="type != 'table'"><button class="btn-reload" @click="ReloadCard"><i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i></button></div>
</div>
</template>

<script>
import ImgCard from "./cards/ImgCard";
import TextCard from "./cards/TextCard";
import { Plotly } from "vue-plotly";
import {s1, s2, s3, s4, s5, s6} from './tags/TagsS'
import {mapGetters} from "vuex";
export default {
  name: "IndexCard",
  components: {
    ImgCard,
    TextCard,
    Plotly,
    s1, s2, s3, s4, s5, s6
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
    type: String,
  },
  methods: {
    ReloadCard(){
      this.$emit('reload', { id: this.block, indexes: [this.index.toString()]})
    },
    rgbToHex(rgb) {
      return "#" + ((1 << 24) + (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]).toString(16).slice(1);
}
  },
  computed: {
    ...mapGetters({
      graphicData: 'deploy/getGraphicData',
      defaultLayout: 'deploy/getDefaultLayout',
      origTextStyle: 'deploy/getOrigTextStyle',
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
        prepareText = prepareText + `${text[i][0]} - ${text[i][1]}% \n`;
      }
      return prepareText;
    },
    SegmentationText(){
      let text = this.data;
      let prepareText = "";
      for(let i=0; i<text.length; i++){
        prepareText = prepareText + `${text[i][0]} - название ${i+1} \n`;
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
s1, s2, s3, s4, s5, s6{
  border-radius: 4px;
  color: #FFFFFF;
}
.p-segmentation{
  margin-top: 5px;
  color: #65B9F4;
}
</style>


<template>
<div class="card">
  <div class="card__content">
    <div v-if="type == 'card'">
      <div class="card__original" >
        <ImgCard v-if="original.type == 'image'" :imgUrl="original.imgUrl"/>
        <TextCard v-if="original.type == 'text'" :style="origTextStyle">{{ original.data }}</TextCard>
      </div>
      <div class="card__result">
        <ImgCard v-if="result.type == 'image'" :imgUrl="result.imgUrl"/>
        <TextCard v-if="result.type == 'text'" :style="original.type == 'image' ? { width: '224px' } : {}">{{ result.data }}</TextCard>
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
import {mapGetters} from "vuex";
export default {
  name: "IndexCard",
  components: {
    ImgCard,
    TextCard,
    Plotly,
  },
  data: () => ({
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
.card__table{
  width: 100%;
}
</style>
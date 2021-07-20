<template>
  <main class="page-modeling">
    <div class="container">
      <Toolbar @click="click"/>
      <ModCanvas2 @loadParams="loadParams"/>
      <Params :node="node_settings"/>
      <ModalLoadModel        
      ></ModalLoadModel>
    </div>
  </main>
</template>

<script>
// import SimpleFlowchart from "@/components/flowchart/SimpleFlowchart";
import Toolbar from "@/components/modeling/Toolbar";
import ModCanvas2 from "@/components/modeling/ModCanvas2";
import Params from "@/components/modeling/Params";
import ModalLoadModel from "@/components/modeling/ModalLoadModel";
import { mapGetters } from "vuex";

export default {
  name: "Modeling",
  components: {
    // SimpleFlowchart,
    Toolbar,
    ModCanvas2,
    Params,
    ModalLoadModel,
  },
  data() {
    return {
      load_model_flag: false,
      node_settings: {},
      nodeType: 1,
      nodeLabel: "",
      nodeCategory: ["input", "action", "output"],
      nodeIcons: [
        "mdi-format-horizontal-align-left",
        "mdi-format-horizontal-align-center",
        "mdi-format-horizontal-align-right",
      ],
      rules: {
        length: (len) => (v) => (v || "").length >= len || `Length < ${len}`,
      },
      middleLayer: {
        name: "nameeee",
        type: "Conv1D",
        group: "middle",
        bind: [3],
        shape: [1, 1, 1],
        location: null,
        position: [100, 100],
        parameters: {}
      }
    };
  },
  computed: {
    ...mapGetters({
      scene: "data/getData",
    }),
  },
  methods: {
    loadParams(node){
      this.node_settings = node
      console.log(node)
    },
    canvasClick(e) {
      console.log("canvas Click, event:", e);
      console.log(e.type);
    },
    click(e) {
      console.log(e);
      switch (e){
        case "load":
          this.$store.dispatch('modeling/setDialog', true)
          break
        case "middle":
          this.$store.dispatch('data/addLayer', Object.assign({}, this.middleLayer));
          break
      }

    },
  },
};
</script>

<style scoped>
/*@import "./../../public/css/project/modeling.css";*/
</style>

<template>
  <main class="page-modeling">
    <div class="container">
      <Toolbar @click="click"/>
      <ModCanvas />
      <Params />
      <ModalLoadModel        
      ></ModalLoadModel>
    </div>
  </main>
</template>

<script>
// import SimpleFlowchart from "@/components/flowchart/SimpleFlowchart";
import Toolbar from "@/components/modeling/Toolbar";
import ModCanvas from "@/components/modeling/ModCanvas";
import Params from "@/components/modeling/Params";
import ModalLoadModel from "@/components/modeling/ModalLoadModel";
import { mapGetters } from "vuex";

export default {
  name: "Modeling",
  components: {
    // SimpleFlowchart,
    Toolbar,
    ModCanvas,
    Params,
    ModalLoadModel,
  },
  data() {
    return {
      load_model_flag: false,
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
    };
  },
  computed: {
    ...mapGetters({
      scene: "data/getData",
    }),
  },
  methods: {
    canvasClick(e) {
      console.log("canvas Click, event:", e);
      console.log(e.type);
    },
    click(e) {
      console.log(e);
      this.$store.dispatch('modeling/setDialog', true)
    },
  },
};
</script>

<style scoped>
/*@import "./../../public/css/project/modeling.css";*/
</style>

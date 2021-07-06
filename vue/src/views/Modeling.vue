<template>
<main class="page-modeling">
    <div class="container">
      <Toolbar :load_model_flag="load_model_flag"></Toolbar>
      <Canvas/>
      <Params/>
      <ModalWindowLoadModel v-if="load_model_flag"></ModalWindowLoadModel>
    </div>
</main>
</template>

<script>
// import SimpleFlowchart from "@/components/flowchart/SimpleFlowchart";
import Toolbar from "@/components/modeling/Toolbar";
import Canvas from "@/components/modeling/Canvas";
import Params from "@/components/modeling/Params";
import ModalWindowLoadModel from "@/components/modeling/ModalWindowLoadModel";
import { mapGetters } from "vuex";

export default {
  name: "Modeling",
  components: {
    // SimpleFlowchart,
    Toolbar,
    Canvas,
    Params,
    ModalWindowLoadModel
  },
  data() {
    return {
      dialog: false,
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
    drawer: {
      set(value) {
        this.$store.dispatch("settings/setDrawer", value);
      },
      get() {
        return this.$store.getters["settings/getDrawer"];
      },
    },
  },
  methods: {
    canvasClick(e) {
      console.log("canvas Click, event:", e);
      console.log(e.type);
    },
    add() {
      if (this.$refs.form.validate()) {
        let maxID = Math.max(
          0,
          ...this.scene.nodes.map((link) => {
            return link.id;
          })
        );
        this.scene.nodes.push({
          id: maxID + 1,
          x: -400,
          y: -100,
          type: this.nodeCategory[this.nodeType],
          label: this.nodeLabel ? this.nodeLabel : `test${maxID + 1}`,
        });
        this.nodeLabel = "";
        this.dialog = false;
      }
    },
    cancel() {
      console.log(this.$refs.form.reset());
      this.nodeLabel = "";
      this.dialog = false;
    },
    save() {
      const { nodes, links } = this.scene;
      console.log({ nodes, links });
      alert(JSON.stringify({ nodes, links }));
    },
    nodeClick(id) {
      console.log("node click", id);
      // this.drawer = true
    },
    nodeDelete(id) {
      console.log("node delete", id);
    },
    linkBreak(id) {
      console.log("link break", id);
    },
    linkAdded(link) {
      console.log("new link added:", link);
    },
    isColor(type) {
      return this.nodeType !== type ? "text" : "white";
    },
  },
};
</script>

<style lang="scss" scoped>
.sidebar {
  float: left;
  width: auto;
}
</style>

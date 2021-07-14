<template>
<div class="board">
    <div class="wrapper">
        <div class="canvas-container">
            <div class="canvas">
              <VueFlowchart></VueFlowchart>
            </div>
            <ul class="zoom">
                <li class="inc" id="zoom-inc" data-type="inc"><span class="icon-zoom-inc"></span></li>
                <li class="res" id="zoom-reset" data-type="reset"><span class="icon-zoom-reset"></span></li>
                <li class="dec" id="zoom-dec" data-type="dec"><span class="icon-zoom-dec"></span></li>
            </ul>
        </div>
    </div>
</div>
</template>

<script>
import VueFlowchart from "@/components/modeling/flowchart/SimpleFlowchart";
export default {
  name: "ModCanvas",
  components:{
    VueFlowchart,
  },
  data() {
    return {
      dialog: false,
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
      scene: {
        centerX: 1024,
        centerY: 140,
        scale: 1,
        nodes: [
          {
            id: 1,
            x: -650,
            y: -100,
            type: "input",
            label: "test1",
          },
          {
            id: 2,
            x: -500,
            y: 50,
            type: "middle",
            label: "test2",
          },
          {
            id: 3,
            x: -800,
            y: 50,
            type: "middle",
            label: "test3",
          },
          {
            id: 4,
            x: -650,
            y: 150,
            type: "output",
            label: "test4",
          },
        ],
        links: [
          {
            id: 1,
            from: 1,
            to: 2,
          },
          {
            id: 2,
            from: 1,
            to: 3,
          },
          {
            id: 3,
            from: 2,
            to: 4,
          },
          {
            id: 4,
            from: 3,
            to: 4,
          },
        ],
      },
    };
  },
  methods: {
    canvasClick(e) {
      console.log("canvas Click, event:", e);
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
  },
};
</script>

<style scoped>

</style>
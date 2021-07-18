<template>
  <div class="board">
    <div class="wrapper">
      <div class="canvas-container">
        <div class="canvas" :style="style">
          <VueBlocksContainer
            ref="container"
            :blocksContent="blocks"
            :scene.sync="scene"
            class="cont"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import VueBlocksContainer from "@/components/modeling/block/VueBlocksContainer";
export default {
  name: "ModCanvas",
  components: {
    VueBlocksContainer,
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
      blocks: [
        {
          name: "input",
          title: "Input",
          family: "Animations",
          description: "Show text",
          fields: [
            {
              name: "Output",
              type: "event",
              attr: "output",
            },
          ],
        },
        {
          name: "output",
          title: "Output",
          description: "Show text",
          fields: [
            {
              name: "Input",
              type: "event",
              attr: "input",
            },
          ],
        },
        {
          name: "sloy",
          family: "Events",
          description: "",
          fields: [
            {
              name: "Input",
              type: "event",
              attr: "input",
            },
            {
              name: "onMessage",
              type: "event",
              attr: "output",
            },
            {
              name: "Output",
              type: "event",
              attr: "output",
            },
            {
              name: "Output",
              type: "event",
              attr: "output",
            },
          ],
        },
        {
          name: "shortcuts",
          title: "Shortcuts",
          family: "Events",
          description: "Press shortcut for call event",
          fields: [
            {
              name: "keys",
              label: "Activation keys",
              type: "keys",
              attr: "property",
            },
            {
              name: "onPress",
              type: "event",
              attr: "output",
            },
          ],
        },
        {
          name: "splitter",
          title: "Splitter",
          family: "Helpers",
          description: "Press shortcut for call event",
          fields: [
            {
              name: "input",
              type: "event",
              attr: "input",
            },
            {
              name: "output",
              type: "event",
              attr: "output",
            },
            {
              name: "output",
              type: "event",
              attr: "output",
            },
            {
              name: "output",
              type: "event",
              attr: "output",
            },
          ],
        },
      ],
      scene: {
        blocks: [
          {
            id: 1,
            x: -900,
            y: 50,
            name: "input",
            title: "Input",
          },
          {
            id: 2,
            x: -900,
            y: 150,
            name: "sloy",
            title: "Sloy",
          },
          {
            id: 3,
            x: -900,
            y: 250,
            name: "sloy",
            title: "Sloy",
          },
          {
            id: 4,
            x: -900,
            y: 350,
            name: "output",
            title: "Output",
          },
        ],
        links: [
          {
            id: 1,
            originID: 1,
            originSlot: 0,
            targetID: 2,
            targetSlot: 0,
          },
        ],
        container: {
          centerX: 1042,
          centerY: 140,
          scale: 1,
        },
      },
    };
  },
  computed: {
    style() {
      console.log(document.documentElement.clientHeight);
      return {
        height: (document.documentElement.clientHeight - 157) + "px",
      };
    },
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

<style lang="scss" scoped>
.canvas-container {
  position: relative;
}
.canvas {
  // height: 500px;
}
.cont {
  height: 100%;
}
</style>
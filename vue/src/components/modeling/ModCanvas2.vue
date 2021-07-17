<template>
  <div class="board">
    <div class="wrapper">
      <div class="canvas-container">
        <div class="canvas">
          <VueBlocksContainer
            ref="container"
            :blocksContent="blocks"
            :scene.sync="scene"
            class="container"
          />
        </div>
        <ul class="zoom">
          <li class="inc" id="zoom-inc" data-type="inc">
            <span class="icon-zoom-inc"></span>
          </li>
          <li class="res" id="zoom-reset" data-type="reset">
            <span class="icon-zoom-reset"></span>
          </li>
          <li class="dec" id="zoom-dec" data-type="dec">
            <span class="icon-zoom-dec"></span>
          </li>
        </ul>
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
              name: "Input",
              type: "event",
              attr: "input",
            }
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
              attr: "output",
            }
          ],
        },
        {
          name: "Chat message",
          family: "Events",
          description: "",
          fields: [
            {
              name: "message",
              label: "Activation message",
              type: "string",
              attr: "property",
            },
            {
              name: "onMessage",
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
            values: {
              property: {
                text: {
                  label: "Text",
                  type: "string",
                },
              },
            },
          },
          {
            id: 2,
            x: -600,
            y: 50,
            name: "output",
            title: "Output",
            values: {
              property: {
                text: {
                  label: "Text",
                  type: "string",
                },
              },
            },
          },
          {
            id: 3,
            x: -600,
            y: 150,
            name: "shortcuts",
            title: "Text",
            values: {
              property: {
                text: {
                  label: "Text",
                  type: "string",
                },
              },
            },
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
          {
            id: 2,
            originID: 1,
            originSlot: 1,
            targetID: 2,
            targetSlot: 1,
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
.canvas {
  height: 1000px;
}
.container {
  height: 100%;
}
</style>
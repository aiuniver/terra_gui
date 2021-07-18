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
          {
            id: 2,
            originID: 2,
            originSlot: 1,
            targetID: 3,
            targetSlot: 0,
          },
          {
            id: 3,
            originID: 1,
            originSlot: 0,
            targetID: 3,
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
      return {
        height: (document.documentElement.clientHeight - 157) + "px",
      };
    },
  },
  methods: {
    addBlock () {
      console.log(this.selectedType)
      this.$refs.container.addNewBlock(this.selectedType)
    },
  },
};
</script>

<style lang="scss" scoped>
.canvas-container {
  position: relative;
}
.cont {
  height: 100%;
}
</style>
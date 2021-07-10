<template>
  <div
    class="flowchart-node"
    :style="nodeStyle"
    @mousedown="handleMousedown"
    @mouseover="handleMouseOver"
    @mouseleave="handleMouseLeave"
    v-bind:class="{ selected: options.selected === id }"
  >
    <div v-if="type!=='input'"
      class="node-port node-input"
      @mousedown="inputMouseDown"
      @mouseup="inputMouseUp"
    ></div>
    <div :class="`node-main ${type}`">
      <div v-text="type"></div>
      <div v-text="label"></div>
    </div>
    <div v-if="type!=='output'" class="node-port node-output" @mousedown="outputMouseDown"></div>
    <div v-show="show.delete" class="node-delete">&times;</div>
  </div>
</template>

<script>
export default {
  name: "FlowchartNode",
  props: {
    id: {
      type: Number,
      default: 1000,
      validator(val) {
        return typeof val === "number";
      },
    },
    x: {
      type: Number,
      default: 0,
      validator(val) {
        return typeof val === "number";
      },
    },
    y: {
      type: Number,
      default: 0,
      validator(val) {
        return typeof val === "number";
      },
    },
    type: {
      type: String,
      default: "middle",
    },
    label: {
      type: String,
      default: "input name",
    },
    options: {
      type: Object,
      default() {
        return {
          centerX: 1024,
          scale: 1,
          centerY: 140,
        };
      },
    },
  },
  data() {
    return {
      show: {
        delete: false,
      },
    };
  },
  mounted() {},
  computed: {
    nodeStyle() {
      return {
        top: this.options.centerY + this.y * this.options.scale + "px", // remove: this.options.offsetTop +
        left: this.options.centerX + this.x * this.options.scale + "px", // remove: this.options.offsetLeft +
        transform: `scale(${this.options.scale})`,
      };
    },
  },
  methods: {
    handleMousedown(e) {
      const target = e.target || e.srcElement;
      // console.log(target);
      if (
        target.className.indexOf("node-input") < 0 &&
        target.className.indexOf("node-output") < 0
      ) {
        this.$emit("nodeSelected", e);
      }
      e.preventDefault();
    },
    handleMouseOver() {
      this.show.delete = true;
    },
    handleMouseLeave() {
      this.show.delete = false;
    },
    outputMouseDown(e) {
      this.$emit("linkingStart");
      e.preventDefault();
    },
    inputMouseDown(e) {
      e.preventDefault();
    },
    inputMouseUp(e) {
      this.$emit("linkingStop");
      e.preventDefault();
    },
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
$themeColor: rgb(85, 125, 255);
$imputColor: rgb(207 190 7);
$outputColor: rgb(29, 172, 48);
$portSize: 10;

.flowchart-node {
  margin: 0;
  width: 160px;
  height: 40px;
  position: absolute;
  box-sizing: border-box;
  border: none;
  border-radius: 5px;
  background: rgb(61, 32, 32);
  z-index: 1;
  opacity: 0.9;
  cursor: move;
  transform-origin: top left;
  .node-main {
    text-align: center;
    border-radius: 5px;
    color: black;
    font-size: 13px;
  }
  .node-port {
    position: absolute;
    width: #{$portSize}px;
    height: #{$portSize}px;
    left: 50%;
    transform: translate(-50%);
    border: 1px solid #ccc;
    border-radius: 100px;
    background: white;
    &:hover {
      background: $themeColor;
      border: 1px solid $themeColor;
    }
  }
  .node-input {
    top: -7px;
  }
  .node-output {
    bottom: -7px;
  }
  .node-delete {
    position: absolute;
    right: -6px;
    top: -6px;
    font-size: 12px;
    width: 18px; 
    height: 18px;
    color: $themeColor;
    cursor: pointer;
    background: white;
    border: 1px solid $themeColor;
    border-radius: 100px;
    text-align: center;
    &:hover {
      background: $themeColor;
      border: 1px solid white;
      color: white;
    }
  }

  .input{
    background: #FFB054;
  }
  .middle{
    background: #89D764;
  }
  .output{
    background: #8E51F2;
  }
}
</style>

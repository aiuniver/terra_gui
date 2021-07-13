<template>
  <div
    class="flowchart-node"
    :style="nodeStyle"
    @mousedown="handleMousedown"
    @mouseover="handleMouseOver"
    @mouseleave="handleMouseLeave"
    v-bind:class="{ selected: options.selected === id }"
  >
    <div v-if="group!=='input'"
      class="node-port node-input"
      @mousedown="inputMouseDown"
      @mouseup="inputMouseUp"
    ></div>
    <div :class="`node-main ${group}`">
      <div class="node-naming">{{ name }}: {{ type }}</div>
      <div class="node-params">parameters</div>
    </div>
    <div v-if="group!=='output'" class="node-port node-output" @mousedown="outputMouseDown"></div>
    <div v-show="show.delete" class="node-delete">&times;</div>
  </div>
</template>

<script>
export default {
  name: "FlowchartNode",
  props: {
    id: {
      type: Number,
    },
    name: {
      type: String,
      default: "Default Name"
    },
    type: {
      type: String,
      default: "Conv2D",
    },
    group: {
      type: String,
      default: "middle"
    },
    bind: {
      type: Array,
      default: ()=>{return []}
    },
    shape: {
      type: Array,
      default: ()=>{return [1, 1, 1]}
    },
    location: {
      type: Array,
      default: ()=>{return null}
    },
    position: {
      type: Array,
      default: ()=>{return [20, 300]}
    },
    parameters: {
      type: Object,
      default: ()=>{return {}}
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
      let fontSize = 11
      return {
        top: this.position[1] * this.options.scale + "px", // remove: this.options.offsetTop +
        left: this.position[0] * this.options.scale + "px", // remove: this.options.offsetLeft +
        width: ((this.name.length + this.type.length)*fontSize).toString() + "px",
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
  height: 42px;
  position: absolute;
  box-sizing: border-box;
  border: none;
  border-radius: 5px;
  background: white;
  z-index: 1;
  opacity: 0.9;
  cursor: move;
  transform-origin: top left;
  background: #17212B;
  .node-main {
    height: 100%;
    text-align: center;
    border-radius: 5px;
    color: black;
    font-size: 13px;
    &:hover{
      color: #FFFFFF;
      .node-params{
        color: #2B5278;
      }
    }
  }
  .node-naming{
    padding-top: 2px;
  }
  .node-params{
    color: #2B5278;
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

  //Colors of layers from their types
  .input{
    background: #FFB054;
    &:hover{
      background: none;
      box-shadow: 0 0 0 1.5pt #FFB054;
    }
  }
  .middle{
    background: #89D764;
    &:hover{
      background: none;
      box-shadow: 0 0 0 1.5pt #89D764;
    }
  }
  .output{
    background: #8E51F2;
    &:hover{
      background: none;
      box-shadow: 0 0 0 1.5pt #8E51F2;
    }
  }
}
</style>

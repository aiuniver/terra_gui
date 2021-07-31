<template>
  <div
    class="vue-block"
    :style="style"
    @mouseover="hover = true"
    @mouseleave="hover = false"
  >
    <div :class="['header', group, { selected: selected }]">
      <div class="title" :title="name">{{ id }}: {{ name }}</div>
      <div class="parametr" :title="parameters">[dsd]sdsds</div>
      <!-- <a class="delete" @click="deleteBlock">x</a> -->
    </div>
    <div
      v-if="group.indexOf('middle') === -1"
      v-show="hover || selected"
      class="hover-over"
    >
      <i class="icon icon-link"></i>
      <i class="icon icon-link-2"></i>
    </div>
    <div v-else v-show="hover || selected" class="hover-sloy">
      <i class="icon icon-link"></i>
      <i class="icon icon-link-2"></i>
      <i class="icon icon-trash-2" @click="deleteBlock"></i>
    </div>
    <div class="inputs">
      <div
        v-for="(slot, index) in inputs"
        :key="'input' + index"
        class="input inputSlot"
        :class="{ active: slot.active }"
        @mouseup="slotMouseUp($event, index)"
        @mousedown="slotBreak($event, index)"
      ></div>
    </div>
    <div class="outputs">
      <div
        v-for="(slot, index) in outputs"
        class="output"
        :class="[{ active: slot.active }, type[index]]"
        :key="'output' + index"
        @mousedown="slotMouseDown($event, index)"
      ></div>
    </div>
  </div>
</template>

<script>
export default {
  name: "VueBlock",
  props: {
    id: {
      type: Number
    },
    name: {
      type: String,
    },
    group: {
      type: String,
    },
    position: {
      type: Array,
      default: () => [],
      validator: function (arr) {
        return (typeof arr[0] === "number") && (typeof arr[1] === "number");
      },
    },
    selected: Boolean,
    title: {
      type: String,
      default: "Title",
    },
    inputs: Array,
    outputs: Array,
    parameters: {
      type: Object,
      default: () => {}
    },
    options: {
      type: Object,
    },
  },
  data: () => ({
    hover: false,
    hasDragged: false,
    type: ["bottom", "right", "left"],
  }),
  created() {
    this.mouseX = 0;
    this.mouseY = 0;

    this.lastMouseX = 0;
    this.lastMouseY = 0;

    this.linking = false;
    this.dragging = false;
  },
  mounted() {
    document.documentElement.addEventListener(
      "mousemove",
      this.handleMove,
      true
    );
    document.documentElement.addEventListener(
      "mousedown",
      this.handleDown,
      true
    );
    document.documentElement.addEventListener("mouseup", this.handleUp, true);
  },
  beforeDestroy() {
    document.documentElement.removeEventListener(
      "mousemove",
      this.handleMove,
      true
    );
    document.documentElement.removeEventListener(
      "mousedown",
      this.handleDown,
      true
    );
    document.documentElement.removeEventListener(
      "mouseup",
      this.handleUp,
      true
    );
  },
  methods: {
    handleMove(e) {
      this.mouseX = e.pageX || e.clientX + document.documentElement.scrollLeft;
      this.mouseY = e.pageY || e.clientY + document.documentElement.scrollTop;

      if (this.dragging && !this.linking) {
        let diffX = this.mouseX - this.lastMouseX;
        let diffY = this.mouseY - this.lastMouseY;

        this.lastMouseX = this.mouseX;
        this.lastMouseY = this.mouseY;

        this.moveWithDiff(diffX, diffY);

        this.hasDragged = true;
      }
    },
    handleDown(e) {
      this.mouseX = e.pageX || e.clientX + document.documentElement.scrollLeft;
      this.mouseY = e.pageY || e.clientY + document.documentElement.scrollTop;

      this.lastMouseX = this.mouseX;
      this.lastMouseY = this.mouseY;

      const target = e.target || e.srcElement;
      if (this.$el.contains(target) && e.which === 1) {

        this.dragging = true;

        this.$emit("select");
        if (e.preventDefault) e.preventDefault();
      }
    },
    handleUp() {
      if (this.dragging) {
        this.dragging = false;

        if (this.hasDragged) {
          this.save();
          this.hasDragged = false;
        }
      }

      if (this.linking) {
        this.linking = false;
      }
    },
    // Slots
    slotMouseDown(e, index) {
      this.linking = true;

      this.$emit("linkingStart", index);
      if (e.preventDefault) e.preventDefault();
    },
    slotMouseUp(e, index) {
      this.$emit("linkingStop", index);
      if (e.preventDefault) e.preventDefault();
    },
    slotBreak(e, index) {
      this.linking = true;

      this.$emit("linkingBreak", index);
      if (e.preventDefault) e.preventDefault();
    },
    save() {
      this.$emit("update");
    },
    deleteBlock() {
      this.$emit("delete");
    },
    moveWithDiff(diffX, diffY) {
      let left = this.position[0] + diffX / this.options.scale;
      let top = this.position[1] + diffY / this.options.scale;

      this.$emit("update:position", [left, top]);
      // this.$emit("update:y", top);
    },
  },
  computed: {
    style() {
      return {
        left: this.options.center.x + this.position[0] * this.options.scale + "px",
        top: this.options.center.y + this.position[1] * this.options.scale + "px",
        width: this.options.width + "px",
        transform: "scale(" + (this.options.scale + "") + ")",
        transformOrigin: "top left",
      };
    },
  },
};
</script>

<style lang="scss" scoped>
$blockBorder: 3px;

$ioPaddingInner: 2px 0;
$ioHeight: 16px;
$ioFontSize: 14px;

$circleBorder: 4px;
$circleSize: 10px;
$circleMargin: 2px; // left/right

$circleNewColor: #00ff00;
$circleRemoveColor: #ff0000;
$circleConnectedColor: #569dcf;

.vue-block {
  position: absolute;
  box-sizing: border-box;
  // border: $blockBorder solid black;
  border-radius: 5px;
  // background: white;
  background: #17212b;
  z-index: 1;
  opacity: 0.9;
  cursor: move;
  height: 50px;

  .hover-over {
    position: absolute;
    top: 0px;
    right: -69px;
    height: 48px;
    background-color: #294c6f;
    border-radius: 5px;
    // width: 70px;
    padding: 10px 0px;
    cursor: context-menu;
    > i {
      font-size: 1.5em;
      margin: 0 5px;
      cursor: pointer;
    }
  }
  .hover-sloy {
    position: absolute;
    top: 0px;
    right: -102px;
    height: 48px;
    background-color: #294c6f;
    border-radius: 5px;
    // width: 70px;
    padding: 10px 0px;
    cursor: context-menu;
    > i {
      font-size: 1.5em;
      margin: 0 5px;
      cursor: pointer;
    }
  }

  > .header {
    background: #bfbfbf;
    text-align: center;
    height: 48px;
    border-radius: 5px;
    color: #000;
    font-size: 0.9em;
    &:hover, &.selected {
      color: rgb(211, 210, 210);
     .parametr {
      color: #3098e7;
    }
    }

    .title {
      white-space: nowrap;
      overflow: hidden; 
      text-overflow: ellipsis;
    }
    .parametr {
      color: #2b5275;
      font-size: 0.8em;
    }

    > .delete {
      color: red;
      cursor: pointer;
      float: right;
      position: absolute;
      right: 3px;
      top: -5px;
      width: 15px;
      height: 15px;
    }

    &.input {
      background: #ffb054;
      border: $blockBorder solid #ffb054;
      &:hover {
        background: none;
        border: $blockBorder solid #ffb054;
      }
      &.selected {
        background: none;
        border: $blockBorder solid #ffb054;
      }
    }
    &.middle {
      background: #89d764;
      border: $blockBorder solid #89d764;
      &:hover {
        background: none;
        border: $blockBorder solid #89d764;
      }
      &.selected {
        background: none;
        border: $blockBorder solid #89d764;
      }
    }
    &.output {
      background: #8e51f2;
      border: $blockBorder solid #8e51f2;
      &:hover {
        background: none;
        border: $blockBorder solid #8e51f2;
      }
      &.selected {
        background: none;
        border: $blockBorder solid #8e51f2;
      }
    }
  }

  .inputs,
  .outputs {
    width: 100%;
    display: flex;
    justify-content: center;
    .input,
    .output {
      position: absolute;
      overflow: hidden;
      font-size: $ioFontSize;
      box-sizing: border-box;
      width: $circleSize;
      height: $circleSize;
      border: $circleBorder solid rgba(0, 0, 0, 0.178);
      border-radius: 100%;
      cursor: crosshair;
      z-index: 999;
      &.active {
        background: $circleConnectedColor;
      }
    }
    .input {
      top: -6px;
      &:hover {
        background: $circleNewColor;
        &.active {
          background: $circleRemoveColor;
        }
      }
    }

    .output {
      &.bottom {
        bottom: -4px;
      }
      &.left {
        left: -6px;
        top: 18px;
      }
      &.right {
        right: -6px;
        top: 18px;
      }
      &:hover {
        background: $circleNewColor;

        &.active {
          background: $circleRemoveColor;
        }
      }
    }
  }
}
</style>

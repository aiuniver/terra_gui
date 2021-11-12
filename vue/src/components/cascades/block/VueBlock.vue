<template>
  <div class="t-block-cascades" :style="style" @mouseover="hover = true" @mouseleave="hover = false">
    <div :class="['t-block-cascades__header', group, { selected: selected }, { error: !!error }]">
      <div class="t-block-cascades__header--title" :title="name">{{ `${id}) ${group}: ${name}` }}</div>
      <div class="t-block-cascades__header--parametr" :title="parametr">{{ }}</div>
    </div>
    <div class="t-block-cascades__base"></div>
    <div v-if="error" v-show="hover || selected" class="t-block-cascades__error">
      {{ error }}
    </div>
    <div v-show="hover || selected" class="t-block-cascades__hover" :style="styleHover">
      <template v-for="(item, i) of iconsFilter">
        <i :class="['t-icon', item.icon]" :key="'icon_' + i" @click="$emit('clickIcons', item)"></i>
      </template>
    </div>

    <div class="t-block-cascades__inputs">
      <div
        v-for="(slot, index) in inputs"
        :key="'input' + index"
        class="input inputSlot"
        :class="{ active: slot.active, 'input--linking-active': linkingCheck && !linking }"
        @mouseup="slotMouseUp($event, index)"
        @mousedown="slotBreak($event, index)"
      ></div>
    </div>
    <div class="t-block-cascades__outputs">
      <div
        v-for="(slot, index) in outputs"
        class="output"
        :class="[{ active: hover && !linkingCheck }, typeLink[index]]"
        :key="'output' + index"
        @mousedown="slotMouseDown($event, index)"
      ></div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'VueBlock',
  props: {
    id: {
      type: Number,
    },
    linkingCheck: {
      type: Object,
    },
    errors: {
      type: Object,
      default: () => {},
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
        return typeof arr[0] === 'number' && typeof arr[1] === 'number';
      },
    },
    selected: Boolean,
    type: String,
    typeLabel: String,
    title: {
      type: String,
      default: 'Title',
    },
    inputs: Array,
    outputs: Array,
    parameters: {
      type: Object,
      default: () => {},
    },
    options: {
      type: Object,
    },
    icons: Array,
    filter: {
      type: Object,
      default: () => {},
    },
    shape: Object,
    bind: Object
  },
  data: () => ({
    hover: false,
    hasDragged: false,
    typeLink: ['bottom', 'right', 'left'],
  }),
  computed: {
    iconsFilter() {
      return this.icons.filter(item => this.filter[this.group].includes(item.event));
    },
    error() {
      return this.errors?.[this.id] || '';
    },
    parametr() {
      const parametr = Object.values(this.parameters?.main || {}).filter(item => item);
      return this.group === 'input' ? this.shape?.input?.join(' ') || '' : parametr.join(' ');
    },
    styleHover() {
      const len = this.iconsFilter.length;
      return { right: -(32 * len) - 3 + 'px' };
    },
    style() {
      return {
        left: this.options.center.x + this.position[0] * this.options.scale + 'px',
        top: this.options.center.y + this.position[1] * this.options.scale + 'px',
        width: this.options.width + 'px',
        transform: 'scale(' + (this.options.scale + '') + ')',
        transformOrigin: 'top left',
        zIndex: this.selected || this.hover ? 10 : 1,
      };
    },
  },
  created() {
    this.mouseX = 0;
    this.mouseY = 0;

    this.lastMouseX = 0;
    this.lastMouseY = 0;

    this.linking = false;
    this.dragging = false;
  },
  mounted() {
    document.documentElement.addEventListener('mousemove', this.handleMove, true);
    document.documentElement.addEventListener('mousedown', this.handleDown, true);
    document.documentElement.addEventListener('mouseup', this.handleUp, true);
  },
  beforeDestroy() {
    document.documentElement.removeEventListener('mousemove', this.handleMove, true);
    document.documentElement.removeEventListener('mousedown', this.handleDown, true);
    document.documentElement.removeEventListener('mouseup', this.handleUp, true);
  },
  methods: {
    getHeight() {
      return this.$el.clientHeight;
    },
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

        this.$emit('select');
        if (e.preventDefault) e.preventDefault();
      }
    },
    handleUp() {
      if (this.dragging) {
        this.dragging = false;

        if (this.hasDragged) {
          this.$emit('moveBlock');
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

      this.$emit('linkingStart', index);
      if (e.preventDefault) e.preventDefault();
    },
    slotMouseUp(e, index) {
      this.$emit('linkingStop', index);
      if (e.preventDefault) e.preventDefault();
    },
    slotBreak(e, index) {
      this.linking = true;

      this.$emit('linkingBreak', index);
      if (e.preventDefault) e.preventDefault();
    },
    save() {
      this.$emit('update');
    },
    moveWithDiff(diffX, diffY) {
      let left = this.position[0] + diffX / this.options.scale;
      let top = this.position[1] + diffY / this.options.scale;
      this.$emit('position', [left, top]);
    },
  },
};
</script>

<style lang="scss" scoped>
$blockBorder: 2px;

$ioPaddingInner: 2px 0;
$ioHeight: 16px;
$ioFontSize: 14px;

$circleBorder: 3px;
$circleSize: 10px;
$circleMargin: 2px; // left/right

$circleNewColor: #00ff003b;
$circleRemoveColor: #ff0000;
$circleConnectedColor: #569dcf;

.t-block-cascades {
  position: absolute;
  box-sizing: border-box;
  // border: $blockBorder solid black;
  border-radius: 5px;
  // background: white;
  background: #17212b;
  z-index: 1;
  opacity: 0.9;
  cursor: move;
  // height: 46px;

  &__base {
    z-index: 1;
    width: 104%;
    height: 100%;
    position: absolute;
    top: 0;
    left: -2%;
  }
  &__hover {
    position: absolute;
    top: 0px;
    right: 0px;
    height: 52px;
    background-color: #294c6f;
    border-radius: 5px;
    cursor: context-menu;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 0px 1px;
    cursor: default;
    > i {
      width: 18px;
      margin: 0 7px;
      cursor: pointer;
    }
  }

  &__error {
    cursor: default;
    position: absolute;
    white-space: break-word;
    left: -201px;
    width: 200px;
    top: 0;
    height: auto;
    padding: 5px 10px;
    color: #fff;
    border: 2px solid red;
    background-color: #2b5278;
    border-radius: 5px;
    font-size: 12px;
    line-height: 1.2;
    text-align: center;
  }

  &__header {
    background: #bfbfbf;
    text-align: center;
    min-height: 52px;
    padding: 0 10px;
    // height: 42px;
    border-radius: 5px;
    color: #000;
    &:hover,
    &.selected {
      color: #fff;
      .parametr {
        color: #3098e7;
      }
    }

    &--title {
      font-size: 14px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    &--parametr {
      color: #2b5275;
      font-size: 11px;
      word-spacing: 3px;
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

    &.InputData {
      background: #54e346;
      border: $blockBorder solid #54e346;
      &:hover {
        border: $blockBorder solid #ffffff;
      }
      &.selected {
        border: $blockBorder solid #ffffff;
      }
    }
    &.Model {
      background: #64c9cf;
      border: $blockBorder solid #64c9cf;
      &:hover {
        border: $blockBorder solid #ffffff;
      }
      &.selected {
        border: $blockBorder solid #ffffff;
      }
    }
    &.Function {
      background: #ff4c29;
      border: $blockBorder solid #ff4c29;
      &:hover {
        border: $blockBorder solid #ffffff;
      }
      &.selected {
        border: $blockBorder solid #ffffff;
      }
    }
    &.Custom {
      background: #ffb740;
      border: $blockBorder solid #ffb740;
      &:hover {
        border: $blockBorder solid #ffffff;
      }
      &.selected {
        border: $blockBorder solid #ffffff;
      }
    }
    &.OutputData {
      background: #ae00fb;
      border: $blockBorder solid #ae00fb;
      &:hover {
        border: $blockBorder solid #ffffff;
      }
      &.selected {
        border: $blockBorder solid #ffffff;
      }
    }
    &.error {
      border: 2px solid red !important;
    }
  }

  &__inputs,
  &__outputs {
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
      // border: $circleBorder solid rgba(0, 0, 0, 0.178);
      // background: #65b9f4;
      border-radius: 100%;
      cursor: crosshair;
      z-index: 999;
      &.active {
        background: $circleConnectedColor;
      }
    }
    .input {
      top: -5px;
      &--linking-active {
        top: 0px;
        width: 100%;
        height: 100%;
        z-index: 20;
        opacity: 0;
      }
    }

    .output {
      &.bottom {
        bottom: -5px;
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
      }
      &.active {
        background: $circleNewColor;
      }
    }
  }
}
</style>

<template>
  <div
    :class="['block', { 'block--selected': selected }, `block--${type}`]"
    :style="style"
    @mouseover="onHover(true)"
    @mouseleave="onHover(false)"
  >
    <div :class="['block__header', type.toLowerCase()]">
      {{ type }}
      <i class="ci-last_page"></i>
      {{ id }}
    </div>
    <div class="block__inputs">
      <div
        v-for="(slot, index) in inputs"
        :key="'input' + index"
        class="input inputSlot"
        :class="{
          active: slot.active,
          'input--linking-active': linkingCheck && !linking,
        }"
        @mouseup="slotMouseUp($event, index)"
        @mousedown="slotBreak($event, index)"
      ></div>
    </div>
    <div class="block__outputs">
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
  name: 'Block',
  props: {
    id: {
      type: Number,
    },
    linkingCheck: {
      type: Object,
    },
    name: {
      type: String,
    },
    position: {
      type: Array,
      default: () => [],
    },
    selected: Boolean,
    type: String,
    inputs: Array,
    outputs: Array,
    options: {
      type: Object,
    },
  },
  data: () => ({
    hover: false,
    hasDragged: false,
    typeLink: ['bottom', 'right', 'left'],
    mouseX: 0,
    mouseY: 0,
    lastMouseX: 0,
    lastMouseY: 0,
    linking: false,
    dragging: false,
  }),
  computed: {
    style() {
      return {
        left: this.options.x + this.position[0] * this.options.scale + 'px',
        top: this.options.y + this.position[1] * this.options.scale + 'px',
        // width: this.options.width + 'px',
        transform: 'scale(' + (this.options.scale + '') + ')',
        transformOrigin: 'top left',
        zIndex: this.selected || this.hover ? 10 : 1,
      };
    },
  },
  mounted() {
    const doc = document.documentElement;
    doc.addEventListener('mousemove', this.onMove, true);
    doc.addEventListener('mousedown', this.onDown, true);
    doc.addEventListener('mouseup', this.onUp, true);
  },
  beforeDestroy() {
    const doc = document.documentElement;
    doc.removeEventListener('mousemove', this.onMove, true);
    doc.removeEventListener('mousedown', this.onDown, true);
    doc.removeEventListener('mouseup', this.onUp, true);
  },
  methods: {
    onHover(value) {
      this.hover = value;
    },
    onMove(e) {
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
    onDown(e) {
      this.mouseX = e.pageX || e.clientX + document.documentElement.scrollLeft;
      this.mouseY = e.pageY || e.clientY + document.documentElement.scrollTop;
      this.lastMouseX = this.mouseX;
      this.lastMouseY = this.mouseY;
      const target = e.target || e.srcElement;
      if (this.$el.contains(target) && e.which === 1) {
        console.log('hasDragged', this.hasDragged);
        console.log('dragging', this.dragging);
        this.dragging = true;
        this.$emit('select');
        if (e.preventDefault) e.preventDefault();
      }
    },
    onUp() {
      console.log('sdsdsdsdsdsd');
      if (this.dragging) {
        this.dragging = false;
        if (this.hasDragged) {
          this.$emit('moveBlock');
          this.hasDragged = false;
        }
      }
      if (this.linking) {
        this.linking = false;
      }
    },
    getH() {
      return {
        width: this.$el?.clientWidth,
        heigth: this.$el?.clientHeight,
      };
    },
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
    moveWithDiff(diffX, diffY) {
      if (this.selected) {
        let left = diffX / this.options.scale;
        let top = diffY / this.options.scale;
        this.$emit('position', { left, top });
      }
    },
  },
};
</script>

<style lang="scss" scoped>
$ioFontSize: 14px;
$circleSize: 10px;
$circleNewColor: #00ff003b;
$circleConnectedColor: #569dcf;
$containCircle: 9px;
$borderCircle: 2px;

@mixin circle {
  content: '';
  width: $containCircle;
  height: $containCircle;
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  border: $borderCircle solid $color-dark;
  border-radius: 50%;
  @content;
}

.block {
  width: 140px;
  height: 50px;
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
  position: relative;

  cursor: default;
  &--selected {
    background: #ac2b2b !important;
    cursor: move;
  }
  &__header {
    min-height: 42px;
    padding: 0 10px;
    border-radius: 5px;
    color: #000;
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
  &--input {
    box-shadow: 0px 0px 4px transparentize($color-orange, 0.25);
    background: $color-orange;
    &::after {
      @include circle {
        bottom: -(($containCircle + $borderCircle) / 2);
        background: $color-orange;
      }
    }
  }
  &--middle {
    background: $color-green;
    box-shadow: 0px 0px 4px transparentize($color-green, 0.25);
    &::after,
    &::before {
      @include circle {
        background: $color-green;
      }
    }
    &::before {
      top: -(($containCircle + $borderCircle) / 2);
    }

    &::after {
      bottom: -(($containCircle + $borderCircle) / 2);
    }
  }
  &--output {
    box-shadow: 0px 0px 4px transparentize($color-pirple, 0.25);
    background: $color-pirple;
    &::before {
      @include circle {
        top: -(($containCircle + $borderCircle) / 2);
        background: $color-pirple;
      }
    }
  }
}
</style>


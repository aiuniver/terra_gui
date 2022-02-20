<template>
  <div
    :class="['block', { 'block--selected': selected || hover }, `block--${type}`, { 'block--error': isError }]"
    :style="style"
    @mouseover="onHover(true)"
    @mouseleave="onHover(false)"
  >
    <div class="block__line-left" :style="setColor"></div>
    <div class="block__line-right" :style="setColor"></div>
    <div :class="['block__wrapper', `block__wrapper--${type}`]" :style="setColor">
      <div class="block__header">
        <div class="block__title text--bold">{{ id }}) {{ name || '' }}</div>
        <div class="block__subtitle text--bold">{{ textSubtitle }}</div>
      </div>
      <div v-if="showTop" class="block__inputs">
        <div
          class="input inputSlot"
          :class="{ 'input--linking-active': linkingCheck && !linking }"
          @mouseup="slotMouseUp($event, 0)"
          @mousedown="slotBreak($event, 0)"
        ></div>
      </div>
    </div>
    <div class="block__part">
      <div v-if="showTop" class="block__point block__point--top" :style="setColor"></div>
      <div v-if="showBottom" class="block__point block__point--bottom" :style="setColor" @mousedown="slotMouseDown($event, 0)"></div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Block',
  props: {
    id: {
      type: Number,
      default: 0,
    },
    linkingCheck: {
      type: Object,
      default: () => ({}),
    },
    typeBlock: {
      type: String,
      default: '',
    },
    color: {
      type: String,
      default: '#6c7883',
    },
    name: {
      type: String,
      default: '',
    },
    position: {
      type: Array,
      default: () => [],
    },
    selected: {
      type: Boolean,
      default: false,
    },
    type: {
      type: String,
      default: '',
    },
    options: {
      type: Object,
      default: () => ({}),
    },
    parameters: {
      type: Object,
      default: () => ({}),
    },
    error: {
      type: String,
      default: '',
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
        boxShadow: `0px 0px 4px ${this.color}75`,
      };
    },
    textSubtitle() {
      let test = '';
      if (['data'].includes(this.type)) test = this?.parameters?.data?.join(' ,') || '';
      if (['input', 'output'].includes(this.type)) test = '';
      if (['handler'].includes(this.type)) test = this?.parameters?.type || '';
      return test;
    },
    isError() {
      return Boolean(this.error);
    },
    setColor() {
      return { backgroundColor: this.color };
    },
    showTop() {
      return ['output', 'middle'].includes(this.typeBlock);
    },
    showBottom() {
      return ['input', 'middle'].includes(this.typeBlock);
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

<style lang="scss">
@import '@/assets/scss/variables/default.scss';
$ioFontSize: 14px;
$circleSize: 15px;
$circleNewColor: #00ff003b;
$circleConnectedColor: #569dcf;
$containCircle: 9px;
$borderCircle: 2px;
$bg-color: #0e1621;
$borderBlock: 2px;

.block {
  user-select: none;
  position: absolute;
  background-color: $bg-color;
  border-radius: 23px 4px 24px 4px;
  // box-shadow: 0px 0px 4px transparentize($color-orange, 0.75);
  &__wrapper {
    position: relative;
    width: 140px;
    height: 50px;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    border-radius: 2px;
    cursor: default;
    margin: 5px;

    clip-path: polygon(13px 0, 100% 0, 100% calc(100% - 13px), calc(100% - 13px) 100%, 0 100%, 0 13px);
    &::after {
      content: '';
      height: 100%;
      width: 100%;
      display: none;
      position: absolute;
      clip-path: polygon(
        13px $borderBlock,
        calc(100% - $borderBlock) $borderBlock,
        calc(100% - $borderBlock) calc(100% - 13px),
        calc(100% - 13px) calc(100% - $borderBlock),
        $borderBlock calc(100% - $borderBlock),
        $borderBlock 13px
      );
      background-color: $bg-color;
      z-index: 1;
      border-radius: 2px;
    }
  }

  &__title {
    text-align: center;
    height: 18px;
    font-size: 12px;
  }
  &__subtitle {
    text-align: center;
    white-space: nowrap;
    text-overflow: ellipsis;
    font-size: 10px;
    color: #6c7883;
  }

  &--selected {
    cursor: move;
    .block__wrapper::after {
      display: block;
    }
    .block__title {
      color: ivory;
    }
  }

  &__header {
    min-height: 42px;
    padding: 0 10px;
    border-radius: 5px;
    color: #000;
    z-index: 3;
  }
  &__inputs,
  &__outputs {
    width: 100%;
    display: flex;
    justify-content: center;
    .input,
    .output {
      position: absolute;
      // overflow: hidden;
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
  }

  &:hover::after {
    display: block;
  }

  &:hover:not(.block--error) {
    .block__title {
      color: ivory;
    }
    .block__point--bottom {
      background-color: rgb(45, 42, 238);
    }
  }

  &__part {
    z-index: 5;
  }
  &__point {
    position: absolute;
    border-radius: 50%;
    width: 11px;
    height: 11px;
    border: 2px solid $bg-color;
    &--top {
      transform: translate(-50%, -50%);
      left: 50%;
      top: 0px;
    }
    &--bottom {
      transform: translate(-50%, 50%);
      left: 50%;
      bottom: 0px;
      cursor: crosshair;
    }
  }
  &__line-left {
    z-index: 0;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    // background-color: $color-orange;
    clip-path: polygon(0 70%, 0% 14px, 14px 0%, 70% 0%, 70% 1px, 14px 1px, 1px 14px, 1px 70%);
  }
  &__line-right {
    z-index: 0;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    // background-color: $color-orange;
    clip-path: polygon(
      100% 30%,
      100% calc(100% - 15px),
      calc(100% - 15px) 100%,
      30% 100%,
      30% calc(100% - 1px),
      calc(100% - 15px) calc(100% - 1px),
      calc(100% - 1px) calc(100% - 15px),
      calc(100% - 1px) 30%
    );
  }
  // &--input {
  //   & .block__point,
  //   & .block__line-left,
  //   & .block__line-right,
  //   & .block__wrapper {
  //     background-color: $color-orange;
  //   }
  //   box-shadow: 0px 0px 4px transparentize($color-orange, 0.75);
  //   border-radius: 23px 4px 24px 4px;
  //   & .block__point--top {
  //     display: none;
  //   }
  // }

  // &--middle {
  //   & .block__point,
  //   & .block__line-left,
  //   & .block__line-right,
  //   & .block__wrapper {
  //     background-color: $color-green;
  //   }
  //   box-shadow: 0px 0px 4px transparentize($color-green, 0.75);
  //   border-radius: 23px 4px 24px 4px;
  // }

  // &--handler {
  //   & .block__point,
  //   & .block__line-left,
  //   & .block__line-right,
  //   & .block__wrapper {
  //     background-color: $color-yello;
  //   }
  //   box-shadow: 0px 0px 4px transparentize($color-yello, 0.75);
  //   border-radius: 23px 4px 24px 4px;
  // }

  // &--output {
  //   & .block__point,
  //   & .block__line-left,
  //   & .block__line-right,
  //   & .block__wrapper {
  //     background-color: $color-pirple;
  //   }
  //   box-shadow: 0px 0px 4px transparentize($color-pirple, 0.75);
  //   border-radius: 23px 4px 24px 4px;
  //   & .block__point--bottom {
  //     display: none;
  //   }
  // }
  &--error {
    .block__line-left,
    .block__line-right {
      background-color: #ca5035;
    }
    .block__title {
      color: #ca5035;
    }
  }
}
</style>


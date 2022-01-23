<template>
  <div :class="['context-menu', { 'context-menu--show': isTrue }]" :style="style" v-outside="onOutside">
    <ul class="menu">
      <li class="menu__item" @click="onClick('add')">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M19 21H5C3.89543 21 3 20.1046 3 19V5C3 3.89543 3.89543 3 5 3H19C20.1046 3 21 3.89543 21 5V19C21 20.1046 20.1046 21 19 21ZM5 5V19H19V5H5ZM13 17H11V13H7V11H11V7H13V11H17V13H13V17Z"
            fill="#2E3A59"
          ></path>
        </svg>
        <span>Add</span>
      </li>
      <li :class="['menu__item', ...one]" title="Hotkey 'Delete'" @click="onClick('delete')">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M17 22H7C5.89543 22 5 21.1046 5 20V7H3V5H7V4C7 2.89543 7.89543 2 9 2H15C16.1046 2 17 2.89543 17 4V5H21V7H19V20C19 21.1046 18.1046 22 17 22ZM7 7V20H17V7H7ZM9 4V5H15V4H9ZM15 18H13V9H15V18ZM11 18H9V9H11V18Z"
            fill="#2E3A59"
          ></path>
        </svg>
        <span>Delete</span>
      </li>
      <li :class="['menu__item']" title="Hotkey 'Ctrl + A'" @click="onClick('select')">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M12 23.27L3 16.27L4.62 15.01L11.99 20.74L19.37 15.001L21 16.27L12 23.27ZM12 19L3 12L4.62 10.74L11.99 16.47L19.37 10.73L21 12L12 19ZM12 14.73L4.63 9.00001L3 7.73001L12 0.730011L21 7.73001L19.36 9.00001L12 14.73Z"
            fill="#2E3A59"
          ></path>
        </svg>
        <span>Select all</span>
      </li>
      <li :class="['menu__item', ...one]" title="Hotkey 'Ctrl + C | Ctrl + V'" @click.prevent="onClick('clone')">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M16 22H4C2.89543 22 2 21.1046 2 20V8H4V20H16V22ZM20 18H8C6.89543 18 6 17.1046 6 16V4C6 2.89543 6.89543 2 8 2H20C21.1046 2 22 2.89543 22 4V16C22 17.1046 21.1046 18 20 18ZM8 4V16H20V4H8ZM15 14H13V11H10V9H13V6H15V9H18V11H15V14Z"
            fill="#2E3A59"
          ></path>
        </svg>
        <span>Clone</span>
      </li>
      <li :class="['menu__item', ...two]" title="Hotkey 'Ctrl + left'" @click="onClick('left')">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M15 19H3V17H15V19ZM21 15H3V13H21V15ZM15 11H3V9H15V11ZM21 7H3V5H21V7Z" fill="#2E3A59"></path>
        </svg>
        <span>Left alignment</span>
      </li>
      <li :class="['menu__item', ...two]" title="Hotkey 'Ctrl + right'" @click="onClick('right')">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M21 19H9V17H21V19ZM21 15H3V13H21V15ZM21 11H9V9H21V11ZM21 7H3V5H21V7Z" fill="#2E3A59"></path>
        </svg>
        <span>Right alignment</span>
      </li>
      <li :class="['menu__item', ...two]" title="Hotkey 'Ctrl + up'" @click="onClick('up')">
        <svg
          style="transform: rotate(90deg)"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M15 19H3V17H15V19ZM21 15H3V13H21V15ZM15 11H3V9H15V11ZM21 7H3V5H21V7Z" fill="#2E3A59"></path>
        </svg>
        <span>Top alignment</span>
      </li>
      <li :class="['menu__item', ...two]" title="Hotkey 'Ctrl + down'" @click="onClick('down')">
        <svg
          style="transform: rotate(270deg)"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M15 19H3V17H15V19ZM21 15H3V13H21V15ZM15 11H3V9H15V11ZM21 7H3V5H21V7Z" fill="#2E3A59"></path>
        </svg>
        <span>Bottom alignment</span>
      </li>

      <li :class="['menu__item', ...two]" @click="onClick('center')">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M18 19H6V17H18V19ZM21 15H3V13H21V15ZM18 11H6V9H18V11ZM21 7H3V5H21V7Z" fill="#2E3A59"></path>
        </svg>
        <span>Center</span>
      </li>
      <li :class="['menu__item', ...two]" title="Hotkey 'Ctrl + Shift + up'" @click="onClick('vertical')">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 22L17 17H13V7H17L12 2L7 7H11V17H7L12 22Z" fill="#2E3A59"></path>
        </svg>
        <span>Vertical</span>
      </li>
      <li :class="['menu__item', ...two]" @click="onClick('horizon')">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M7 11V7L2 12L7 17V13H17V17L22 12L17 7V11H7Z" fill="#2E3A59"></path>
        </svg>
        <span>Horizon</span>
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'context-menu',
  props: {
    x: Number,
    y: Number,
    select: {
      type: Number,
      default: 0,
    },
  },
  computed: {
    isTrue() {
      return this.x && this.y;
    },
    style() {
      return {
        top: `${this.y}px`,
        left: `${this.x - 50}px`,
      };
    },
    isSelect() {
      return Boolean(this.select);
    },
    one() {
      return {
        'menu__item--disabled': this.select < 1,
      };
    },
    two() {
      return {
        'menu__item--disabled': this.select < 2,
      };
    },
  },
  methods: {
    onClick(comm) {
      this.$emit('click', comm);
    },
    onOutside() {
      this.$emit('click', null);
    },
  },
};
</script>

<style lang="scss" scoped>
.context-menu {
  position: absolute;
  top: 50%;
  left: 50%;
  display: none;
  z-index: 10;
  background-color: var(--color-bg-list);
  border-radius: 4px;
  overflow: hidden;
  &--show {
    display: block;
  }
}
.menu {
  margin: 0;
  padding: 0;
  list-style-type: none;
  &__item {
    display: flex;
    align-items: center;
    padding: 5px 15px;
    cursor: pointer;
    &:hover:not(.menu__item--disabled) {
      color: #65b9f4;
      background-color: #1e2734;
    }
    &--disabled {
      opacity: 0.4;
      cursor: default;
    }
    &--hide {
      display: none;
    }
  }
  svg {
    height: 25px;
    width: 25px;
  }
  span {
    margin-left: 10px;
  }
}
</style>
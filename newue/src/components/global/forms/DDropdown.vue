<template>
  <div class="dropdown" v-outside="outside">
    <div class="dropdown__activator">
      <slot name="activator" :on="on"></slot>
      <i class="ci-icon ci-caret_down" @click="click"/>
    </div>
    <div v-show="show" class="dropdown__content" @click="click">
      <slot></slot>
    </div>
  </div>
</template>

<script>
export default {
  name: 'd-dropdown',
  props: {
    list: {
      type: Array,
      dufault: () => [],
    },
  },
  data() {
    return {
      on: {
        click: this.click,
      },
      show: false,
    };
  },
  methods: {
    outside() {
      this.show = false;
    },
    click() {
      this.show = !this.show;
      this.$emit('click', this.show);
    },
  },
};
</script>

<style lang="scss" scoped>
.dropdown {
  position: relative;
  z-index: 100;
  &__activator {
    display: flex;
    align-items: center;
    cursor: pointer;
    i {
      font-size: 18px;
      color: var(--color-light-blue);
    }
  }
  &__content {
    min-width: 100%;
    position: absolute;
    top: calc(100% + 10px);
    left: 0;
  }
}
</style>
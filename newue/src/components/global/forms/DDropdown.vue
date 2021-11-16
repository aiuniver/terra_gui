<template>
  <div class="dropdown" v-outside="outside">
    <div class="dropdown__icon">
      <slot name="activator" :on="on"></slot>
      <i class="ci-icon cl-circle_chevron_right" />
    </div>
    <div v-show="show" class="dropdown__list" @click="click">
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
      show: false
    };
  },
  methods: {
    outside() {
      this.show = false
    },
    click() {
      this.show = !this.show
      this.$emit('click', this.show)
    }
  },
};
</script>

<style lang="scss" scoped>
.dropdown {
  position: relative;
  &__list {
    min-width: 100%;
    position: absolute;
    top: calc(100% + 10px);
    left: 0;
  }
}
</style>
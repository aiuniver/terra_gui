<template>
  <div class="dropdown" :class='{ "t-dropdown--active": show }' v-outside="outside">
    <div class="dropdown__activator">
      <slot name="activator" :on="on"></slot>
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
    // &::before {
    //   position: absolute;
    //   z-index: 1;
    //   top: 0;
    //   right: 3px;
    //   bottom: 0;
    //   width: 0.8rem;
    //   height: 0.4rem;
    //   margin: auto;
    //   content: '';
    //   pointer-events: none;
    //   background: url("data:image/svg+xml,%3Csvg viewBox='0 0 10 6' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0L10 0L5 6L0 0Z' fill='%23777E78'/%3E%3C/svg%3E%0A") no-repeat center center/cover
    // }

  }
  &__content {
    min-width: 100%;
    position: absolute;
    left: 0;
    top: calc(100% + 1rem);
    z-index: 1000;
    width: auto;
    box-shadow: 0 0.3rem 3rem 0 #36363633;
    border: none;
    border-radius: 0;
    overflow: hidden;
    min-height: 4rem;
  }
  &--active {
    .dropdown__activator::before {
      transform: rotate(180deg);
    }
  }
}

</style>
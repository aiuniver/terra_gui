<template>
  <div class="d-modal" v-if="dialog">
    <div class="d-modal__mask"></div>
    <div class="d-modal__inner">
      <div class="d-modal__sloy"></div>
      <div class="d-modal__content">
        <div class="d-modal__header">
          <slot name="header">{{ title }}</slot>
        </div>
        <div class="d-modal__body">
          <slot></slot>
        </div>
        <div class="d-modal__footer">
          <slot name="footer"></slot>
        </div>
      </div>
      <div class="d-modal__close" @click="dialog = false">
        <i class="ci-icon ci-close_big" />
      </div>
      <div class="d-modal__triangle"></div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'd-modal',
  props: {
    title: String,
    value: Boolean,
  },
  data: () => ({}),
  computed: {
    dialog: {
      set(value) {
        this.$emit('input', value);
      },
      get() {
        return this.value;
      },
    },
  },
};
</script>

<style lang="scss" scoped>
.d-modal {
  z-index: 1000;
  &__mask {
    position: fixed;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    background-color: #a7bed320;
  }
  &__inner {
    z-index: 999;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    min-width: 440px;

    // min-height: 600px;
    // top: 50%;
    // left: 50%;
    &:hover {
      .d-modal__sloy {
        // background-color: #65b9f4;
      }
    }
  }
  &__sloy {
    position: absolute;
    width: calc(100% + 2px);
    height: calc(100% + 2px);
    top: -1px;
    left: -1px;
    background-color: #2b5278;
    transition: all 0.3s ease-in-out;
    -webkit-clip-path: polygon(
      63px 100%,
      0% calc(100% - 40px),
      0% 38px,
      38px 0%,
      202px 0%,
      228px 24px,
      calc(100% - 67px) 24px,
      100% 83px,
      100% calc(100% - 15px),
      calc(100% - 15px) 100%
    );
    clip-path: polygon(
      63px 100%,
      0% calc(100% - 40px),
      0% 38px,
      38px 0%,
      202px 0%,
      228px 24px,
      calc(100% - 67px) 24px,
      100% 83px,
      100% calc(100% - 15px),
      calc(100% - 15px) 100%
    );
  }
  &__content {
    padding: 50px;
    background-color: #17212b;
    -webkit-clip-path: polygon(
      63px 100%,
      0% calc(100% - 40px),
      0% 38px,
      38px 0%,
      200px 0%,
      226px 24px,
      calc(100% - 67px) 24px,
      100% 83px,
      100% calc(100% - 15px),
      calc(100% - 15px) 100%
    );
    clip-path: polygon(
      63px 100%,
      0% calc(100% - 40px),
      0% 38px,
      38px 0%,
      200px 0%,
      226px 24px,
      calc(100% - 67px) 24px,
      100% 83px,
      100% calc(100% - 15px),
      calc(100% - 15px) 100%
    );
  }
  &__header {
    font-weight: 600;
    font-size: 16px;
    line-height: 130%;
    color: #f2f5fa;
    margin-bottom: 40px;
  }
  &__body {
    margin-bottom: 30px;
  }
  &__footer {
    width: 100%;
    display: flex;
    div {
      width: 50%;
    }
  }
  &__close {
    position: absolute;
    top: 23px;
    right: 0;
    height: 54px;
    width: 61px;
    background-color: #163952;
    border-top-right-radius: 4px;
    -webkit-clip-path: polygon(100% 0, 0 0, 100% 100%);
    clip-path: polygon(100% 0, 0 0, 100% 100%);
    cursor: pointer;
    &::after {
      position: absolute;
      display: block;
      content: '';
      top: 1px;
      right: 1px;
      height: 50px;
      width: 57px;
      background-color: #65b9f4;
      border-top-right-radius: 4px;
      -webkit-clip-path: polygon(100% 0, 0 0, 100% 100%);
      clip-path: polygon(100% 0, 0 0, 100% 100%);
    }
    i {
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 1;
    }
  }
  &__triangle {
    position: absolute;
    bottom: -1px;
    left: -1px;
    height: 37px;
    width: 57px;
    background-color: #65b9f4;
    border-bottom-left-radius: 4px;
    -webkit-clip-path: polygon(0 100%, 0 0, 100% 100%);
    clip-path: polygon(0 100%, 0 0, 100% 100%);
    &::after {
      position: absolute;
      display: block;
      content: '';
      bottom: 1px;
      left: 1px;
      height: 34px;
      width: 53px;
      background-color: #17212b;
      border-bottom-left-radius: 4px;
      -webkit-clip-path: polygon(0 100%, 0 0, 100% 100%);
      clip-path: polygon(0 100%, 0 0, 100% 100%);
    }
  }
}
</style>
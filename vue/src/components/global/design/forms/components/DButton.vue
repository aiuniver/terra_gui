<template>
  <div :class="['d-button', { 'd-button--disabled': disabled }]" @click="$emit('click', $event)">
    <div
      :class="['d-button__sloy', `d-button__sloy--${color}`, `d-button__sloy--${size}`, `d-button__sloy--${direction}`]"
    ></div>
    <button
      :class="['d-button__btn', `d-button__btn--${color}`, `d-button__btn--${size}`, `d-button__btn--${direction}`]"
    >
      <slot>{{ text }}</slot>
    </button>
  </div>
</template>

<script>
export default {
  name: 'd-button',
  props: {
    disabled: Boolean,
    color: { type: String, default: 'clear' },
    size: { type: String, default: 'medium' },
    direction: { type: String, default: 'right' },
    text: { type: String, default: 'Кнопка' },
  },
};
</script>

<style lang="scss" scoped>
.d-button {
  color: #6c7883;
  margin: 0 0 10px 0;
  position: relative;
  &__btn,
  &__sloy {
    width: fit-content;
    color: inherit;
    border: none;
    border-radius: 0;
    background: none;
    box-shadow: none;
    font-size: 14px;
    line-height: 24px;
    cursor: pointer;
    user-select: none;
    padding: 0;
    border: 1px solid #65b9f400;

    &--medium {
      height: 42px;
    }
    &--small {
      height: 33px;
    }
    &--primary {
      width: calc(100% - 2px);
      color: #0e1621;
      background-color: #65b9f4;
      &.d-button__sloy--right,
      &.d-button__btn--right {
        -webkit-clip-path: polygon(
          0% 100%,
          0% 17px,
          12px 6px,
          78px 6px,
          84px 0%,
          100% 0,
          calc(100% - 8px) 21%,
          calc(100% - 8px) calc(100% - 14px),
          calc(100% - 22px) 100%
        );
        clip-path: polygon(
          0% 100%,
          0% 17px,
          12px 6px,
          78px 6px,
          84px 0%,
          100% 0%,
          calc(100% - 8px) 21%,
          calc(100% - 8px) calc(100% - 14px),
          calc(100% - 22px) 100%
        );
      }
      &.d-button__sloy--left,
      &.d-button__btn--left {
        -webkit-clip-path: polygon(
          0% 0%,
          calc(100% - 84px) 0%,
          calc(100% - 78px) 6px,
          calc(100% - 12px) 6px,
          100% 17px,
          100% 100%,
          22px 100%,
          8px calc(100% - 14px),
          8px 8px
        );
        clip-path: polygon(
          0% 0%,
          calc(100% - 84px) 0%,
          calc(100% - 78px) 6px,
          calc(100% - 12px) 6px,
          100% 17px,
          100% 100%,
          22px 100%,
          8px calc(100% - 14px),
          8px 8px
        );
      }
    }
    &--secondary {
      width: calc(100% - 2px);
      color: #65b9f4;
      background-color: #242f3d;
      &.d-button__btn--right {
        -webkit-clip-path: polygon(
          calc(100% - 1px) 100%,
          108px 100%,
          102px calc(100% - 6px),
          73px calc(100% - 6px),
          15px calc(100% - 6px),
          5px calc(100% - 15px),
          5px 0%,
          calc(100% - 23px) 0%,
          calc(100% - 8px) 14px,
          calc(100% - 8px) calc(100% - 7px)
        );
        clip-path: polygon(
          calc(100% - 1px) 100%,
          108px 100%,
          102px calc(100% - 6px),
          73px calc(100% - 6px),
          15px calc(100% - 6px),
          5px calc(100% - 15px),
          5px 0%,
          calc(100% - 23px) 0%,
          calc(100% - 8px) 14px,
          calc(100% - 8px) calc(100% - 7px)
        );
      }
      &.d-button__sloy--right {
        -webkit-clip-path: polygon(
          100% 100%,
          108px 100%,
          102px calc(100% - 6px),
          75px calc(100% - 6px),
          75px calc(100% - 3px),
          12px calc(100% - 3px),
          0% calc(100% - 15px),
          0% 2px,
          4px 2px,
          4px 0%,
          calc(100% - 23px) 0%,
          calc(100% - 8px) 14px,
          calc(100% - 8px) calc(100% - 8px)
        );
        clip-path: polygon(
          100% 100%,
          108px 100%,
          102px calc(100% - 6px),
          75px calc(100% - 6px),
          75px calc(100% - 3px),
          12px calc(100% - 3px),
          0% calc(100% - 15px),
          0% 2px,
          4px 2px,
          4px 0%,
          calc(100% - 23px) 0%,
          calc(100% - 8px) 14px,
          calc(100% - 8px) calc(100% - 8px)
        );
      }
      &.d-button__sloy--left {
        -webkit-clip-path: polygon(
          0% 100%,
          9px calc(100% - 8px),
          9px 14px,
          22px 0%,
          calc(100% - 4px) 0%,
          calc(100% - 4px) 2px,
          100% 2px,
          100% calc(100% - 15px),
          calc(100% - 14px) calc(100% - 3px),
          calc(100% - 75px) calc(100% - 3px),
          calc(100% - 75px) calc(100% - 7px),
          calc(100% - 102px) calc(100% - 7px),
          calc(100% - 108px) 100%
        );
        clip-path: polygon(
          0% 100%,
          9px calc(100% - 8px),
          9px 14px,
          22px 0%,
          calc(100% - 4px) 0%,
          calc(100% - 4px) 2px,
          100% 2px,
          100% calc(100% - 15px),
          calc(100% - 14px) calc(100% - 3px),
          calc(100% - 75px) calc(100% - 3px),
          calc(100% - 75px) calc(100% - 7px),
          calc(100% - 102px) calc(100% - 7px),
          calc(100% - 108px) 100%
        );
      }
      &.d-button__btn--left {
        -webkit-clip-path: polygon(
          2px 100%,
          9px calc(100% - 6px),
          9px 14px,
          22px 0%,
          calc(100% - 4px) 0%,
          calc(100% - 4px) calc(100% - 15px),
          calc(100% - 14px) calc(100% - 7px),
          calc(100% - 102px) calc(100% - 7px),
          calc(100% - 108px) 100%
        );
        clip-path: polygon(
          2px 100%,
          9px calc(100% - 6px),
          9px 14px,
          22px 0%,
          calc(100% - 4px) 0%,
          calc(100% - 4px) calc(100% - 15px),
          calc(100% - 14px) calc(100% - 7px),
          calc(100% - 102px) calc(100% - 7px),
          calc(100% - 108px) 100%
        );
      }
    }

    &--clear {
      height: 24px;
      &:hover {
        color: #65b9f4;
        border-bottom: 1px solid #65b9f4;
      }
    }
  }
  // &:hover &__sloy {
  //   background-color: #9cc8e7;
  //   color: #9cc8e7;
  // }
  &__sloy {
    display: none;
    position: absolute;
    height: calc(100% + 2px);
    background-color: #65b9f4;
    // width: calc(100% + 2px);
    top: -1px;
    left: -1px;
    // background-color: rgb(44, 31, 104);
    &--medium {
      height: 44px;
    }
    &--small {
      height: 35px;
    }
    &--primary,
    &--secondary {
      width: 100%;
      display: block;
    }
  }
  &--disabled {
    .d-button__btn--clear {
      color: #242f3d;
      cursor: default;
      border-color: #00000000;
    }
    .d-button__btn--primary,
    .d-button__sloy--primary {
      color: #242f3d;
      cursor: default;
      background-color: #396789;
    }
    .d-button__btn--secondary {
      cursor: default;
      background-color: #19222d;
    }
    .d-button__sloy--secondary {
      color: #242f3d;
      cursor: default;
      background-color: #396789;
    }
  }
}
</style>
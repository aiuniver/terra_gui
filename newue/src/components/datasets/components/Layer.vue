<template>
  <div :class="['layer', 'layer-' + type, { 'layer-error': error }]">
    <div class="layer-header">
      <p>{{ title }}</p>
    </div>
    <div class="layer-data">
      <p>{{ data.toString() }}</p>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Layer',
  props: {
    type: {
      type: String,
      default: 'input',
    },
    error: {
      type: String,
      default: '',
    },
    data: {
      type: Array,
      default: () => [],
    },
    title: {
      type: String,
      default: '',
    },
  },
};
</script>

<style lang="scss" scoped>
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

.layer {
  width: 140px;
  height: 50px;
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
  position: relative;

  &:hover {
    // .layer-header {
    //   color: $color-white;
    // }
  }

  &-error {
    background: transparent !important;
    .layer-header {
      color: $color-error !important;
    }
  }

  &-header {
    color: $color-dark;
    font-size: 12px;
    font-weight: bold;
  }

  &-data {
    color: $color-gray;
    margin-top: 5px;
    font-size: 10px;
  }

  &-input {
    box-shadow: 0px 0px 4px transparentize($color-orange, 0.25);
    background: $color-orange;
    &::after {
      @include circle {
        bottom: -(calc(($containCircle + $borderCircle) / 2));
        background: $color-orange;
      }
    }
  }
  &-middle {
    background: $color-green;
    box-shadow: 0px 0px 4px transparentize($color-green, 0.25);
    &::after,
    &::before {
      @include circle {
        background: $color-green;
      }
    }
    &::before {
      top: -(calc(($containCircle + $borderCircle) / 2));
    }

    &::after {
      bottom: -(calc(($containCircle + $borderCircle) / 2));
    }
  }
  &-output {
    box-shadow: 0px 0px 4px transparentize($color-pirple, 0.25);
    background: $color-pirple;
    &::before {
      @include circle {
        top: -(calc(($containCircle + $borderCircle) / 2));
        background: $color-pirple;
      }
    }
  }
}
</style>
<template>
  <div class="panel-content">
    <div class="panel-content__header">
      <h3><slot name="header"></slot></h3>
    </div>
    <div
      :class="[
        'panel-content__inner',
        { 'panel-content__inner--no-padding': noPadding },
        { 'panel-content__inner--no-margin': noMargin },
      ]"
    >
      <slot name="content"></slot>
    </div>
  </div>
</template>

<script>
export default {
  name: 'BasePanelContent',
  props: {
    noPadding: {
      type: Boolean,
      default: false,
    },
    noMargin: {
      type: Boolean,
      default: false,
    },
  },
};
</script>

<style lang="scss" scoped>
.panel {
  &-content {
    margin-bottom: 42px;
    &__header {
      padding: 18px 20px 14px 20px;
      border-top: 1px solid $color-black;
      border-bottom: 1px solid $color-black;
      h3 {
        color: #fff;
        font-family: 'Open Sans', sans-serif;
        font-style: normal;
        font-weight: 600;
        font-size: 14px;
      }
    }
    &__inner {
      margin-top: 30px;
      padding-left: 20px;
      padding-right: 20px;
      &--no-padding {
        padding: 0;
      }
      &--no-margin {
        margin: 0;
      }
    }
    &__inner::v-deep {
      .panel-text-gray {
        color: $color-gray-blue;
      }
    }
  }
}
</style>
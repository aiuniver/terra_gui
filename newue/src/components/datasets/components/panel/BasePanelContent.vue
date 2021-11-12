<template>
  <div class="panel-content">
    <div class="panel-content__header">
      <h3><slot name="header"></slot></h3>
    </div>
    <div class="panel-content__inner">
      <slot name="content"></slot>
    </div>
  </div>
</template>

<script>
export default {
  name: 'BasePanelContent',
};
</script>

<style lang="scss" scoped>
.panel {
  &-content {
    margin-bottom: 42px;
    &__header {
      padding: 10px 20px;
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
    &__inner::v-deep {
      margin-top: 30px;
      padding-left: 20px;
      padding-right: 20px;
      .panel-text-gray {
        color: $color-gray-blue;
      }
    }
  }
}
</style>
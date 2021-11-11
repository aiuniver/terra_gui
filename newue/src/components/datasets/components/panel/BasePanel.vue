<template>
  <div class="panel">
    <slot></slot>
    <div class="panel-footer">
      <BasePanelPagination />
    </div>
  </div>
</template>

<script>
import BasePanelPagination from '@/components/datasets/components/panel/BasePanelPagination';

export default {
  name: 'BasePanel',
  components: {
    BasePanelPagination,
  },
};
</script>

<style lang="scss" scoped>
.panel {
  width: 600px;
  height: 94vh;
  display: flex;
  flex-direction: column;
  background: transparent;
  border-left: 1px solid $color-black;

  &-footer{
    margin-top: auto;
    margin-bottom: 40px;
    padding: 0 20px;
  }
}
</style>
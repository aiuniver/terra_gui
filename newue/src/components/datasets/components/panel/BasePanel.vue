<template>
  <div class="panel">
    <div class="panel__body">
      <scrollbar>
        <slot></slot>
      </scrollbar>
    </div>
    <div class="panel__footer">
      <BasePanelPagination class="mt-5" @action="$emit('action', $event)" />
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
  height: 100%;
  display: flex;
  flex-direction: column;
  border-left: 1px solid $color-black;
  &__body {
    flex: 1 1 auto;
  }

  &__footer {
    flex: 0 0 50px;
    // margin-top: auto;
    // margin-bottom: 40px;
    padding: 0 20px;
  }
}
</style>
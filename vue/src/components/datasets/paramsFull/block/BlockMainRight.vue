<template>
  <div class="block-right">
    <div class="block-right__fab">
      <Fab />
    </div>
    <div class="block-right__header">Выходные параметры</div>
    <div class="block-right__body"></div>
  </div>
</template>

<script>
import Fab from '../components/forms/Fab.vue'
export default {
  name: 'BlockMainRight',
  components: {
    Fab
  }
}
</script>

<style lang="scss" scoped>
.block-right {
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;
  width: 100%;
  position: relative;
  &__header {
    position: absolute;
    height: 24px;
    width: 100%;
    top: 0;
    background: #242f3d;
    font-family: Open Sans;
    font-style: normal;
    font-weight: normal;
    font-size: 12px;
    line-height: 16px;
    display: flex;
    align-items: center;
    text-align: center;
    color: #ffffff;
    padding: 4px 16px;
  }
  &__fab{
    position: absolute;
    right: 16px;
    top: 40px;
  }
}
</style>
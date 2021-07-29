<template>
  <div class="block-file">
    <div class="block-file__header">Выбор папки/файла</div>
    <div class="block-file__body"></div>
  </div>
</template>

<script>
export default {
  name: 'BlockMainRight',
}
</script>

<style lang="scss" scoped>
.block-file {
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
}
</style>
<template>
  <div class="modal" v-if="dialog">
    <div class="modal-close" @click="$emit('close')">
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M8.59 0L5 3.59L1.41 0L0 1.41L3.59 5L0 8.59L1.41 10L5 6.41L8.59 10L10 8.59L6.41 5L10 1.41L8.59 0Z" fill="white"/>
        </svg>
      </div>
    <div class="modal-enging">
      <svg width="441" height="277" viewBox="0 0 441 277" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M226.183 23.387L226.322 23.5H226.5H373.311L440.5 82.7258V262.793L426.793 276.5H63.1469L0.5 236.227V38.2071L37.7071 1H198.822L226.183 23.387Z" fill="#17212B" stroke="#2B5278"/>
        <path d="M0.5 273V240.921L55.3114 276.5H4C2.067 276.5 0.5 274.933 0.5 273Z" stroke="#65B9F4"/>
        <path d="M440.5 35.5598V75.8896L381.319 23.5L428.269 23.5002C435.031 23.5002 440.5 28.9058 440.5 35.5598Z" fill="#65B9F4" stroke="#2B5278"/>
      </svg>
    </div>
    <div class="modal-content">
      <div class="modal-header">
        <h3>Вы собираетесь удалить проект</h3>
      </div>
      <div class="modal-body">
        <div class="modal-body__box">
          <p>Проект будет удален безвозвратно и вы не сможете его восстановить.</p>
        </div>  
        <div class="modal-body__box flex">
          <t-button class="mr-4" @click.native="$emit('close')" cancel :disabled="loading">Отмена</t-button>
          <t-button @click.native="$emit('delete', project), $emit('close')" :loading="loading">Удалить</t-button>
        </div>  
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'NewModalDeleteProject',
  props: ['dialog', 'loading', 'project'],
};
</script>

<style lang="scss" scoped>
.modal{
  z-index: 5;
  width: 441px;
  height: 277px;
  position: absolute;
  top:50%;
  left:50%;
  transform: translate(-50%, -50%);
  &-enging{
    position: relative;
  }

  &-close{
    position: absolute;
    top: 23px;
    right: 0px;
    width: 40px;
    display: flex;
    z-index: 10;
    justify-content: center;
    align-items: center;
    height: 36px;
    cursor: pointer;
  }

  &-content{
    width: 320px;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    cursor: pointer;
  }
  &-body{
    &__box{
      p{
        font-family: "Open Sans";
        font-size: 12px;
        color: #A7BED3;
      }
      margin-top: 20px;
    }
  }
}
</style>
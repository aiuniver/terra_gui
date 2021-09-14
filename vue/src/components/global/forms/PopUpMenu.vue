<template>
    <ul class="menu" style="margin-left: 30px; margin-top: -24px">
      <li class="menu__item">Выбрать все</li>
      <li class="menu__item">Инвертировать</li>
      <li class="menu__item">Удалить</li>
    </ul>
</template>

<script>
export default {
  name: "PopUpMenu"
}
</script>

<style scoped lang="scss">
.menu{
  position: absolute;
  top: 24px;
  left: 35px;
  background: #242F3D;
  border: 1px solid #6C7883;
  box-sizing: border-box;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.25);
  border-radius: 4px;
  z-index: 105;
  &__item{
    width: 100%;
    padding: 0 5px;
    &:hover{
      background: #2B5278;
    }
  }
  &__dropdown{
    padding: 0 5px;
    display: flex;
    .menu{
      visibility: hidden;
      opacity: 0;
    }
    li{
      position: relative;
    }
    i{
      margin-left: auto;
    }
    &:hover{
      background: #2B5278;
      .menu{
        visibility: visible;
        opacity: 1;
      }
    }
  }
}
</style>
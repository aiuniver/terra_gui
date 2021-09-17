<template>
    <ul class="menu">
      <li class="menu__item">Добавить новый график</li>
      <li class="menu__item">Копировать график</li>
      <li class="menu__item">Свернуть график</li>
      <li class="menu__item">Удалить график</li>
      <li class="menu__dropdown">
        <span>Показать данные</span>
        <i :class="['t-icon', 'icon-training-dropdown']"></i>
        <ul class="menu" style="margin-left: -0px; margin-top: -4px">
          <li class="menu__item">Accuracy</li>
          <li class="menu__item">CategoricalHinge</li>
          <li class="menu__item">KLDivergence</li>
        </ul>
      </li>
      <li class="menu__dropdown">Показать метрику</li>
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
  top: 0;
  left: -200px;
  background: #242F3D;
  border: 1px solid #6C7883;
  box-sizing: border-box;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.25);
  border-radius: 4px;
  width: 200px;
  z-index: 1;
  &__item{
    user-select: none;
    width: 100%;
    padding: 0 10px;
    font-size: 14px;
    &:hover{
      background: #2B5278;
    }
  }
  &__dropdown{
    padding: 0 10px;
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
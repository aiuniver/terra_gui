<template>
    <ul class="menu" style="margin-left: 30px; margin-top: -24px">
      <li class="menu__item">Добавить новый график</li>
      <li class="menu__item">Копировать график</li>
      <li class="menu__item">Свернуть график</li>
      <li class="menu__item">Удалить график</li>
      <li class="menu__dropdown">
        <span>Показать данные</span>
        <i :class="['t-icon', 'icon-training-dropdown']"></i>
        <ul class="menu" style="margin-left: 196px; margin-top: -4px">
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
  background: #242F3D;
  border: 1px solid #6C7883;
  box-sizing: border-box;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.25);
  border-radius: 4px;
  z-index: 1;
  &__item{
    width: 100%;
    padding: 5px;
    &:hover{
      background: #2B5278;
    }
  }
  &__dropdown{
    padding: 5px;
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
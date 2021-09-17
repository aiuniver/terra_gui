<template>
  <ul class="menu">
    <li class="menu__item" @click="$emit('menu-click', 'add')">Добавить новый график</li>
    <li class="menu__item" @click="$emit('menu-click', 'copy')">Копировать график</li>
    <li v-if="data.length" class="menu__dropdown">
      <i :class="['t-icon', 'icon-training-dropdown']"></i>
      <span>Показать данные</span>
      <ul class="menu">
        <li class="menu__item" v-for="(item, idx) in data" :key="idx" @click="$emit('menu-click', item)">{{ item }}</li>
      </ul>
    </li>
    <li v-if="metrics.length" class="menu__dropdown">
      <i :class="['t-icon', 'icon-training-dropdown']"></i>
      <span>Показать метрики</span>
      <ul class="menu">
        <li class="menu__item" v-for="(item, idx) in metrics" :key="idx" @click="$emit('menu-click', item)">
          {{ item }}
        </li>
      </ul>
    </li>
  </ul>
</template>

<script>
export default {
  name: 'PopUpMenu',
  props: {
    data: {
      type: Array,
      default: () => [],
    },
    metrics: {
      type: Array,
      default: () => [],
    },
  },
};
</script>

<style scoped lang="scss">
.menu {
  position: absolute;
  top: 0;
  right: 100%;
  background: #242f3d;
  border: 1px solid #6c7883;
  box-sizing: border-box;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.25);
  border-radius: 4px;
  width: max-content;
  z-index: 10;
  font-size: 14px;

  &__item {
    user-select: none;
    cursor: pointer;
    width: 100%;
    padding: 0 10px;
    text-align: right;
    height: 24px;
    display: flex;
    flex-direction: row;
    align-items: center;
    &:hover {
      background: #2b5278;
    }
  }

  &__dropdown {
    padding: 0 10px 0 0;
    display: flex;
    position: relative;
    user-select: none;
    cursor: default;
    justify-content: space-between;
    display: flex;
    flex-direction: row;
    align-items: center;
    .menu {
      visibility: hidden;
    }
    &:hover {
      background: #2b5278;
    }
    &:hover > .menu {
      visibility: visible;
    }
    li {
      position: relative;
    }
    i {
      transform: rotateY(180deg);
    }
  }
}
</style>
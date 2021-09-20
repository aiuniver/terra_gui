<template>
  <ul class="menu">
    <!-- <li class="menu__item" @click="$emit('event', 'add')">Добавить новый график</li> -->
    <li class="menu__item" @click="$emit('event', { name: 'general', data: 'hide' })">Свернуть</li>
    <li class="menu__item" @click="$emit('event', { name: 'general', data: 'remove' })">Удалить</li>
    <template v-for="({ name, list }, i) of menus">
      <li class="menu__dropdown" :key="'menu_' + i">
        <i :class="['t-icon', 'icon-training-dropdown']"></i>
        <span>{{ name }}</span>
        <ul class="menu">
          <li
            class="menu__item"
            v-for="({ title, event }, idx) in list"
            :key="`list_${i}_${idx}`"
            @click="$emit('event', event)"
          >
            {{ title }}
          </li>
        </ul>
      </li>
    </template>
  </ul>
</template>

<script>
export default {
  name: 'PopUpMenu',
  props: {
    menus: {
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
    justify-content: flex-end;
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
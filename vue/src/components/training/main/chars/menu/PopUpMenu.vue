<template>
  <ul class="menu">
    <!-- <li class="menu__item" @click="$emit('event', 'add')">Добавить новый график</li> -->
    <li class="menu__item" @click="$emit('event', { name: 'general', data: 'add' })">Добавить график</li>
    <li class="menu__item" @click="$emit('event', { name: 'general', data: 'copy' })">Копировать график</li>
    <li class="menu__item" @click="$emit('event', { name: 'general', data: 'hide' })">
      {{ show ? 'Свернуть' : 'Развернуть' }} график
    </li>
    <li class="menu__item" @click="$emit('event', { name: 'general', data: 'remove' })">Удалить график</li>
    <li class="menu__dropdown">
      <i :class="['t-icon', 'icon-training-dropdown']"></i>
      <span>Показывать данные</span>
      <ul class="menu">
        <li class="menu__item" @click="$emit('event', { name: 'data', data: 'model' })">По всей модели</li>
        <li v-if="isClass" class="menu__item" @click="$emit('event', { name: 'data', data: 'classes' })">По классам</li>
      </ul>
    </li>
    <li v-if="type !== 'loss_graphs'" class="menu__dropdown">
      <i :class="['t-icon', 'icon-training-dropdown']"></i>
      <span>Показывать метрики</span>
      <ul class="menu" style="max-height: 240px; overflow-y:auto;">
        <template v-for="(item, i) of metrics">
          <li class="menu__item" :key="'metrics' + i" @click="$emit('event', { name: 'metric', data: item })">
            {{ item }}
          </li>
        </template>
      </ul>
    </li>
    <li class="menu__dropdown">
      <i :class="['t-icon', 'icon-training-dropdown']"></i>
      <span>Показывать выход</span>
      <ul class="menu" style="max-height: 240px; overflow-y:auto;">
        <template v-for="(item, i) of exits">
          <li class="menu__item" :key="'output' + i" @click="$emit('event', { name: 'chart', data: item })">
            Выход {{ item }}
          </li>
        </template>
      </ul>
    </li>
  </ul>
</template>

<script>
export default {
  name: 'PopUpMenu',
  props: {
    menus: {
      type: Object,
      default: () => {},
    },
    settings: {
      type: Object,
      default: () => {},
    },
    show: Boolean,
  },
  computed: {
    show_metric() {
      return this.settings?.show_metric || '';
    },
    id() {
      return this.settings?.output_idx || 0;
    },
    isClass() {
      const clas = this.menus?.isClass || {};
      return clas[this.id];
    },
    type() {
      return this.menus?.type || '';
    },
    outputs() {
      return this.menus?.outputs || [];
    },
    exits() {
      return this.outputs.map(item => item.id); //.filter(item => item !== this.id);
    },
    metrics() {
      const metrics = this.outputs.find(item => item.id === this.id)?.metrics || [];
      return metrics; //.filter(item => item !== this.show_metric);
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
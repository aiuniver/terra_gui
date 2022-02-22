<template>
  <div class="menu">
    <div class="menu-list">
      <div @click="$emit('select', 0)" :class="['menu-list__item', { 'menu-list__item--selected': selectedType === 0 }]">
        <d-svg name="clock" />
        <span>Недавние</span>
      </div>
      <div @click="$emit('select', 1)" :class="['menu-list__item', { 'menu-list__item--selected': selectedType === 1 }]">
        <d-svg name="file-outline" />
        <span>Проектные</span>
      </div>
      <div @click="$emit('select', 2)" :class="['menu-list__item', { 'menu-list__item--selected': selectedType === 2 }]">
        <d-svg name="world" />
        <span>Terra</span>
      </div>
    </div>
    <hr />
    <scrollbar :ops="{ rail: { gutterOfSide: '0px' } }">
      <ul class="menu-categories" v-for="tag in tags" :key="tag.name">
        <p 
        :class="{ 'menu-categories--selected': selectedTag.name === tag.name }"
        @click="$emit('tagClick', { type: 'group', name: tag.name })">{{ tag.name }}</p>
        <li
        v-for="item in tag.items" :key="item"
        :class="{ 'menu-categories--selected': selectedTag.name === item && selectedTag.group === tag.name }"
        @click="$emit('tagClick', { type: 'tag', name: item, group: tag.name })"
        class="menu-categories__item"
        >{{ item }}</li>
      </ul>
    </scrollbar>
  </div>
</template>

<script>
export default {
  name: 'choice-menu',
  props: ['selectedType', 'tags', 'selectedTag']
};
</script>

<style lang="scss">
@import "@/assets/scss/variables/default.scss";

.menu {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.menu-list {
  &__item {
    display: flex;
    align-items: center;

    padding: 10px 0 10px 20px;
    color: $color-gray-blue;
    font-size: 14px;
    cursor: pointer;

    span {
      margin-left: 15px;
    }

    &--selected {
      background-color: $color-black;
    }

    &:hover {
      color: #65b9f4;
      .ci-world {
        background-color: #65b9f4;
      }
    }
    &.selected {
      background-color: var(--color-light-gray);
    }
    input {
      display: none;
    }
  }
}

.menu-categories {
  font-size: 14px;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 15px;
  &--selected, & > *:hover {
    background: #1E2734;
    color: #65B9F4 !important;
  }
  p {
    font-weight: bold;
    cursor: pointer;
    padding-left: 20px;
    font-size: 14px;
  }
  &__item {
    padding-left: 20px;
    color: $color-gray;
    margin-top: 5px;
    cursor: pointer;
  }
}
</style>
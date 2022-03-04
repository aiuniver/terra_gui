<template>
  <div class="menu">
    <div class="menu-list">
      <div @click="select(1)" 
      :class="['menu-list__item', { 'menu-list__item--selected': selectedType === 1 }]">
        <d-svg name="clock" />
        <span>Недавние</span>
      </div>
      <div @click="select(2)" 
      :class="['menu-list__item', { 'menu-list__item--selected': selectedType === 2 }]">
        <d-svg name="file-outline" />
        <span>Проектные</span>
      </div>
      <div @click="select(3)" 
      :class="['menu-list__item', { 'menu-list__item--selected': selectedType === 3 }]">
        <d-svg name="world" />
        <span>Terra</span>
      </div>
    </div>
    <hr />
    <scrollbar :ops="{ rail: { gutterOfSide: '0px' } }">
      <ul class="menu-categories" v-for="(tag, idx) in tags" :key="tag.alias">
        <p 
        :class="{ 'menu-categories--selected': selectedTag.alias === tag.alias && selectedTag.group === tag.group,
          'menu-categories--selected-group': idx === selectedTag.idx
        }"
        @click="$emit('tagClick', { type: 'group', alias: tag.alias, name: tag.name, idx })">{{ tag.name }}</p>
        <transition-group name="slide-fade">
          <template v-if="idx === selectedTag.idx">
            <li
            v-for="item in tag.items" :key="item"
            :class="{ 'menu-categories--selected': selectedTag.alias === item && selectedTag.group === tag.alias }"
            @click="$emit('tagClick', { type: 'tag', alias: item, group: tag.alias, idx })"
            class="menu-categories__item"
            >{{ item }}</li>
          </template>
        </transition-group>
      </ul>
    </scrollbar>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'

export default {
  name: 'choice-menu',
  props: ['selectedType', 'selectedTag'],
  computed: {
    ...mapGetters({
      tags: 'datasets/getTags'
    })
  },
  methods: {
    select(val) {
      if (this.selectedType === val) return this.$emit('select', 0)
      this.$emit('select', val)
    }
  }
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
      background-color: #1E2734;
      color: #65b9f4;
    }
    &:hover {
      color: #65b9f4;
      background-color: #1E2734;
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
  margin-bottom: 5px;
  &--selected, &__item:hover, & p:hover {
    background: #1E2734;
    color: #65B9F4 !important;
  }
  &--selected-group {
    color: #65B9F4 !important;
  }
  p {
    // font-weight: bold;
    cursor: pointer;
    padding: 5px 10px 5px 20px;
    font-size: 14px;
  }
  &__item {
    padding: 5px 0 5px 25px;
    color: $color-gray;
    cursor: pointer;
  }
}

.slide-fade-enter-active {
  transition: all .3s ease;
}

.slide-fade-enter, .slide-fade-leave-to {
  transform: translateY(-10px);
  opacity: 0;
}
</style>
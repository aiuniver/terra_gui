<template>
  <div>
    <div class="menu-list">
      <div @click="selectedType = 0" :class="['menu-list__item', { 'menu-list__item--selected': selectedType === 0 }]">
        <d-svg name="clock" />
        <span>Недавние</span>
      </div>
      <div @click="selectedType = 1" :class="['menu-list__item', { 'menu-list__item--selected': selectedType === 1 }]">
        <d-svg name="file-outline" />
        <span>Проектные</span>
      </div>
      <div @click="selectedType = 2" :class="['menu-list__item', { 'menu-list__item--selected': selectedType === 2 }]">
        <d-svg name="world" />
        <span>Terra</span>
      </div>
    </div>
    <hr />
    <div class="menu-categories">
      <ul class="menu-categories__item">
        <li>Изображения</li>
        <li>Машины</li>
        <li>Круглые</li>
        <li>Самолеты</li>
        <li>Квадраты</li>
      </ul>
      <ul class="menu-categories__item">
        <li>Видео</li>
        <li>Машины</li>
        <li>Круглые</li>
        <li>Самолеты</li>
        <li>Квадраты</li>
      </ul>
      <ul class="menu-categories__item">
        <li>Текст</li>
        <li>Машины</li>
        <li>Круглые</li>
        <li>Самолеты</li>
        <li>Квадраты</li>
      </ul>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';

export default {
  name: 'choice-menu',
  data: () => ({
    selectedType: 2,
  }),
  computed: {
    ...mapGetters({
      datasets: 'datasets/getDatasets',
      choiceDataset: 'datasets/choiceDataset',
    }),
  },
};
</script>

<style lang="scss">
@import "@/assets/scss/variables/default.scss";
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
  padding-left: 20px;
  &__item {
    margin-bottom: 10px;
    li {
      &:first-child {
        padding-bottom: 5px;
        color: $color-white;
        cursor: default;
      }
      color: $color-gray;
      padding-top: 5px;
      cursor: pointer;
    }
  }
}
</style>
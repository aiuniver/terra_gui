<template>
  <div class="panel-pagination flex align-center">
    <button @click="$emit('action', 'prev')" class="btn panel-pagination-btn">
      <SvgContainer name="arrow-left-longer-big" />
    </button>
    <div class="panel-pagination-container">
      <div class="panel-pagination-container__list flex align-center">
        <div class="panel-pagination-container__list-item"></div>
        <div class="panel-pagination-container__list-item panel-pagination-container__list-item--active"></div>
        <div class="panel-pagination-container__list-item"></div>
        <div class="panel-pagination-container__list-item"></div>
      </div>
      <div class="panel-pagination-container__header">
        <span>Входные данные</span>
      </div>
    </div>
    <d-button @click="$emit('action', 'next')" style="width: 40%" color="secondary" direction="left" text="Далее" />
  </div>
</template>

<script>
import SvgContainer from '@/components/app/SvgContainer';

export default {
  components: {
    SvgContainer,
  },
};
</script>

<style lang="scss" scoped>
.panel-pagination {
  justify-content: space-between;
  &-btn {
    background: $color-dark-gray;
    border-radius: 4px;
    transition: background 0.3s ease;
    border: none;
    width: 36px;
    height: 36px;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    &:hover {
      background: $color-black;
      box-shadow: 0px 0px 4px rgba(101, 185, 244, 0.2);
    }
  }
  &-container {
    margin: 0 10px;
    text-align: center;
    width: 200px;
    &__header {
      margin-top: 15px;
    }
    &__list {
      justify-content: center;
      &-item {
        background: $color-gray-blue;
        border-radius: 50%;
        display: block;
        margin: 0 auto;
        margin-right: 25px;
        width: 5px;
        height: 5px;
        &--active {
          width: 10px;
          height: 10px;
          background: $color-light-blue;
        }
      }
    }
  }
}
</style>
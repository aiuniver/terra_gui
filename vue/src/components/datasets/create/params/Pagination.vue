<template>
  <div class="d-pagination">
    <button class="d-pagination__btn" :disabled="isDisabled" @click="$emit('prev', $event)">
      <d-svg name="arrow-carret-left-longer-big" />
    </button>
    <div class="d-pagination__inner">
      <div class="d-pagination__list">
        <div v-for="item of qty" :key="item" :class="['d-pagination__item', { 'd-pagination__item--active': isActive(item) }]"></div>
      </div>
      <div class="d-pagination__title">
        <span>{{ title }}</span>
      </div>
    </div>
    <d-button style="width: 40%" color="secondary" direction="left" text="Далее" :disabled="isStatus" @click="$emit('next', $event)" />
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
export default {
  name: 'DPagination',
  props: {
    qty: {
      type: Number,
      default: 4,
    },
    title: {
      type: String,
      default: '',
    },
    value: {
      type: Number,
      default: 0,
    },
  },
  data: () => ({}),
  computed: {
    ...mapGetters({
      select: 'createDataset/getSelectSource',
    }),
    isDisabled() {
      return this.value === 1;
    },
    isStatus() {
      console.log(this.select);
      if (this.value === 1 && !this?.select?.value) return true;
      return false;
    },
  },
  methods: {
    isActive(value) {
      return this.value === value;
    },
  },
};
</script>

<style lang="scss">
@import '@/assets/scss/variables/default.scss';
.d-pagination {
  display: flex;
  align-items: center;
  justify-content: space-between;
  &__btn {
    background: $color-dark-gray;
    border-radius: 4px;
    transition: background 0.3s ease;
    border: none;
    width: 36px;
    height: 36px;
    display: flex;
    justify-content: center;
    align-items: center;
    &:hover:not(.d-pagination__btn:disabled) {
      cursor: pointer;
      background: $color-black;
      box-shadow: 0px 0px 4px rgba(101, 185, 244, 0.2);
    }
    &:disabled {
      opacity: 0.4;
      cursor: default;
    }
  }
  &__list {
    display: flex;
    justify-content: center;
    height: 10px;
    margin-bottom: 5px;
    align-items: center;
  }
  &__item {
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
</style>
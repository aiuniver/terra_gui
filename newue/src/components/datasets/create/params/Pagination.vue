<template>
  <div class="d-pagination">
    <button @click="onPrev" class="d-pagination__btn">
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
    <DButton @click="onNext" style="width: 40%" color="secondary" direction="left" text="Далее" />
  </div>
</template>

<script>
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
  computed: {},
  methods: {
    onNext() {
      if (this.value < this.qty) this.$emit('input', this.value + 1);
    },
    onPrev() {
      if (this.value > 1) this.$emit('input', this.value - 1);
    },
    isActive(value) {
      return this.value === value;
    },
  },
};
</script>

<style lang="scss">
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
    cursor: pointer;
    &:hover {
      background: $color-black;
      box-shadow: 0px 0px 4px rgba(101, 185, 244, 0.2);
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
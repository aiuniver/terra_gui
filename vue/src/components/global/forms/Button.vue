<template>
  <button :class="['t-button', { 't-button--cancel': cancel }]" :disabled="disabled" @click="click">
    <i v-if="loading" class="t-btn__loading t-icon icon-loading-new"></i>
    <span v-else class="t-btn__text">
      <slot>Загрузить</slot>
    </span>
  </button>
</template>

<script>
export default {
  name: 't-button',
  props: {
    loading: Boolean,
    cancel: Boolean,
    disabled: Boolean,
  },
  methods: {
    click(e) {
      if (!this.loading && !this.disabled) {
        this.$emit('click', e);
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-button {
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 0.875rem;
  width: 100%;
  color: #fff;
  background-color: #2b5278;
  box-shadow: 0 1px 3px 0 rgb(0 133 255 / 50%);
  height: 40px;
  line-height: 1;
  border: 1px solid #65b9f4;
  border-radius: 4px;
  cursor: pointer;
  user-select: none;
  transition: border-color 0.3s ease-in-out, box-shadow 0.3s ease-in-out, background-color 0.3s ease-in-out,
    color 0.3s ease-in-out;
  &:hover {
    border-color: #2b5278;
    box-shadow: 0 1px 3px 0 rgb(0 0 0 / 50%);
  }
  &:disabled {
    cursor: auto;
    color: #242f3d;
    border-color: #2b5278;
    background-color: #6c7883;
    box-shadow: 0 1px 3px 0 rgb(0 133 255 / 50%);
  }
  &__loading {
    display: inline-block;
    width: 30px;
    height: 30px;
    margin-right: 10px;
  }
  &__text {
    font-size: 0.875rem;
    font-family: inherit;
    font-size: inherit;
    color: inherit;
    line-height: inherit;
  }
  &--cancel {
    background-color: #17212b;
    &:hover {
      background-color: #2b5278;
      border-color: #2b5278;
      box-shadow: 0 1px 3px 0 rgb(0 0 0 / 50%);
    }
  }
}
</style>